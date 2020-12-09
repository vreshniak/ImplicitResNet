import time
import math
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla
import scipy.optimize as opt
import matplotlib.pyplot as plt
from pathlib import Path

from abc import ABCMeta, abstractmethod

import torch
from torch.autograd import Function
from torch.nn import Linear, ReLU, Conv2d, Module, Sequential
from torch.nn.functional import linear, conv2d, conv_transpose2d
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import utils
# from jacobian import JacobianReg



###############################################################################
###############################################################################


# global parameters
_TOL = 1.e-6
_linTOL = 1.e-5
_max_iters = 200

_collect_stat  = True
_forward_stat  = {}
_backward_stat = {}

# _dtype  = torch.float
# _device = torch.device("cpu")

_forward_scipy  = False
_backward_scipy = False

# _to_numpy = lambda x, dtype: x.cpu().detach().numpy().astype(dtype)


###############################################################################
###############################################################################


# class neumann_backprop(Function):
# 	@staticmethod
# 	def forward(ctx, y, y_fp):
# 		# ctx.obj = obj
# 		ctx.save_for_backward(y, y_fp)
# 		return y

# 	@staticmethod
# 	def backward(ctx, dy):
# 		y, y_fp, = ctx.saved_tensors

# 		# residual = lambda dx: (dx-A_dot(dx)-dy).flatten().norm() # \| (I-A) * dx - dy \|
# 		A_dot    = lambda x: torch.autograd.grad(y_fp, y, grad_outputs=x, retain_graph=True, only_inputs=True)[0]
# 		residual = lambda Adx: (Adx-dy).reshape((dy.size()[0],-1)).norm(dim=1).max() #.flatten().norm() # \| (I-A) * dx - dy \|

# 		tol = atol = torch.tensor(_TOL)
# 		TOL = torch.max(tol*dy.norm(), atol)

# 		#######################################################################
# 		# Neumann series

# 		dx  = dy
# 		Ady = A_dot(dy)
# 		Adx = Ady
# 		r1  = residual(dx-Adx)
# 		neu_iters = 1
# 		while r1>=TOL and neu_iters<_max_iters:
# 			r0  = r1
# 			dx  = dx + Ady
# 			Ady = A_dot(Ady)
# 			Adx = Adx + Ady
# 			r1  = residual(dx-Adx)
# 			neu_iters += 1
# 			assert r1<r0, "Neumann series hasn't converged at iteration "+str(neu_iters)+" out of "+str(_max_iters)+" max iterations"

# 		if _collect_stat:
# 			global _backward_stat
# 			_backward_stat['steps']        = _backward_stat.get('steps',0) + 1
# 			_backward_stat['neu_residual'] = _backward_stat.get('neu_residual',0) + r1
# 			_backward_stat['neu_iters']    = _backward_stat.get('neu_iters',0) + neu_iters
# 		return None, dx



class linsolve_backprop(Function):
	@staticmethod
	def forward(ctx, self, y, y_fp, y_fp2):
		ctx.mark_non_differentiable(y)
		ctx.save_for_backward(y, y_fp2)
		ctx.self = self
		return y_fp

	@staticmethod
	def backward(ctx, dy):
		y, y_fp, = ctx.saved_tensors
		ndof  = y.nelement()
		batch_dim = dy.size()[0]

		ctx.residual  = 1
		ctx.lin_iters = 0

		# freeze parameters so that .backward() does not propagate corresponding gradients
		ctx.self.rhs.requires_grad_(False)

		start = time.time()
		if _backward_scipy:
			tol = atol = torch.tensor(_linTOL)
			TOL = torch.max(tol*dy.norm(), atol)

			A_dot       = lambda x: torch.autograd.grad(y_fp, y, grad_outputs=x, create_graph=False, retain_graph=True, only_inputs=True)[0]
			residual_fn = lambda Adx: (Adx-dy).reshape((dy.size(0),-1)).norm(dim=1).max() # \| (I-A) * dx - dy \|

			#######################################################################

			# torch to numpy dtypes
			numpy_dtype = {torch.float: np.float, torch.float32: np.float32, torch.float64: np.float64}

			# 'Matrix-vector' product of the linear operator
			def matvec(v):
				ctx.lin_iters = ctx.lin_iters + 1
				v0 = torch.from_numpy(v).view_as(y).to(device=dy.device, dtype=dy.dtype)
				Av = v0 - A_dot(v0)
				return Av.cpu().detach().numpy().ravel()
			A = sla.LinearOperator(dtype=numpy_dtype[dy.dtype], shape=(ndof,ndof), matvec=matvec)


			# Note that norm(residual) <= max(tol*norm(b), atol. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lgmres.html
			dx, info = sla.lgmres( A, dy.cpu().detach().numpy().ravel(), x0=dy.cpu().detach().numpy().ravel(), maxiter=_max_iters, tol=TOL, atol=atol, M=None )
			dx = torch.from_numpy(dx).view_as(dy).to(device=dy.device, dtype=dy.dtype)

			ctx.residual = residual_fn(dx-A_dot(dx)).detach()
		else:
			A_dot = lambda x: torch.autograd.grad(y_fp, y, grad_outputs=x, create_graph=True, only_inputs=True)[0] # need create_graph to find it's derivative

			# initial condition
			dx = dy.clone().detach().requires_grad_(True)

			nsolver = torch.optim.LBFGS([dx], lr=1, max_iter=_max_iters, max_eval=None, tolerance_grad=1.e-9, tolerance_change=1.e-9, history_size=5, line_search_fn='strong_wolfe')
			def closure():
				nsolver.zero_grad()
				residual = ( dx - A_dot(dx) - dy ).pow(2).sum() / batch_dim
				if residual>_linTOL**2:
				# if residual>_linTOL:
					residual.backward(retain_graph=True)
				ctx.residual = torch.sqrt(residual)
				ctx.lin_iters += 1
				return ctx.residual
			nsolver.step(closure)
			dx = dx.detach()
		stop = time.time()

		# unfreeze parameters
		ctx.self.rhs.requires_grad_(True)

		if _collect_stat:
			ctx.self._stat['backward/steps']        = ctx.self._stat.get('backward/steps',0)        + 1
			ctx.self._stat['backward/walltime']     = ctx.self._stat.get('backward/walltime',0)     + (stop-start)
			ctx.self._stat['backward/lin_residual'] = ctx.self._stat.get('backward/lin_residual',0) + ctx.residual if isinstance(ctx.residual,int) else ctx.residual.detach() #(dx-A_dot(dx))
			ctx.self._stat['backward/lin_iters']    = ctx.self._stat.get('backward/lin_iters',0)    + ctx.lin_iters

		ctx.residual  = 0
		ctx.lin_iters = 0
		return None, None, dx, None



class ode_solver(Module, metaclass=ABCMeta):
	def __init__(self, rhs, T, num_steps):
		super().__init__()

		# ODE count in a network as a class property
		self.__class__.ode_count = getattr(self.__class__,'ode_count',-1) + 1
		self.name = str(self.__class__.ode_count)+".ode"

		self.rhs = rhs # this is registered as a submodule
		self.register_buffer('_T', torch.tensor(T))
		self.register_buffer('_num_steps', torch.tensor(num_steps))
		self.register_buffer('_h', torch.tensor(T/num_steps))

		self._stat = {}

	########################################

	@property
	def num_steps(self):
		return self._num_steps.item()
	@num_steps.setter
	def num_steps(self, num_steps):
		self._num_steps.fill_(num_steps)
		self._h = self.T / num_steps

	@property
	def T(self):
		return self._T.item()
	@T.setter
	def T(self, T):
		self._T.fill_(T)
		self._h = T / self.num_steps

	@property
	def h(self):
		return self._h.item()

	########################################

	@property
	def statistics(self, reset=True):
		with torch.no_grad():
			stat = {}

			# get number of training/validation propagations and remove it from dict
			train_steps = self._stat.pop('forward_train/steps',0)
			valid_steps = self._stat.pop('forward_valid/steps',0)
			backw_steps = self._stat.pop('backward/steps',0)

			for key, value in self._stat.items():
				mode, stat_name = key.split('/')
				new_key = 'stat_'+mode+'/'+self.name+'_'+stat_name
				if mode=='forward_train':
					stat[new_key] = value / train_steps
				elif mode=='forward_valid':
					stat[new_key] = value / valid_steps
				elif mode=='backward':
					stat[new_key] = value / backw_steps
				# stat[new_key] = value / (valid_steps if mode=='forward_valid' else train_steps)

			if reset: self._stat = {}
		return stat

	########################################

	@abstractmethod
	def residual(self, xy, fval=[None,None], param=None):
		pass

	@abstractmethod
	def ode_step(self, t0, x, param=None):
		pass

	def nsolve(self, fun, y, tol):
		iters = [0]
		resid = [0]

		# if _forward_scipy: # use SciPy solver

		# 	def functional(z):
		# 		z0   = torch.from_numpy(z).view_as(x).to(device=y.device,dtype=y.dtype).requires_grad_(True)
		# 		fun  = residual_fun(z0)
		# 		print(fun)
		# 		dfun = torch.autograd.grad(fun, z0)[0] # no retain_graph because derivative through nonlinearity is different at each iteration
		# 		return fun.cpu().detach().numpy().astype(np.double), dfun.cpu().detach().numpy().ravel().astype(np.double)
		# 	# opt_res = opt.minimize(functional, x.cpu().detach().numpy(), method='CG', jac=True, tol=_TOL, options={'maxiter': _max_iters,'disp': False})
		# 	opt_res = opt.minimize(functional, y.cpu().detach().numpy(), method='L-BFGS-B', jac=True, tol=_TOL, options={'maxiter': _max_iters,'disp': False})

		# 	y = torch.from_numpy(opt_res.x).view_as(x).to(device=y.device,dtype=y.dtype)
		# 	residual[0] = torch.tensor(opt_res.fun).to(device=y.device,dtype=y.dtype)
		# 	cg_iters[0], success = opt_res.nit, opt_res.success

		# 	# assert success, "CG solver hasn't converged with residual "+str(r.detach().numpy())+" at iteration "+str(cg_iters)+" out of "+str(_max_iters)+" max iterations"
		# 	# assert r<=1.e-3, "CG solver hasn't converged with residual "+str(r.detach().numpy())+" after "+str(cg_iters)+" iterations"

		# 	if self.training:
		# 		y_fp = fixed_point_map(y.requires_grad_(True)) # required_grad_(True) to compute Jacobian in linsolve_backprop
		# 		y    = linsolve_backprop.apply(self, y, y_fp)

		# 	# print(residual[0])
		# 	# print('-------------')
		# 	# exit()


		# assert False, "torch.enable_grad is required here"


		# check initial residual and, as a side effect, perform spectral normalization
		init_resid = fun(y)
		if init_resid<tol:
			return y.detach(), {'iters': 0, 'residual': init_resid.detach()}

		# self.rhs.eval()					# freeze spectral normalization
		self.rhs.requires_grad_(False)	# freeze parameters

		# NOTE: in torch.optim.LBFGS, all norms are max norms hence no need to account for batch size in tolerance_grad & tolerance_change
		nsolver = torch.optim.LBFGS([y], lr=1, max_iter=_max_iters, max_eval=None, tolerance_grad=1.e-9, tolerance_change=1.e-9, history_size=10, line_search_fn='strong_wolfe')
		def closure():
			iters[0] += 1
			nsolver.zero_grad()
			resid[0] = fun(y)
			if resid[0]>tol: resid[0].backward()
			# if resid[0]>tol**2: resid[0].backward()
			# resid[0] = torch.sqrt(resid[0])
			return resid[0]
		nsolver.step(closure)

		# TODO: what if some parameters need to have requires_grad=False?
		self.rhs.requires_grad_(self.training)		# unfreeze parameters
		# self.rhs.requires_grad_(True)		# unfreeze parameters
		# self.rhs.train(mode=self.training)	# unfreeze spectral normalization

		return y.detach().requires_grad_(True), {'iters': iters[0], 'residual': resid[0].detach()}

	########################################

	# TODO: replace with generator
	def forward(self, y0, t0=0, param=None):
		y = [y0]
		for step in range(self.num_steps):
			t = t0 + step * self.h
			y.append(self.ode_step(t, y[-1], param))
		return torch.stack(y)


###############################################################################################


class theta_solver(ode_solver):
	def __init__(self, rhs, T, num_steps, theta=0.0, tol=_TOL):
		super().__init__(rhs, T, num_steps)

		self.register_buffer('_theta', torch.tensor(theta))
		self.tol = tol

	########################################

	@property
	def theta(self):
		return self._theta.item()
	@theta.setter
	def theta(self, theta):
		self._theta.fill_(theta)

	########################################

	@property
	def statistics(self, reset=True):
		stat = super().statistics
		stat['hparams/theta'] = self.theta
		return stat

	########################################


	def step_fun(self, t, x, y):
		# if theta==1, use left endpoint
		t = (t + self.theta * self.h) if self.theta<1 else t
		z = ((1-self.theta)*x if self.theta<1 else 0) + (self.theta*y if self.theta>0 else 0)
		return self.h * self.rhs(t, z)


	def residual(self, t, xval, fval=None, param=None):
		x, y = xval
		batch_dim = x.size(0)
		return ( y - x - self.step_fun(t,x,y) ).pow(2).sum() / batch_dim


	def ode_step(self, t, x, param=None):
		res = {'iters': 0, 'residual': 0}
		batch_dim = x.size(0)

		residual_fn = lambda z: self.residual(t, [x.detach(),z])

		start = time.time()
		if self.theta>0:
			# initial condition: make new (that's why clone) leaf (that's why detach) node which requires gradient
			y = x.clone().detach().requires_grad_(True)

			# residual of initial condition
			init_resid = residual_fn(y).detach()

			# spectral normalization has to be performed only once per forward pass
			# It has been performed in nsolve, so freeze here
			# self.rhs.eval()
			if init_resid<self.tol:
				res = {'iters': 0, 'residual': init_resid}
			else:
				y, res = self.nsolve( residual_fn, y, self.tol )

			# assert residual[0]<1.e-2, 'Error: divergence at t=%d, residual is %.2E'%(self.t, residual[0].detach().numpy())
			if res['residual']>self.tol:
				print('Warning: convergence not achieved for %s at t=%d, residual is %.2E'%(self.name, t, res['residual'].cpu().detach().numpy()))

			# NOTE: two evaluations of step_fun are requried anyway!
			y = linsolve_backprop.apply(self, y, x + self.step_fun(t, x, y.detach()), self.step_fun(t, x.detach(), y))

			# unfreeze spectral normalization
			# self.rhs.train(mode=self.training)
		else:
			y = x + self.step_fun(t, x, x)
		stop = time.time()

		if _collect_stat:
			mode = 'forward_train/' if self.training else 'forward_valid/'
			#######################
			self._stat[mode+'steps']    = self._stat.get(mode+'steps',0)    + 1
			self._stat[mode+'walltime'] = self._stat.get(mode+'walltime',0) + (stop-start)
			self._stat[mode+'iters']    = self._stat.get(mode+'iters',0)    + res['iters']
			self._stat[mode+'residual'] = self._stat.get(mode+'residual',0) + res['residual']
		return y



###############################################################################################
###############################################################################################


def choose_activation(activation):
	if isinstance(activation,str):
		if activation=='relu':
			return torch.nn.ReLU()
		elif activation=='gelu':
			return torch.nn.GELU()
		elif activation=='celu':
			return torch.nn.CELU()
		elif activation=='tanh':
			return torch.nn.Tanh()
		elif activation=='tanhshrink':
			return torch.nn.Tanhshrink()
		elif activation=='softsign':
			return torch.nn.Softsign()
	elif isinstance(activation,Module):
		return activation


###############################################################################################


class MLP(Module):
	def __init__(self, in_dim, out_dim, width, depth, activation='relu', final_activation=None, power_iters=0):
		super().__init__()

		# activation function of hidden layers and last layer
		sigma1 = choose_activation(activation)
		sigma2 = choose_activation(final_activation) if final_activation is not None else sigma1

		# linear layers
		linear_inp =   torch.nn.Linear(in_dim, width,   bias=True)
		linear_hid = [ torch.nn.Linear(width,  width,   bias=True) for _ in range(depth) ]
		linear_out =   torch.nn.Linear(width,  out_dim, bias=False)

		# spectral normalization
		if power_iters>0:
			linear_inp =   torch.nn.utils.spectral_norm( linear_inp, name='weight', n_power_iterations=power_iters, eps=1e-12, dim=None)
			linear_hid = [ torch.nn.utils.spectral_norm( linear,     name='weight', n_power_iterations=power_iters, eps=1e-12, dim=None) for linear in linear_hid ]
			linear_out =   torch.nn.utils.spectral_norm( linear_out, name='weight', n_power_iterations=power_iters, eps=1e-12, dim=None)

		# Multilayer perceptron
		net = [linear_inp] + [val for pair in zip([sigma1]*depth,linear_hid) for val in pair] + [sigma2,linear_out]
		self.net = torch.nn.Sequential(*net)

	def forward(self, x):
		return self.net(x)


class HollowMLP(Module):
	def __init__(self, dim, width, depth, activation='relu', final_activation=None, power_iters=0):
		super().__init__()

		self.conditioner = MLP(dim, width, width, depth, activation, activation, power_iters)
		self.transformer = MLP(width+1, 1, width+1, 1, activation, final_activation, power_iters)
		self.mask = torch.ones(1,dim,dim, dtype=torch.uint8)
		for d in range(dim):
			self.mask[:,d,d] = 0
		# self.mask = []
		# for d in range(dim):
		# 	self.mask.append( torch.ones((1,dim), dtype=torch.uint8) )
		# 	self.mask[d][:,d] = 0


	def jacdiag(self, x):
		x.requires_grad_(True)

		xx = torch.unsqueeze(x,1).expand(-1,x.shape[-1],-1)
		h  = self.conditioner(self.mask*xx.detach())
		f  = self.transformer(torch.cat((torch.unsqueeze(x,2),h),2)).squeeze()

		jac_diag, = torch.autograd.grad(
			outputs=f,
			inputs=x,
			grad_outputs=torch.ones_like(f),
			create_graph=True,  # need create_graph to find it's derivative
			only_inputs=True)
		return jac_diag
	# def jacdiag(self,x):
	# 	out = []
	# 	for d in range(x.shape[1]):
	# 		h = self.conditioner(self.mask[d]*x.detach())
	# 		f = self.transformer(torch.cat((torch.unsqueeze(x[:,d],1),h),1))
	# 		out.append(f)

	# 	v = torch.ones_like(x)
	# 	jac_diag, = torch.autograd.grad(
	# 		outputs=torch.cat(out,1),
	# 		inputs=x.requires_grad_(True),
	# 		grad_outputs=v,
	# 		create_graph=True,  # need create_graph to find it's derivative
	# 		only_inputs=True)
	# 	return jac_diag

	def trace(self, x):
		assert False, 'hollow trace not implemented'


	# def forward_hollow(self,x):
	# 	xx = torch.unsqueeze(x,1).expand(-1,x.shape[-1],-1)
	# 	h  = self.conditioner(self.mask*xx)
	# 	f  = self.transformer(torch.cat((torch.unsqueeze(x.detach(),2),h),2)).squeeze()
	# 	return f
	# def eval_jacnodiag(self,x):
	# 	out = []
	# 	xx = x.detach()
	# 	for d in range(x.shape[1]):
	# 		h = self.conditioner(self.mask[d]*x)
	# 		f = self.transformer(torch.cat((torch.unsqueeze(xx[:,d],1),h),1))
	# 		out.append(f)
	# 	return torch.cat(out,1)


	def forward(self, x, hollow=False):
		xx = torch.unsqueeze(x,1).expand(-1,x.shape[-1],-1)
		h  = self.conditioner(self.mask*xx)
		# h  = self.conditioner(self.mask*xx.repeat(1,x.shape[1],1))
		if hollow:
			f = self.transformer(torch.cat((torch.unsqueeze(x.detach(),2),h),2)).squeeze()
		else:
			f = self.transformer(torch.cat((torch.unsqueeze(x,2),h),2)).squeeze()
		return f
	# def forward(self, x):
	# 	out = []
	# 	for d in range(x.shape[1]):
	# 		h = self.conditioner(self.mask[d]*x)
	# 		f = self.transformer(torch.cat((torch.unsqueeze(x[:,d],1),h),1))
	# 		out.append(f)
	# 	return torch.cat(out,1)



class ParabolicPerceptron(Module):
	def __init__(self, dim, width, activation='relu', power_iters=0):
		super().__init__()

		# activation function
		self.sigma = choose_activation(activation)

		# parameters
		self.weight = torch.nn.Parameter(torch.Tensor(width, dim))
		self.bias   = torch.nn.Parameter(torch.Tensor(width))

		# intialize weights
		torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
		fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
		bound = 1 / math.sqrt(fan_in)
		torch.nn.init.uniform_(self.bias, -bound, bound)

		# spectral normalization
		if power_iters>0:
			torch.nn.utils.spectral_norm( self, name='weight', n_power_iterations=power_iters, eps=1e-12, dim=None)

	def forward(self, x):
		return -torch.nn.functional.linear( self.sigma(torch.nn.functional.linear(x, self.weight, self.bias)), self.weight.t(), None )



class HamiltonianPerceptron(Module):
	def __init__(self, dim, width, activation='relu', power_iters=0):
		super().__init__()
		assert dim%2==0,   'dim must be power of 2 for HamiltonianPerceptron'
		assert width%2==0, 'width must be power of 2 for HamiltonianPerceptron'

		# activation function
		self.sigma = choose_activation(activation)

		# parameters
		self.weight1 = torch.nn.Parameter(torch.Tensor(width//2, dim//2))
		self.weight2 = torch.nn.Parameter(torch.Tensor(width//2, dim//2))
		self.bias1   = torch.nn.Parameter(torch.Tensor(width//2))
		self.bias2   = torch.nn.Parameter(torch.Tensor(width//2))

		# intialize weights
		torch.nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
		torch.nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
		fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight1)
		bound = 1 / math.sqrt(fan_in)
		torch.nn.init.uniform_(self.bias1, -bound, bound)
		torch.nn.init.uniform_(self.bias2, -bound, bound)

		# spectral normalization
		if power_iters>0:
			torch.nn.utils.spectral_norm( self, name='weight1', n_power_iterations=power_iters, eps=1e-12, dim=None)
			torch.nn.utils.spectral_norm( self, name='weight2', n_power_iterations=power_iters, eps=1e-12, dim=None)

	def forward(self, x):
		batch_dim, x_dim = x.shape
		x1, x2 = x[:,:x_dim//2], x[:,x_dim//2:]
		y1 =  torch.nn.functional.linear( self.sigma(torch.nn.functional.linear(x2, self.weight1, self.bias1)), self.weight1.t(), None )
		y2 = -torch.nn.functional.linear( self.sigma(torch.nn.functional.linear(x1, self.weight2, self.bias2)), self.weight2.t(), None )
		return torch.cat( (y1, y2), 1 )



class LinearParabolic(torch.nn.Linear):
	def __init__(self, dim, bias=True):
		super().__init__(dim,dim,bias)

	def forward(self, x):
		return torch.nn.functional.linear(x, -self.weight.t()@self.weight, self.bias)



class LinearHyperbolic(torch.nn.Linear):
	def __init__(self, dim, bias=True):
		super().__init__(dim,dim,bias)

	def forward(self, x):
		return torch.nn.functional.linear(x, self.weight.t()-self.weight, self.bias)



###############################################################################################

class PreActConv2d(Module):
	def __init__(self, channels, depth, kernel_size, activation='relu', power_iters=0):
		super().__init__()

		# activation function of hidden layers
		sigma = choose_activation(activation)

		# conv layers
		conv_hid = [ torch.nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2, bias=True) for _ in range(depth) ]

		# spectral normalization
		if power_iters>0:
			conv_hid = [ torch.nn.utils.spectral_norm(conv, name='weight', n_power_iterations=power_iters, eps=1e-12, dim=None) for conv in conv_hid ]

		# normalization layers
		# norm_hid = [ torch.nn.BatchNorm2d(channels) for _ in range(depth) ]

		# network
		# self.net = torch.nn.Sequential(*[val for triple in zip(norm_hid,[sigma]*depth,conv_hid) for val in triple])
		self.net = torch.nn.Sequential(*[val for pair in zip([sigma]*depth,conv_hid) for val in pair])

	def forward(self, x):
		return self.net(x)





###############################################################################################
###############################################################################################



class diff_clamp(Function):
	@staticmethod
	def forward(ctx, x, min, max):
		return torch.clamp(x, min, max)

	@staticmethod
	def backward(ctx, dy):
		return dy.clone(), None, None




