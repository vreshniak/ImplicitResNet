import time
import math
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla
import scipy.optimize as opt
import matplotlib.pyplot as plt
from pathlib import Path

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
_linTOL = 1.e-8
_max_iters = 100

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


class neumann_backprop(Function):
	@staticmethod
	def forward(ctx, y, y_fp):
		# ctx.obj = obj
		ctx.save_for_backward(y, y_fp)
		return y

	@staticmethod
	def backward(ctx, dy):
		y, y_fp, = ctx.saved_tensors

		# residual = lambda dx: (dx-A_dot(dx)-dy).flatten().norm() # \| (I-A) * dx - dy \|
		A_dot    = lambda x: torch.autograd.grad(y_fp, y, grad_outputs=x, retain_graph=True, only_inputs=True)[0]
		residual = lambda Adx: (Adx-dy).reshape((dy.size()[0],-1)).norm(dim=1).max() #.flatten().norm() # \| (I-A) * dx - dy \|

		tol = atol = torch.tensor(_TOL)
		TOL = torch.max(tol*dy.norm(), atol)

		#######################################################################
		# Neumann series

		dx  = dy
		Ady = A_dot(dy)
		Adx = Ady
		r1  = residual(dx-Adx)
		neu_iters = 1
		while r1>=TOL and neu_iters<_max_iters:
			r0  = r1
			dx  = dx + Ady
			Ady = A_dot(Ady)
			Adx = Adx + Ady
			r1  = residual(dx-Adx)
			neu_iters += 1
			assert r1<r0, "Neumann series hasn't converged at iteration "+str(neu_iters)+" out of "+str(_max_iters)+" max iterations"

		if _collect_stat:
			global _backward_stat
			_backward_stat['steps']        = _backward_stat.get('steps',0) + 1
			_backward_stat['neu_residual'] = _backward_stat.get('neu_residual',0) + r1
			_backward_stat['neu_iters']    = _backward_stat.get('neu_iters',0) + neu_iters
		return None, dx


class linsolve_backprop(Function):
	@staticmethod
	def forward(ctx, self, y, y_fp):
		ctx.save_for_backward(y, y_fp)
		ctx.self = self
		return y_fp

	@staticmethod
	def backward(ctx, dy):
		y, y_fp, = ctx.saved_tensors
		ndof  = y.nelement()
		batch_dim = dy.size()[0]

		ctx.residual  = 0
		ctx.lin_iters = 0

		tol = atol = torch.tensor(_linTOL)
		TOL = torch.max(tol*dy.norm(), atol)

		if _backward_scipy:
			A_dot    = lambda x: torch.autograd.grad(y_fp, y, grad_outputs=x, retain_graph=True, only_inputs=True)[0]
			residual_fn = lambda Adx: (Adx-dy).reshape((dy.size()[0],-1)).norm(dim=1).max() # \| (I-A) * dx - dy \|

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

			ctx.residual = residual_fn(dx-A_dot(dx))

		else:
			A_dot = lambda x: torch.autograd.grad(y_fp, y, grad_outputs=x, create_graph=True, only_inputs=True)[0] # need create_graph to find it's derivative

			# initial condition
			dx = dy.clone().detach().requires_grad_(True)
			nsolver = torch.optim.LBFGS([dx], lr=1, max_iter=_max_iters, max_eval=None, tolerance_grad=1.e-9, tolerance_change=1.e-9, history_size=5, line_search_fn='strong_wolfe')
			def closure():
				nsolver.zero_grad()
				ctx.residual = ( dx - A_dot(dx) - dy ).pow(2).sum() / batch_dim
				if ctx.residual>_linTOL:
					ctx.residual.backward(retain_graph=True)
				ctx.lin_iters += 1
				return ctx.residual
			nsolver.step(closure)
			# dx = dx.detach()

		if _collect_stat:
			# global _backward_stat
			ctx.self.backward_stat['steps']        = ctx.self.backward_stat.get('steps',0) + 1
			ctx.self.backward_stat['lin_residual'] = ctx.self.backward_stat.get('lin_residual',0) + ctx.residual if isinstance(ctx.residual,int) else ctx.residual.detach() #(dx-A_dot(dx))
			ctx.self.backward_stat['lin_iters']    = ctx.self.backward_stat.get('lin_iters',0)    + ctx.lin_iters

		ctx.residual  = 0
		ctx.lin_iters = 0
		return None, None, dx



class fp_step(Module):
	def __init__(self, rhs, t, h=1, theta=0.0):
		super(fp_step,self).__init__()

		########################################
		self.rhs       = rhs
		self.t         = t
		self.h         = h
		self.theta     = theta
		self.theta_h   = theta * h
		########################################

		# self.forward_stat = { 'residual_in': 0, 'residual_out': 0, 'fp_iters': 0}
		self.regularizers = { 'Lipschitz': 0 }
		# self.fstep     = 0
		# self.converged = 0

		# self.device = self.parameters().__next__().device
		# self.dtype  = self.parameters().__next__().dtype

	@property
	def h(self):
		return self._h
	@h.setter
	def h(self, h):
		self._h = h

	@property
	def theta(self):
		return self._theta
	@theta.setter
	def theta(self, theta):
		self._theta = theta

	def forward(self, y, param):
		fp_iters = 0
		success  = True
		y_in = y.clone()

		# explicit part
		explicit = y_in if self.theta==1 else y_in + (1-self.theta) * self.h * self.rhs(self.t, y_in, param)

		# fixed point map and residual
		def fixed_point_map(x):
			y = explicit + self.theta_h * self.rhs(self.t, x, param)
			return y, (y-x).reshape((y.size()[0],-1)).norm(dim=1).mean()
		def residual_fun(x0):
			x1, r1 = fixed_point_map(x0)
			return r1

		TOL = torch.max(_TOL*self.theta_h*self.rhs(self.t, y_in, param).norm(), torch.tensor(_TOL))

		# self.regularizers['Lipschitz'] = 0.0
		if self.theta>0:
			with torch.no_grad():
				y1, r1 = fixed_point_map(y_in); fp_iters+=1
				while fp_iters<_max_iters and r1>TOL:
					y0, r0 = y1, r1
					y1, r1 = fixed_point_map(y0)
					# print((y0-y1))
					# print(fp_iters, r1/r0, r1, self.theta_h * self.rhs(self.t, y0, param).norm())
					# self.regularizers['Lipschitz'] = self.regularizers['Lipschitz'] + r1 / r0.detach()
					if r1>r0:
						y1, r1  = y0, r0
						success = False
						if self.training:
							# print(y0-y1)
							assert success, "Fixed-point iteration hasn't converged at iteration "+str(fp_iters)+" out of "+str(_max_iters)+" max iterations"
						break
					fp_iters+=1
				# self.regularizers['Lipschitz'] = self.regularizers['Lipschitz'] / (fp_iters-1) if fp_iters>1 else 0.0
				# exit()
				if self.training:
					assert r1<=TOL, "Fixed-point iteration hasn't converged with residual "+str(r1)+"after "+str(fp_iters)+" iterations"

			y = y1
			if self.training:
				y = y.detach().requires_grad_(True)
				y_fp, _ = fixed_point_map(y)
				y = neumann_backprop.apply(y, y_fp)
		else:
			y = explicit

		if _collect_stat:
			global _forward_stat
			mode = 'train/' if self.training else 'val/'
			_forward_stat[mode+'steps']        = _forward_stat.get(mode+'steps',0)        + 1
			_forward_stat[mode+'residual_in']  = _forward_stat.get(mode+'residual_in',0)  + residual_fun(y_in).detach().numpy()
			_forward_stat[mode+'residual_out'] = _forward_stat.get(mode+'residual_out',0) + residual_fun(y).detach().numpy()
			_forward_stat[mode+'fp_iters']     = _forward_stat.get(mode+'fp_iters',0)     + fp_iters
		return y


class ode_step(Module):
	def __init__(self, rhs, t, h=1, theta=0.0, method='outer', tol=_TOL):
		super(ode_step,self).__init__()

		self.tol = tol
		self.rhs = rhs
		self.t   = t
		self.h   = h
		self.method = method
		self.register_buffer('_theta', torch.tensor(theta))

		self.forward_stat  = {}
		self.backward_stat = {}


	@property
	def h(self):
		return self._h
	@h.setter
	def h(self, h):
		self._h = h

	@property
	def theta(self):
		return self._theta
	@theta.setter
	def theta(self, theta):
		self._theta.fill_(theta)


	def residual(self, xy, fval=[None, None], param=None):
		x, y = xy
		batch_dim = x.size(0)
		if self.method=='inner':
			t = self.t + self.theta * self.h
			z = ( 0 if self.theta==1 else (1-self.theta)*x ) + self.theta*y
			f = self.rhs(t, z, param)
		elif self.method=='outer':
			fx = fval[0] if fval[0] is not None else ( 0 if self.theta==1 else self.rhs(self.t, x, param) )
			fy = fval[1] if fval[1] is not None else self.rhs(self.t+self.h, y, param)
			f  = (1-self.theta) * fx + self.theta * fy
		return ( y - x - self.h * f  ).pow(2).sum() / batch_dim


	def forward(self, x, param=None):
		cg_iters  = [0] # need list to access cg_iters by reference from def closure()
		residual  = [0] # same for residual
		batch_dim = x.size(0)

		if self.theta>0:
			if self.method=='outer':
				theta_h = self.theta * self.h

				# explicit part
				explicit = x if self.theta==1 else x + (1-self.theta) * self.h * self.rhs(self.t, x, param)

				# fixed point map and residual
				fp_map       = lambda z: explicit + theta_h * self.rhs(self.t+1, z, param)
				# residual_fun = lambda z: ( z - explicit.detach() - theta_h * self.rhs(self.t+1, z, param) ).pow(2).sum() / batch_dim
				# residual_fun = lambda z: ( z - x.detach() - (1-self.theta) * self.h * self.rhs(self.t, x, param).detach() - theta_h * self.rhs(self.t+1, z, param) ).pow(2).sum() / batch_dim
			elif self.method=='inner':
				# fixed point map and residual
				fp_map       = lambda z: x + self.h * self.rhs(self.t+self.theta, ((1-self.theta)*x if self.theta<1 else 0) + self.theta*z, param)
				# residual_fun = lambda z: ( z - x.detach() - self.h * self.rhs(self.t+self.theta, ((1-self.theta)*x.detach() if self.theta<1 else 0) + self.theta*z, param) ).pow(2).sum() / batch_dim

			fx = None if self.method=='inner' else self.rhs(self.t, x, param).detach()

			# initial condition (make new leaf node which requires gradient)
			# y = explicit.detach().requires_grad_(True)
			# y = y_in.detach().requires_grad_(True)
			y = x.clone().detach().requires_grad_(True)

			if _forward_scipy: # use SciPy solver

				def functional(z):
					z0   = torch.from_numpy(z).view_as(x).to(device=y.device,dtype=y.dtype).requires_grad_(True)
					fun  = residual_fun(z0)
					print(fun)
					dfun = torch.autograd.grad(fun, z0)[0] # no retain_graph because derivative through nonlinearity is different at each iteration
					return fun.cpu().detach().numpy().astype(np.double), dfun.cpu().detach().numpy().ravel().astype(np.double)
				# opt_res = opt.minimize(functional, x.cpu().detach().numpy(), method='CG', jac=True, tol=_TOL, options={'maxiter': _max_iters,'disp': False})
				opt_res = opt.minimize(functional, y.cpu().detach().numpy(), method='L-BFGS-B', jac=True, tol=_TOL, options={'maxiter': _max_iters,'disp': False})

				y = torch.from_numpy(opt_res.x).view_as(x).to(device=y.device,dtype=y.dtype)
				residual[0] = torch.tensor(opt_res.fun).to(device=y.device,dtype=y.dtype)
				cg_iters[0], success = opt_res.nit, opt_res.success

				# assert success, "CG solver hasn't converged with residual "+str(r.detach().numpy())+" at iteration "+str(cg_iters)+" out of "+str(_max_iters)+" max iterations"
				# assert r<=1.e-3, "CG solver hasn't converged with residual "+str(r.detach().numpy())+" after "+str(cg_iters)+" iterations"

				if self.training:
					y_fp = fixed_point_map(y.requires_grad_(True)) # required_grad_(True) to compute Jacobian in linsolve_backprop
					y    = linsolve_backprop.apply(self, y, y_fp)

				# print(residual[0])
				# print('-------------')
				# exit()

			else: # use PyTorch solver
				self.rhs.eval()					# freeze spectral normalization
				self.rhs.requires_grad_(False)	# freeze parameters

				# NOTE: in torch.optim.LBFGS, all norms are max norms hence no need to account for batch size in tolerance_grad & tolerance_change
				nsolver = torch.optim.LBFGS([y], lr=1, max_iter=_max_iters, max_eval=None, tolerance_grad=1.e-9, tolerance_change=1.e-9, history_size=10, line_search_fn='strong_wolfe')
				def closure():
					cg_iters[0] += 1
					nsolver.zero_grad()
					residual[0] = self.residual([x.detach(),y], [fx,None], param)
					# residual[0] = residual_fun(y)
					# residual[0] = ( y - explicit.detach() - theta_h * self.rhs(self.t+1, y, param) ).pow(2).sum() / batch_dim
					# residual[0] = ( y - explicit.detach() - self.theta_h * self.rhs(self.t, y, param) ).reshape((batch_dim,-1)).pow(2).sum(dim=1).mean()
					if residual[0]>self.tol:
						residual[0].backward()
					return residual[0]
				nsolver.step(closure)

				# assert residual[0]<1.e-2, 'Error: divergence at t=%d, residual is %.2E'%(self.t, residual[0].detach().numpy())
				if residual[0]>self.tol:
					print('Warning: convergence not achieved at t=%d, residual is %.2E'%(self.t, residual[0].detach().numpy()))

				# TODO: what if some parameters need to have requires_grad=False?
				self.rhs.requires_grad_(True)		# unfreeze parameters
				self.rhs.train(mode=self.training)	# unfreeze spectral normalization

			y = linsolve_backprop.apply(self, y, fp_map(y))
		else:
			y = x + self.h * self.rhs(self.t, x, param)


		if _collect_stat:
			mode = 'train/' if self.training else 'val/'
			#######################
			self.forward_stat[mode+'steps']    = self.forward_stat.get(mode+'steps',0)    + 1
			self.forward_stat[mode+'iters']    = self.forward_stat.get(mode+'iters',0)    + cg_iters[0]
			#######################
			# self.forward_stat[mode+'residual_in']  = self.forward_stat.get(mode+'residual_in',0)  + residual_fun(x).detach()
			self.forward_stat[mode+'residual'] = self.forward_stat.get(mode+'residual',0) + residual[0] if isinstance(residual[0],int) else residual[0].detach()
			#######################

		# cg_iters  = [0]
		# residual  = [0]
		return y

	# def forward(self, x, param=None):
	# 	cg_iters  = [0] # need this to access cg_iters by reference from def closure()
	# 	residual  = [0]
	# 	batch_dim = x.size()[0]

	# 	theta_h = self.theta * self.h

	# 	# RHS at the left endpoint
	# 	rhs_l = self.h * self.rhs(self.t, x, param)

	# 	# explicit part
	# 	explicit = x if self.theta==1 else x + (1-self.theta) * rhs_l

	# 	# fixed point map and residual
	# 	fixed_point_map = lambda z: explicit + theta_h * self.rhs(self.t, z, param)
	# 	residual_fun    = lambda z: (z-fixed_point_map(z)).pow(2).sum() / batch_dim
	# 	# residual_fun    = lambda z: (fixed_point_map(z)-z).reshape((batch_dim,-1)).norm(dim=1).mean()
	# 	# residual_fun    = lambda z: (fixed_point_map(z)-z).reshape((batch_dim,-1)).pow(2).sum(dim=1).mean()

	# 	if self.theta>0:
	# 		# initial condition (make new leaf node which requires gradient)
	# 		# y = explicit.detach().requires_grad_(True)
	# 		# y = y_in.detach().requires_grad_(True)
	# 		y = x.clone().detach().requires_grad_(True)

	# 		if _forward_scipy: # use SciPy solver

	# 			def functional(z):
	# 				z0   = torch.from_numpy(z).view_as(x).to(device=y.device,dtype=y.dtype).requires_grad_(True)
	# 				fun  = residual_fun(z0)
	# 				print(fun)
	# 				dfun = torch.autograd.grad(fun, z0)[0] # no retain_graph because derivative through nonlinearity is different at each iteration
	# 				return fun.cpu().detach().numpy().astype(np.double), dfun.cpu().detach().numpy().ravel().astype(np.double)
	# 			# opt_res = opt.minimize(functional, x.cpu().detach().numpy(), method='CG', jac=True, tol=_TOL, options={'maxiter': _max_iters,'disp': False})
	# 			opt_res = opt.minimize(functional, y.cpu().detach().numpy(), method='L-BFGS-B', jac=True, tol=_TOL, options={'maxiter': _max_iters,'disp': False})

	# 			y = torch.from_numpy(opt_res.x).view_as(x).to(device=y.device,dtype=y.dtype)
	# 			residual[0] = torch.tensor(opt_res.fun).to(device=y.device,dtype=y.dtype)
	# 			cg_iters[0], success = opt_res.nit, opt_res.success

	# 			# assert success, "CG solver hasn't converged with residual "+str(r.detach().numpy())+" at iteration "+str(cg_iters)+" out of "+str(_max_iters)+" max iterations"
	# 			# assert r<=1.e-3, "CG solver hasn't converged with residual "+str(r.detach().numpy())+" after "+str(cg_iters)+" iterations"

	# 			if self.training:
	# 				y_fp = fixed_point_map(y.requires_grad_(True)) # required_grad_(True) to compute Jacobian in linsolve_backprop
	# 				y    = linsolve_backprop.apply(self, y, y_fp)

	# 			# print(residual[0])
	# 			# print('-------------')
	# 			# exit()

	# 		else: # use PyTorch solver
	# 			self.rhs.eval()					# freeze spectral normalization
	# 			self.rhs.requires_grad_(False)	# freeze parameters

	# 			# NOTE: in torch.optim.LBFGS, all norms are max norms hence no need to account for batch size in tolerance_grad & tolerance_change
	# 			nsolver = torch.optim.LBFGS([y], lr=1, max_iter=_max_iters, max_eval=None, tolerance_grad=1.e-9, tolerance_change=1.e-9, history_size=10, line_search_fn='strong_wolfe')
	# 			def closure():
	# 				cg_iters[0] += 1
	# 				nsolver.zero_grad()
	# 				residual[0] = ( y - explicit.detach() - theta_h * self.rhs(self.t+1, y, param) ).pow(2).sum() / batch_dim
	# 				# residual[0] = ( y - x.detach() - self.h * self.rhs(self.t, (1-self.theta)*x.detach()+self.theta*y, param) ).pow(2).sum() / batch_dim
	# 				# residual[0] = ( y - explicit.detach() - self.theta_h * self.rhs(self.t, y, param) ).reshape((batch_dim,-1)).pow(2).sum(dim=1).mean()
	# 				if residual[0]>_TOL:
	# 					residual[0].backward()
	# 				return residual[0]
	# 			nsolver.step(closure)

	# 			# TODO: what if some parameters need to have requires_grad=False?
	# 			self.rhs.requires_grad_(True)		# unfreeze parameters
	# 			self.rhs.train(mode=self.training)	# unfreeze spectral normalization

	# 		# RHS at the right endpoint
	# 		rhs_r = self.h * self.rhs(self.t+1, y, param)
	# 		y = linsolve_backprop.apply(self, y, explicit + self.theta * rhs_r)
	# 	else:
	# 		y = explicit


	# 	# assert residual_fun(y).detach()<_TOL, "ode_step did not converge "+str(residual_fun(y).detach())
	# 	if _collect_stat:
	# 		mode = 'train/' if self.training else 'val/'
	# 		#######################
	# 		self.forward_stat[mode+'steps']        = self.forward_stat.get(mode+'steps',0)        + 1
	# 		self.forward_stat[mode+'cg_iters']     = self.forward_stat.get(mode+'cg_iters',0)     + cg_iters[0]
	# 		#######################
	# 		self.forward_stat[mode+'residual_in']  = self.forward_stat.get(mode+'residual_in',0)  + residual_fun(x).detach()
	# 		self.forward_stat[mode+'residual_out'] = self.forward_stat.get(mode+'residual_out',0) + residual[0] if isinstance(residual[0],int) else residual[0].detach() #residual_fun(y)
	# 		#######################

	# 	cg_iters  = [0]
	# 	residual  = [0]
	# 	return y



##############################################################################################################
##############################################################################################################


class ode_solver(Module):
	# def __init__(self, rhs, T, num_steps, theta=0.0, fp_iters=0, Lip=0.5, reg_iters=1, solver='cg', method='inner'):
	def __init__(self, rhs, T, num_steps, theta=0.0, solver='cg', method='inner', tol=_TOL):
		super().__init__()

		# ODE count in a network as a class property
		self.__class__.ode_count = getattr(self.__class__,'ode_count',-1) + 1
		self.name = str(self.__class__.ode_count)+".ode"

		self.rhs = rhs # need this to register parameters
		if solver=='fp':
			self.ode_step = torch.nn.ModuleList([fp_step(rhs, step, theta=theta, h=T/num_steps) for step in range(num_steps)])
		elif solver=='cg':
			self.ode_step = torch.nn.ModuleList([ode_step(rhs, step, theta=theta, h=T/num_steps, method=method, tol=tol) for step in range(num_steps)])

		self.register_buffer('_theta', torch.tensor(theta))
		self._num_steps = num_steps
		self._T         = T


	########################################

	@property
	def num_steps(self):
		return self._num_steps
	@num_steps.setter
	def num_steps(self, num_steps):
		self._num_steps = num_steps
		for step in self.ode_step:
			step.h = self.T / num_steps

	@property
	def T(self):
		return self._T
	@T.setter
	def T(self, T):
		self._T = T
		for step in self.ode_step:
			step.h = T / self.num_steps

	@property
	def theta(self):
		return self._theta
	@theta.setter
	def theta(self, theta):
		self._theta.fill_(theta)
		for step in self.ode_step:
			step.theta = theta

	########################################

	@property
	def statistics(self, reset=True):
		stat = {}
		for step in range(self.num_steps):
			train_steps = self.ode_step[step].forward_stat.pop('train/steps',0)
			val_steps   = self.ode_step[step].forward_stat.pop('val/steps',0)
			for key, value in self.ode_step[step].forward_stat.items():
				new_key = ('forward_stat_'+key).replace('/', '/'+self.name+'_')
				if 'train' in key:
					stat[new_key] = stat.get(new_key,0) + value / train_steps / self.num_steps # / (self.num_steps if 'residual' in key else 1)
				if 'val' in key:
					stat[new_key] = stat.get(new_key,0) + value / val_steps / self.num_steps

			back_steps = self.ode_step[step].backward_stat.pop('steps',0)
			for key, value in self.ode_step[step].backward_stat.items():
				new_key = 'backward_stat/'+self.name+'_'+key
				stat[new_key] = stat.get(new_key,0) + value / back_steps / self.num_steps

			if reset:
				self.ode_step[step].forward_stat  = {}
				self.ode_step[step].backward_stat = {}
		stat['hparams/theta'] = self.theta
		return stat

	########################################

	def residual(self, step, xy, fval=[None,None], param=None):
		return self.ode_step[step].residual(xy, fval, param)

	# TODO: replace with generator
	def forward(self, y0, param=None):
		y = [y0]
		for step in range(self.num_steps):
			y.append(self.ode_step[step](y[-1], param))
		return torch.stack(y)




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
		linear_inp =   torch.nn.Linear(in_dim, width,   bias=False)
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
		assert dim%2==0, 'dim must be power of 2 for HamiltonianPerceptron'

		# activation function
		self.sigma = choose_activation(activation)

		# parameters
		self.weight1 = torch.nn.Parameter(torch.Tensor(width, dim//2))
		self.weight2 = torch.nn.Parameter(torch.Tensor(width, dim//2))
		self.bias1   = torch.nn.Parameter(torch.Tensor(width))
		self.bias2   = torch.nn.Parameter(torch.Tensor(width))

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











