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

import custom_utils as utils
from jacobian import JacobianReg



###############################################################################
###############################################################################


# global parameters
_TOL = 1.e-6
_max_iters = 1000

_collect_stat  = True
_forward_stat  = {}
_backward_stat = {}

# _dtype  = torch.float
# _device = torch.device("cpu")

_forward_scipy  = False
_backward_scipy = False

# _to_numpy = lambda x, dtype: x.cpu().detach().numpy().astype(dtype)

_ode_count = 0


###############################################################################
###############################################################################

def jacobian_loss(output, input):
	# jacobian = torch.autograd.grad(
	# 	outputs=output,
	# 	inputs=input,
	# 	grad_outputs=torch.ones(*output.shape).to(input.device),
	# 	create_graph=True,  # need create_graph to find it's derivative
	# 	only_inputs=True)[0]
	# jacobian = torch.autograd.functional.jacobian(output, input, create_graph=True, strict=True)
	jacobian = torch.autograd.functional.jacobian(output, input.flatten(), create_graph=True, strict=False)
	jacobian_norm = jacobian.pow(2).sum() #norm(2)
	# jacobian_norm = torch.trace((jacobian+0.1).pow(2))
	return jacobian_norm

def divergence_loss(output, input):
	# jacobian = torch.autograd.grad(
	# 	outputs=torch.reshape(output,(1,output.numel())),
	# 	inputs=torch.reshape(input,(1,input.numel())),
	# 	grad_outputs=torch.eye(input.numel()).to(input.device),
	# 	create_graph=True,  # need create_graph to find it's derivative
	# 	only_inputs=True)[0]
	# jacobian = torch.autograd.grad(
	# 	outputs=output,
	# 	inputs=input,
	# 	grad_outputs=torch.ones(*output.shape).to(input.device),
	# 	create_graph=True,  # need create_graph to find it's derivative
	# 	only_inputs=True)[0]
	jacobian = torch.autograd.functional.jacobian(output, input.flatten(), create_graph=True, strict=False)
	jacobian_norm = torch.trace((jacobian+0.0).pow(2))
	return jacobian_norm


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

		tol = atol = torch.tensor(_TOL)
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
				if ctx.residual>_TOL:
					ctx.residual.backward(retain_graph=True)
				ctx.lin_iters += 1
				return ctx.residual
			nsolver.step(closure)
			# dx = dx.detach()

		if _collect_stat:
			# global _backward_stat
			ctx.self.backward_stat['steps']        = ctx.self.backward_stat.get('steps',0) + 1
			ctx.self.backward_stat['lin_residual'] = ctx.self.backward_stat.get('lin_residual',0) + ctx.residual.detach() #(dx-A_dot(dx))
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


class cg_step(Module):
	def __init__(self, rhs, t, h=1, theta=0.0, alpha_rhsjac=0, alpha_rhsdiv=0, alpha_fpdiv=0):
		super(cg_step,self).__init__()

		self.rhs = rhs
		self.t   = t
		self.h   = h
		self.register_buffer('_theta', torch.tensor(theta))

		self.forward_stat  = {}
		self.backward_stat = {}
		self.regularizers  = {}
		self.alpha_rhsjac  = alpha_rhsjac
		self.alpha_fpdiv   = alpha_fpdiv
		self.alpha_rhsdiv  = alpha_rhsdiv
		self.jacreg = JacobianReg(n=1)
		self.divreg = utils.divreg(n=1)


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


	def forward(self, x, param=None):
		cg_iters  = [0] # need this to access cg_iters by reference from def closure()
		residual  = [0]
		batch_dim = x.size()[0]
		rhs_jac   = 0
		fp_div    = 0
		rhsjac    = 0
		rhsdiv    = 0

		theta_h = self.theta * self.h

		# RHS at the left endpoint
		rhs_l = self.h * self.rhs(self.t, x.requires_grad_(True), param)
		if self.training:
			if self.alpha_rhsjac>0:
				rhsjac = self.jacreg(x, rhs_l)
				self.regularizers['rhs_jac'] = self.alpha_rhsjac * rhsjac
			if self.alpha_rhsdiv>0:
				rhsdiv = self.divreg(rhs_l, x, create_graph=True)
				self.regularizers['rhs_div'] = self.alpha_rhsdiv * rhsdiv


		# explicit part
		explicit = x if self.theta==1 else x + (1-self.theta) * rhs_l

		# fixed point map and residual
		fixed_point_map = lambda z: explicit + theta_h * self.rhs(self.t, z, param)
		residual_fun    = lambda z: (z-fixed_point_map(z)).pow(2).sum() / batch_dim
		# residual_fun    = lambda z: (fixed_point_map(z)-z).reshape((batch_dim,-1)).norm(dim=1).mean()
		# residual_fun    = lambda z: (fixed_point_map(z)-z).reshape((batch_dim,-1)).pow(2).sum(dim=1).mean()

		# TOL = torch.max(_TOL*self.theta_h*self.rhs(self.t, x, param).norm(), torch.tensor(_TOL))

		# fixed_point_map2 = lambda z: x.detach() + (1-self.theta) * self.h * self.rhs(self.t, x.detach(), param) + self.theta_h * self.rhs(self.t, z.detach(), param)
		# residual_fun2 = lambda z: (fixed_point_map2(z)-z.detach()).pow(2).sum() / batch_dim
		# self.Lip = residual_fun2(x.detach() + self.h * self.rhs(self.t, x.detach(), param)) / residual_fun2(x)

		if self.theta>0:
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
				nsolver = torch.optim.LBFGS([y], lr=1, max_iter=_max_iters, max_eval=None, tolerance_grad=1.e-9, tolerance_change=1.e-9, history_size=100, line_search_fn='strong_wolfe')
				def closure():
					cg_iters[0] += 1
					nsolver.zero_grad()
					residual[0] = ( y - explicit.detach() - theta_h * self.rhs(self.t+1, y, param) ).pow(2).sum() / batch_dim
					# residual[0] = ( y - explicit.detach() - self.theta_h * self.rhs(self.t, y, param) ).reshape((batch_dim,-1)).pow(2).sum(dim=1).mean()
					if residual[0]>_TOL:
						residual[0].backward()
					return residual[0]
				nsolver.step(closure)

				# TODO: what if some parameters need to have requires_grad=False?
				self.rhs.requires_grad_(True)		# unfreeze parameters
				self.rhs.train(mode=self.training)	# unfreeze spectral normalization

			if self.training:
				# RHS at the right endpoint
				rhs_r = self.h * self.rhs(self.t+1, y, param)
				y_fp  = explicit + self.theta * rhs_r
				# self.Lip = residual_fun(y)
				# y_fp2 = explicit + self.theta_h * self.rhs(self.t, y_fp, param)
				# self.Lip = (y_fp2-y_fp).pow(2).sum() / (y_fp-y).pow(2).sum()
				# self.Lip = (y.detach() - x.detach() - (1-self.theta) * self.h * self.rhs(self.t, x.detach(), param) - self.theta_h * self.rhs(self.t, y.detach(), param)).pow(2).sum()
				# xx = x.clone().detach().requires_grad_(True)
				# layer_Jac = jacobian_loss(xx if self.theta==1 else xx + (1-self.theta) * self.h * self.rhs(self.t, xx, param), xx)
				# fp_Jac = jacobian_loss(y_fp, y)
				# fp_Jac = divergence_loss(y_fp, y)
				if self.alpha_fpdiv>0:
					# fp_div = ( utils.divergence(rhs_r, y, create_graph=True) + 1.0 ).pow(2).mean()
					fp_div = self.divreg(rhs_r, y, create_graph=True)
					self.regularizers['fp_div'] = self.alpha_fpdiv * fp_div
					# self.regularizers['fp_div'] = self.alpha_fpdiv * (fp_div+1.0).pow(2)
					# self.regularizers['fp_div'] = rhs_r.pow(2).sum()
					# fp_Jac = self.reg(y, rhs_r)
					# fp_Jac = torch.trace( ( utils.jacobian(y_fp,y,create_graph=True).reshape((y.numel(),y.numel())) + 1.0 ).pow(2)) / y.size()[0]
					# self.regularizers['fp_rhs'] = rhs_r.pow(2).sum()
					# fp_Jac = jacobian_loss(lambda y: (explicit + theta_h * self.rhs(self.t, torch.reshape(y,x.shape), param)).flatten(), y)
					# self.regularizers['fp_Jac'] = self.alpha_rhsjac * fp_Jac
					# fp_Jac = self.reg(y, y_fp)
					# fp_Jac = torch.trace( ( utils.jacobian(y_fp,y,create_graph=True).reshape((y.numel(),y.numel())) + 1.0 ).pow(2)) / y.size()[0]
					# fp_Jac =( utils.jacobian(y_fp,y,create_graph=True).reshape((y.numel(),y.numel()))[1,1] + 1.0 ).pow(2)
					# self.regularizers['fp_Jac'] = self.alpha_rhsjac * fp_Jac
				y = linsolve_backprop.apply(self, y, y_fp)
				# layer_Jac = jacobian_loss(y, x.clone().detach().requires_grad_(True))
				# self.regularizers['layer_Jac'] = self.alpha_rhsjac * layer_Jac
		else:
			y = explicit


		# assert residual_fun(y).detach()<_TOL, "cg_step did not converge "+str(residual_fun(y).detach())
		if _collect_stat:
			mode = 'train/' if self.training else 'val/'
			#######################
			self.forward_stat[mode+'steps']        = self.forward_stat.get(mode+'steps',0)        + 1
			self.forward_stat[mode+'cg_iters']     = self.forward_stat.get(mode+'cg_iters',0)     + cg_iters[0]
			#######################
			self.forward_stat[mode+'residual_in']  = self.forward_stat.get(mode+'residual_in',0)  + residual_fun(x).detach()
			self.forward_stat[mode+'residual_out'] = self.forward_stat.get(mode+'residual_out',0) + residual[0] if isinstance(residual[0],int) else residual[0].detach() #residual_fun(y)
			#######################
			self.forward_stat[mode+'rhs_jac']      = self.forward_stat.get(mode+'rhs_jac',0)      + rhsjac.detach() if self.training and self.alpha_rhsjac>0  else 0
			self.forward_stat[mode+'rhs_div']      = self.forward_stat.get(mode+'rhs_div',0)      + rhsdiv.detach() if self.training and self.alpha_rhsdiv>0  else 0
			self.forward_stat[mode+'fp_div']       = self.forward_stat.get(mode+'fp_div',0)       + fp_div.detach() if self.training and self.alpha_fpdiv>0 and self.theta>0   else 0
			# self.forward_stat[mode+'fp_rhs']       = self.forward_stat.get(mode+'fp_rhs',0)       + self.regularizers['fp_rhs'].detach()  if self.training and self.alpha_fpdiv>0  else 0
			#######################

		cg_iters  = [0]
		residual  = [0]
		return y



##############################################################################################################
##############################################################################################################


class ode_step(Module):
	def __init__(self, rhs, t, h=1, theta=0.0, fp_iters=0, Lip=0.5, reg_iters=1, skip_connection=True):
		super(ode_step,self).__init__()

		self.skip_connection = skip_connection
		########################################
		self.rhs      = rhs
		self.t        = t
		########################################
		self.h         = h
		self.theta     = theta
		self.Lip       = Lip
		self.fp_iters  = fp_iters
		self.TOL       = 1.e-5
		self.reg_iters = reg_iters if theta>0 else 1
		self.max_fp_it = 50
		########################################
		self.stat  = { 'fstep': 0, 'residual': 0, 'cg_iters': 0, 'fp_iters': 0, 'guess_id': 0, 'init_res': 0, 'lin_resid': 0, 'lin_iters': 0, 'neu_iters': 0 }

		self.device = self.parameters().__next__().device
		self.dtype  = self.parameters().__next__().dtype

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

	def nsolve(self,y0,fixed_point_fun,tol=1.e-6):
		''' Nonlinear conjugate gradient solver '''
		def residual(x):
			with torch.enable_grad():
				return 0.5 * (x-fixed_point_fun(x)).pow(2).sum()
		def functional(x, t, d):
			with torch.enable_grad():
				# x.requires_grad_(True)
				z = x if d is None else x + t * d.view_as(x)
				fun  = residual(z.requires_grad_(True))
				dfun = torch.autograd.grad(fun, z)[0].view(-1)
				return fun, dfun

		maxiter = 100
		lr      = 1.e0

		y = y0.requires_grad_(True)
		with torch.no_grad():
			fun, dfun = functional(y,0,None)
			for niter in range(maxiter):
				delta = dfun.neg()
				# update the conjugate direction using Polak–Ribiere rule
				if niter==0:
					d = delta
					# t = min(1., 1. / dfun.abs().sum()) * lr
					t = lr
				else:
					# d = delta + (delta.dot(delta) / delta_old.dot(delta_old)) * d
					d = delta + (delta.dot(delta-delta_old) / delta_old.dot(delta_old)) * d
					t = lr
				# inexact line search
				fun, dfun, t, ls_func_evals = _strong_wolfe(functional, y, t, d, fun, dfun, dfun.dot(d))
				# print(niter,t,fun,"Hello",ls_func_evals,dfun.dot(d))
				# update solution
				y = y + t * d.view_as(y)
				# cache old delta
				delta_old = delta

				if (2*fun).sqrt()<=tol or dfun.abs().max()<=tol*0.01:
					break
		return y.detach(), fun, niter

	def scipy_nsolve(self,y0,residual_fun):
		def functional(z):
			with torch.enable_grad():
				z0   = torch.from_numpy(z).view_as(y0).to(device=self.device,dtype=self.dtype).requires_grad_(True)
				fun  = residual_fun(z0) #(z0-fixed_point_fun(z0)).flatten().norm()
				dfun = torch.autograd.grad(fun, z0)[0] # no retain_graph because derivative through nonlinearity is different at each iteration
			return fun.cpu().detach().numpy(), dfun.cpu().detach().numpy().ravel()
		opt_res = opt.minimize(functional, y0.cpu().numpy(), method='CG', jac=True, tol=self.TOL, options={'maxiter': 10,'disp': False})
		return torch.from_numpy(opt_res.x).view_as(y0).to(device=self.device,dtype=self.dtype), torch.tensor(opt_res.fun).to(device=self.device,dtype=self.dtype), opt_res.nit, opt_res.success

	class implicit_correction(Function):
		@staticmethod
		def forward(ctx,obj,y,z):
			ctx.save_for_backward(y,z)
			ctx.obj = obj
			return y

		@staticmethod
		def backward(ctx,dy):
			y, z, = ctx.saved_tensors
			ndof  = y.nelement()

			def A_dot(x):
				return torch.autograd.grad(z, y, grad_outputs=x, retain_graph=True, only_inputs=True)[0]
			def eval_residual(dx,dy):
				# \| (I-A) * dx - dy \|
				return (dx-A_dot(dx)-dy).flatten().norm()

			tol = atol = torch.tensor(ctx.obj.TOL)
			TOL = torch.max(tol*dy.norm(), atol)

			#######################################################################
			# try Neumann series

			# Constant approximation

			dx0 = dy
			Ady = A_dot(dy)
			r0  = Ady.flatten().norm()

			# Linear approximation
			dx1 = dy + Ady
			r1  = eval_residual(dx1,dy)

			matrix_power = 1
			while r1<0.98*r0 and r0>TOL and matrix_power<100:
				dx0 = dx1
				r0  = r1
				##################################
				# update solution
				# A^power * dy
				Ady = A_dot(Ady)
				dx1 = dx0 + Ady
				##################################
				# check residual
				r1  = eval_residual(dx1,dy)
				##################################
				matrix_power += 1
			ctx.obj.backward_stat['neu_iters'] = matrix_power

			if r0<TOL:
				ctx.obj.backward_stat['neu_residual'] = r1
				return None, None, dx0
			#######################################################################

			numpy_dtype = {torch.float: np.float, torch.float32: np.float32, torch.float64: np.float64}

			# 'Matrix-vector' product of the linear operator
			ctx.obj.backward_stat['lin_iters'] = 0
			def matvec(v):
				ctx.obj.stat['lin_iters'] = ctx.obj.stat['lin_iters'] + 1
				v0 = torch.from_numpy(v).view_as(y).to(device=ctx.obj.device,dtype=ctx.obj.dtype)
				Av = v0 - A_dot(v0)
				return Av.cpu().detach().numpy().ravel()
			A = sla.LinearOperator(dtype=numpy_dtype[ctx.obj.dtype], shape=(ndof,ndof), matvec=matvec)

			# Note that norm(residual) <= max(tol*norm(b), atol. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lgmres.html
			dx, info = sla.lgmres( A, dy.cpu().detach().numpy().ravel(), x0=dx0.cpu().detach().numpy().ravel(), maxiter=1000, tol=tol, atol=atol, M=None )
			dx = torch.from_numpy(dx).view_as(dy).to(device=ctx.obj.device,dtype=ctx.obj.dtype)
			ctx.obj.backward_stat['lin_resid'] = eval_residual(dx,dy)
			return None, None, dx

	def forward(self, y, x):
		self.residual = 0.0 #r1
		self.reg_Lip  = 0.0
		fp_iters = 0
		cg_iters = 0
		success  = True
		y_shape  = y.size()
		# flatten all non-batch dimensions
		y_in = y.reshape((y_shape[0],-1))
		x_in = x.reshape((x.size()[0],-1))

		# print(y_in, y)
		# print(x_in, x)
		# exit()

		theta_h = self.theta * self.h

		# explicit part
		explicit = y_in if self.theta==1 else y_in + (1-self.theta) * self.h * self.rhs(self.t, y_in, x_in)

		# fixed point map and residual
		def fixed_point_fun(x):
			y = explicit + theta_h * self.rhs(self.t, x, x_in)
			return y, (y-x).norm(dim=1).mean()
		# residual_fun = lambda x: (x-fixed_point_fun(x[0]).norm(dim=1).mean()
		def residual_fun(x0):
			x1, r1 = fixed_point_fun(x0)
			return r1


		if self.theta>0:
			y1, r1 = fixed_point_fun(y_in); fp_iters+=1
			while fp_iters<self.max_fp_it and r1>self.TOL:
				y0, r0 = y1, r1
				y1, r1 = fixed_point_fun(y0)
				Lipsch = r1/r0.detach()
				self.reg_Lip = self.reg_Lip + Lipsch
				if Lipsch>=self.Lip:
					y1, r1 = y0, r0
					break
				fp_iters+=1
			self.reg_Lip = self.reg_Lip / (fp_iters-1) if fp_iters>1 else 0.0
			y_out = y1

			if r1>self.TOL:
				success = False
			# elif self.training:
			# 	y = y_out.detach().requires_grad_(True)
			# 	y_fp, _ = fixed_point_fun(y)
			# 	y_out = ode_step.implicit_correction.apply(self, y, y_fp)
		else:
			y_out = explicit

		if self.training:
			self.stat['fstep'] += 1
			self.stat['residual']  = residual_fun(y_out).detach().numpy()
			self.stat['init_res']  = residual_fun(y_in).detach().numpy()
			self.stat['fp_iters']  = fp_iters
			self.stat['cg_iters']  = cg_iters
			self.stat['converged'] = int(success)
		return y_out

	# def forward(self, y):
	# 	self.residual = 0.0 #r1
	# 	self.reg_Lip  = 0.0
	# 	fp_iters = 0
	# 	cg_iters = 0
	# 	success  = True
	# 	y_shape  = y.size()
	# 	# flatten all non-batch dimensions
	# 	y_in = y.reshape((y_shape[0],-1))

	# 	theta_h = self.theta * self.h

	# 	# explicit part
	# 	explicit = y_in if self.theta==1 else y_in + (1-self.theta) * self.h * self.rhs(self.t, y_in)

	# 	# fixed point map and residual
	# 	# fixed_point_fun = lambda x: explicit + theta_h * self.rhs(self.t, x)
	# 	def fixed_point_fun(x):
	# 		y = explicit + theta_h * self.rhs(self.t, x)
	# 		return y, (y-x).norm(dim=1).mean()
	# 	# residual_fun = lambda x: (x-fixed_point_fun(x[0]).norm(dim=1).mean()
	# 	def residual_fun(x0):
	# 		x1, r1 = fixed_point_fun(x0)
	# 		return r1
	# 	# residual_fun = lambda x: (x-fixed_point_fun(x)).flatten().norm()


	# 	if self.theta>0:
	# 		# with torch.no_grad():
	# 			# try fixed-point iterations
	# 			# y0 = y.detach()
	# 			# r0 = residual_fun(y0)
	# 			# y1 = fixed_point_fun(y0)
	# 			# r1 = residual_fun(y1)
	# 			# fp_iters += 1
	# 			# while r1<0.98*r0 and r0>self.TOL and fp_iters<self.max_fp_it:
	# 			# 	y0 = y1
	# 			# 	r0 = r1
	# 			# 	y1 = fixed_point_fun(y0)
	# 			# 	r1 = residual_fun(y1)
	# 			# 	fp_iters += 1
	# 			# ##################################
	# 			# y0 = y.detach()
	# 			# y1 = fixed_point_fun(y0)
	# 			# r0 = (y1-y0).flatten().norm()
	# 			# fp_iters += 1
	# 			# ##################################
	# 			# y0 = y1
	# 			# y1 = fixed_point_fun(y0)
	# 			# r1 = (y1-y0).flatten().norm()
	# 			# fp_iters += 1
	# 			# ##################################
	# 			# while r1<0.98*r0 and r0>self.TOL and fp_iters<self.max_fp_it:
	# 			# 	y0 = y1
	# 			# 	y1 = fixed_point_fun(y0)
	# 			# 	r0 = r1
	# 			# 	r1 = (y1-y0).flatten().norm()
	# 			# 	fp_iters += 1
	# 			# ##################################

	# 			# ##############################################
	# 			# y0 = y_in.detach()
	# 			# y1, r1 = fixed_point_fun(y0); fp_iters+=1
	# 			# ##############################################
	# 			# while fp_iters<self.max_fp_it and r1>self.TOL:
	# 			# 	y0, r0 = y1, r1
	# 			# 	y1, r1 = fixed_point_fun(y0); fp_iters+=1
	# 			# 	Lip = r1/r0.detach()
	# 			# 	if Lip>=self.Lip:
	# 			# 		fp_iters -= 1
	# 			# 		y1, r1 = y0, r0
	# 			# 		break
	# 			# ##############################################
	# 			# y_out = y1

	# 			# if r1<r0:
	# 			# 	y0 = y1
	# 			# 	r0 = r1
	# 			# else:
	# 			# 	fp_iters -= 1

	# 			# switch to CG if necessary
	# 			# if r1>self.TOL: # and self.training:
	# 			# 	success = False
	# 				# y1, r1, cg_iters, success = self.scipy_nsolve(y0=y0, residual_fun=residual_fun)
	# 				# r1 = residual_fun(y1)
	# 				# # y, fmin, cg_iters = self.nsolve(y0=y0, fixed_point_fun=fixed_point_fun,tol=1.e-6*y0.size()[0])
	# 				# # assert success, "implicit solver has not converged, fmin ="+str(fmin)
	# 			# y = y1 if r1<r0 else y0

	# 		# # if nonlinear solver hasn't converged, use regularizer only
	# 		# if self.training and success:
	# 		# 	# if success:
	# 		# 	y.requires_grad_(True)
	# 		# 	y = ode_step.implicit_correction.apply(self, y, fixed_point_fun(y))
	# 		# 	# else:
	# 		# 	# 	y = y_in
	# 		# 	# 	for _ in range(fp_iters):
	# 		# 	# 		y = fixed_point_fun(y)

	# 		y1, r1 = fixed_point_fun(y_in); fp_iters+=1
	# 		# self.residual = 0.0 #r1
	# 		# self.reg_Lip  = 0.0
	# 		while fp_iters<self.max_fp_it and r1>self.TOL:
	# 			y0, r0 = y1, r1
	# 			y1, r1 = fixed_point_fun(y0)
	# 			Lipsch = r1/r0.detach()
	# 			# self.residual = Lipsch
	# 			self.reg_Lip = self.reg_Lip + Lipsch
	# 			if Lipsch>=self.Lip:
	# 				y1, r1 = y0, r0
	# 				break
	# 			fp_iters+=1
	# 		self.reg_Lip  = self.reg_Lip / fp_iters
	# 		y_out = y1

	# 		if r1>self.TOL:
	# 			success = False
	# 		# elif self.training:
	# 		# 	y = y_out.detach().requires_grad_(True)
	# 		# 	y_fp, _ = fixed_point_fun(y)
	# 		# 	y_out = ode_step.implicit_correction.apply(self, y, y_fp)
	# 	else:
	# 		y_out = explicit

	# 	if self.training:
	# 		# residual regularizer
	# 		# self.residual = residual_fun(y_in)

	# 		# # fp regularizer
	# 		# z1, r1 = fixed_point_fun(y_in)
	# 		# self.residual = r1
	# 		# self.reg_Lip  = 0.0
	# 		# if self.theta>0:
	# 		# 	# num_iters = min(self.reg_iters,fp_iters)
	# 		# 	# z1, r1 = fixed_point_fun(y_in)
	# 		# 	# self.residual = r1
	# 		# 	for it in range(1,fp_iters+1):
	# 		# 		z0, r0 = z1, r1
	# 		# 		z1, r1 = fixed_point_fun(z0)
	# 		# 		self.Lip = self.Lip + r1 / r0.detach()
	# 		# 	self.Lip = self.Lip / fp_iters
	# 				# self.residual = self.residual + (r1 - self.Lip*r0).abs() * it / num_iters
	# 				# self.residual = self.residual + self.Lip**(num_iters-1-it) * (z1-z0).flatten().norm()
	# 				# beta = (self.max_fp_it - it) / self.max_fp_it # decreasing
	# 				# beta = (it+1) / num_iters # increasing
	# 				# self.residual = self.residual + self.Lip**(num_iters-1-it) * residual_fun(z)
	# 				# self.residual = self.residual + beta**2 * residual_fun(z)
	# 			# self.residual = self.residual / (num_iters-1) + residual_fun(z1)
	# 			# self.residual = self.residual + r1
	# 			# if success:
	# 			# 	y_out.requires_grad_(True)
	# 			# 	y_fp, _ = fixed_point_fun(y_out)
	# 			# 	y_out = ode_step.implicit_correction.apply(self, y_out, y_fp)
	# 			# else:
	# 			# 	y_out = z0
	# 		# record stat
	# 		# if r0>self.TOL or cg_iters>0:
	# 		self.stat['fstep'] += 1
	# 		self.stat['residual']  = residual_fun(y_out).detach().numpy()
	# 		self.stat['init_res']  = residual_fun(y_in).detach().numpy()
	# 		# self.stat['guess_id']  = guess_id
	# 		self.stat['fp_iters']  = fp_iters
	# 		self.stat['cg_iters']  = cg_iters
	# 		self.stat['converged'] = int(success)
	# 	# else:
	# 		# z0 = y_in
	# 		# z1 = fixed_point_fun(z0)
	# 		# r1 = (z1-z0).flatten().norm()
	# 		# for it in range(fp_iters):
	# 		# 	z0 = z1
	# 		# 	r0 = r1
	# 		# 	z1 = fixed_point_fun(z0)
	# 		# 	r1 = (z1-z0).flatten().norm()
	# 		# 	print(it,(r1/r0).detach().numpy(),r0.detach().numpy(),r1.detach().numpy(),self.Lip**(fp_iters-1-it))
	# 		# exit()

	# 	return y_out


	# def forward(self, y):
	# 	if self.training:
	# 		y_in = y.clone()
	# 		self.residual = torch.tensor(0.0)
	# 		self.residual_noise = torch.tensor(0.0)
	# 		r0 = 0
	# 		guess_id = 0
	# 		fp_iters = 0
	# 		cg_iters = 0
	# 		success  = True

	# 	theta_h = self.theta * self.h

	# 	# explicit part
	# 	explicit = 0 if self.theta==1 else (1-self.theta) * self.h * self.rhs(self.t, y_in)
	# 	if self.skip_connection: explicit = y_in + explicit
	# 	# if self.theta == 0: return explicit

	# 	# fixed point map and residual
	# 	fixed_point_fun = lambda x: explicit + theta_h * self.rhs(self.t, x)
	# 	residual_fun    = lambda x: (x-fixed_point_fun(x)).flatten().norm()

	# 	if self.theta>0:
	# 		# given number of fixed point iterations (if specified)
	# 		if self.fp_iters>0:
	# 			fp_iters = self.fp_iters
	# 			# r0 = residual_fun(y).detach()
	# 			# for _ in range(self.fp_iters):
	# 			# 	y = fixed_point_fun(y)
	# 			# self.residual = residual_fun(y)
	# 		else:
	# 			with torch.no_grad():
	# 				########################################################################################
	# 				# choose best initial guess
	# 				def check_init_guess(y0,r0,y1,guess_id):
	# 					if not torch.isnan(r0) and not torch.isinf(r0):
	# 						r1 = residual_fun(y1)
	# 						if r1<r0 and not torch.isnan(r1) and not torch.isinf(r1):
	# 							return y1, r1, guess_id+1
	# 						else:
	# 							return y0, r0, guess_id
	# 					else:
	# 						return y0, r0, guess_id

	# 				# current value as initial guess
	# 				y0 = y.detach()
	# 				r0 = residual_fun(y0)
	# 				guess_id = 0

	# 				# explicit part as initial guess
	# 				if r0>self.TOL:
	# 					y0, r0, guess_id = check_init_guess( y0, r0, explicit.detach(), guess_id )

	# 				# # forward Euler as initial guess
	# 				# if r0>self.TOL:
	# 				# 	y1 = y.detach()
	# 				# 	trial_steps = 5; h_steps = self.h/trial_steps
	# 				# 	for _ in range(trial_steps):
	# 				# 		y1 = y1 + h_steps * self.rhs(self.t, y1) if self.skip_connection \
	# 				# 		     else h_steps * self.rhs(self.t, y1)
	# 				# 		y0, r1, guess_id = check_init_guess( y0, r0, y1, guess_id )
	# 				# 		if r1>=r0: break
	# 				# 		r0 = r1

	# 				########################################################################################
	# 				# solve nonlinear system

	# 				# try fixed-point iterations
	# 				if r0>self.TOL:
	# 					y1 = fixed_point_fun(y0)
	# 					r1 = residual_fun(y1)
	# 					while r1<0.98*r0 and r0>self.TOL and fp_iters<self.max_fp_it:
	# 						y0 = y1
	# 						r0 = r1
	# 						y1 = fixed_point_fun(y0)
	# 						r1 = residual_fun(y1)
	# 						fp_iters += 1
	# 				y = y0

	# 				# switch to CG if necessary
	# 				if r0>self.TOL:
	# 					success = False
	# 					y, fmin, cg_iters, success = self.scipy_nsolve(y0=y0, fixed_point_fun=fixed_point_fun)
	# 					# # y, fmin, cg_iters = self.nsolve(y0=y0, fixed_point_fun=fixed_point_fun,tol=1.e-6*y0.size()[0])
	# 					# # assert success, "implicit solver has not converged, fmin ="+str(fmin)

	# 				########################################################################################
	# 			# if nonlinear solver hasn't converged, use regularizer only
	# 			if success:
	# 				y.requires_grad_(True)
	# 				y = ode_step.implicit_correction.apply(self, y, fixed_point_fun(y))
	# 			else:
	# 				# y = y_fp
	# 				y = y_in
	# 	else:
	# 		y = explicit


	# 	if self.training:
	# 		# residual regularizer
	# 		y_fp = y_in
	# 		for _ in range(self.reg_iters):
	# 			y_fp = fixed_point_fun(y_fp)
	# 		self.residual = residual_fun(y_fp)

	# 		# noise dumping
	# 		if self.noise_std>0:
	# 			noise = self.noise_std * (torch.rand_like(y_in, requires_grad=False) - 0.5)
	# 			z_fp  = y_in + noise
	# 			explicit_noise = 0 if self.theta==1 else (1-self.theta) * self.h * self.rhs(self.t, z_fp)
	# 			if self.skip_connection: explicit_noise = z_fp + explicit_noise
	# 			for _ in range(self.reg_iters):
	# 				z_fp = explicit_noise + theta_h * self.rhs(self.t, z_fp)
	# 			self.residual_noise = (z_fp-y_fp).flatten().norm()

	# 		# record stat
	# 		# if r0>self.TOL or cg_iters>0:
	# 		self.stat['fstep']   += 1
	# 		self.stat['residual'] = residual_fun(y).detach().numpy()
	# 		self.stat['init_res'] = r0
	# 		self.stat['guess_id'] = guess_id
	# 		self.stat['fp_iters'] = fp_iters
	# 		self.stat['cg_iters'] = cg_iters
	# 		self.stat['success']  = int(success)

	# 	return y

	# def forward(self, y, z=None):
	# 	if self.training:
	# 		self.residual = torch.tensor(0.0)
	# 		self.residual_noise = torch.tensor(0.0)
	# 		r0 = 0
	# 		guess_id = 0
	# 	fp_iters = 0
	# 	cg_iters = 0
	# 	success  = True

	# 	theta_h = self.theta * self.h

	# 	# explicit part
	# 	explicit = y if self.theta==1 else y + (1-self.theta) * self.h * self.rhs(self.t, y)

	# 	# fixed point map and residual
	# 	fixed_point_fun = lambda x: explicit + theta_h * self.rhs(self.t, x)
	# 	residual_fun    = lambda x: (x-fixed_point_fun(x)).flatten().norm()

	# 	if self.theta>0:
	# 		with torch.no_grad():
	# 			# try fixed-point iterations
	# 			y0 = y.detach()
	# 			r0 = residual_fun(y0)
	# 			y1 = fixed_point_fun(y0)
	# 			r1 = residual_fun(y1)
	# 			fp_iters += 1
	# 			while r1<0.98*r0 and r0>self.TOL and fp_iters<self.max_fp_it:
	# 				y0 = y1
	# 				r0 = r1
	# 				y1 = fixed_point_fun(y0)
	# 				r1 = residual_fun(y1)
	# 				fp_iters += 1

	# 			if r1<r0:
	# 				y0 = y1
	# 				r0 = r1
	# 			else:
	# 				fp_iters -= 1

	# 			# switch to CG if necessary
	# 			if r0>self.TOL:
	# 				success = False
	# 				y1, fmin, cg_iters, success = self.scipy_nsolve(y0=y0, fixed_point_fun=fixed_point_fun)
	# 				r1 = residual_fun(y1)
	# 				# # y, fmin, cg_iters = self.nsolve(y0=y0, fixed_point_fun=fixed_point_fun,tol=1.e-6*y0.size()[0])
	# 				# # assert success, "implicit solver has not converged, fmin ="+str(fmin)
	# 			y = y1 if r1<r0 else y0

	# 		# if nonlinear solver hasn't converged, use regularizer only
	# 		if success:
	# 			y.requires_grad_(True)
	# 			y = ode_step.implicit_correction.apply(self, y, fixed_point_fun(y))
	# 		else:
	# 			y = fixed_point_fun(y)
	# 			# y = y_in
	# 	else:
	# 		y = explicit
	# 		# y_fp = explicit
	# 		# fp_iters = 1

	# 	if self.training:
	# 		# residual regularizer
	# 		self.residual = residual_fun(y)

	# 		# noise dumping
	# 		if self.noise_std>0 and success:
	# 			explicit_noise = z if self.theta==1 else z + (1-self.theta) * self.h * self.rhs(self.t, z)
	# 			fixed_point_fun_noise = lambda x: explicit_noise + theta_h * self.rhs(self.t, x)
	# 			residual_fun_noise    = lambda x: (x-fixed_point_fun_noise(x)).flatten().norm()

	# 			if self.theta>0:
	# 				for _ in range(fp_iters):
	# 					z = fixed_point_fun_noise(z)
	# 			else:
	# 				z = explicit_noise

	# 			# self.residual_noise = (z[:,1]-y[:,1]).flatten().norm()
	# 			self.residual_noise = (z-y).flatten().norm()
	# 			self.residual = self.residual + residual_fun_noise(z)
	# 		else:
	# 			z = y

	# 		# record stat
	# 		# if r0>self.TOL or cg_iters>0:
	# 		self.stat['fstep']   += 1
	# 		self.stat['residual'] = residual_fun(y).detach().numpy()
	# 		self.stat['init_res'] = r0
	# 		self.stat['guess_id'] = guess_id
	# 		self.stat['fp_iters'] = fp_iters
	# 		self.stat['cg_iters'] = cg_iters
	# 		self.stat['success']  = int(success)

	# 	return y, z


class ode_solver(Module):
	def __init__(self, rhs, T, num_steps, theta=0.0, fp_iters=0, Lip=0.5, reg_iters=1, solver='cg', alpha_rhsjac=0, alpha_rhsdiv=0, alpha_fpjac=0, alpha_fpdiv=0):
		super(ode_solver,self).__init__()
		global _ode_count
		self.name = str(_ode_count)+".ode"
		_ode_count += 1

		# self.ode_step = torch.nn.ModuleList([ode_step(rhs, step, theta=theta, h=T/num_steps, fp_iters=fp_iters, skip_connection=True) for step in range(num_steps)])
		self.rhs = rhs # need this to register parameters
		# self.ode_step = torch.nn.ModuleList([ode_step(rhs, step, theta=theta, h=T/num_steps, fp_iters=fp_iters, Lip=Lip, reg_iters=reg_iters, skip_connection=True) for step in range(num_steps)])
		if solver=='fp':
			self.ode_step = torch.nn.ModuleList([fp_step(rhs, step, theta=theta, h=T/num_steps, alpha_rhsjac=alpha_rhsjac, alpha_rhsdiv=alpha_rhsdiv, alpha_fpdiv=(alpha_fpdiv if step==num_steps-1 else 0)) for step in range(num_steps)])
		elif solver=='cg':
			self.ode_step = torch.nn.ModuleList([cg_step(rhs, step, theta=theta, h=T/num_steps, alpha_rhsjac=alpha_rhsjac, alpha_rhsdiv=alpha_rhsdiv, alpha_fpdiv=(alpha_fpdiv if step==num_steps-1 else 0)) for step in range(num_steps)])

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

	@property
	def regularizer(self):
		regularizers = {}
		# for step in range(self.num_steps):
		for step in [self.num_steps-1]:
			for key in getattr(self.ode_step[step],'regularizers',{}).keys():
				regularizers[key] = regularizers.get(key,0) + self.ode_step[step].regularizers[key] / self.num_steps
		# for key in getattr(self.ode_step[step],'regularizers',{}).keys():
		# 	regularizers[key] = regularizers[key]
		return regularizers


	########################################

	def forward(self, y0, param=None):
		y = [y0]
		for step in range(self.num_steps):
			y.append(self.ode_step[step](y[-1], param))

			# if self.training:
			# 	for key in getattr(self.ode_step[step],'regularizers',{}).keys():
			# 		self.regularizers[key] = self.regularizers.get(key,0) + self.ode_step[step].regularizers[key] / self.num_steps
		return torch.stack(y)


	# def forward(self, y_in):
	# 	if self.training:
	# 		self.residual       = torch.tensor(0.0)
	# 		self.residual_noise = torch.tensor(0.0)
	# 		self.stat['fstep']   += 1
	# 		self.stat['init_res'] = 0.0
	# 		self.stat['residual'] = 0.0
	# 		self.stat['fp_iters'] = 0
	# 		self.stat['cg_iters'] = 0
	# 		self.stat['lin_residual']  = 0.0
	# 		self.stat['lin_iters']     = 0
	# 		self.stat['neumann_iters'] = 0
	# 		self.stat['success'] = 1

	# 	if self.cache_hidden:
	# 		y_out = y_in[0].repeat(self.num_steps+1,*(y_in[0].ndim*[1]))

	# 	# y = [ yy for yy in y_in ]
	# 	y = list(y_in) # copy input list
	# 	for step in range(self.num_steps):
	# 		y[0] = self.ode_step[step](y[0])
	# 		# y[0][step+1] = self.ode_step[step](y[0][step])
	# 		if self.cache_hidden:
	# 			y_out[step+1] = y[0]

	# 		if self.training:
	# 			# regularizers
	# 			self.residual = self.residual + self.ode_step[step].residual
	# 			# forward stat
	# 			self.stat['init_res'] = self.stat['init_res'] + self.ode_step[step].stat['init_res']
	# 			self.stat['residual'] = self.stat['residual'] + self.ode_step[step].stat['residual']
	# 			self.stat['fp_iters'] = self.stat['fp_iters'] + self.ode_step[step].stat['fp_iters']
	# 			self.stat['cg_iters'] = self.stat['cg_iters'] + self.ode_step[step].stat['cg_iters']
	# 			# backward stat
	# 			self.stat['lin_residual']  = self.stat['lin_residual']  + self.ode_step[step].stat['lin_residual']
	# 			self.stat['lin_iters']     = self.stat['lin_iters']     + self.ode_step[step].stat['lin_iters']
	# 			self.stat['neumann_iters'] = self.stat['neumann_iters'] + self.ode_step[step].stat['neumann_iters']
	# 			# convergence?
	# 			self.stat['success'] = self.stat['success'] * self.ode_step[step].stat['success']

	# 			self.ode_step[step].stat['lin_residual']  = 0.0
	# 			self.ode_step[step].stat['lin_iters']     = 0
	# 			self.ode_step[step].stat['neumann_iters'] = 0

	# 			# beta = (self.num_steps - step) / self.num_steps # decreasing
	# 			# beta = (step+1) / self.num_steps # increasing
	# 			# beta = 1
	# 			beta = 0 if step<(self.num_steps-1) else 1
	# 			for i in range(1,len(y)):
	# 				y[i] = self.ode_step[step](y[i])
	# 				# if self.ode_step[step].stat['success']:
	# 				dy = (y[i] - y[0])[:,y[0].size()[1]//2:]
	# 				# dy[:,0] = 0.1 * dy[:,0]
	# 				# dy[:,1] = 1 * dy[:,1]
	# 				self.residual_noise = self.residual_noise + beta**5 / (len(y)-1) * dy.flatten().norm()

	# 	if self.cache_hidden:
	# 		return y_out
	# 	else:
	# 		return y[0]

	# def regularizer(self, alpha=0.0):
	# 	if alpha>0:
	# 		return alpha * 1.e1 * self.residual
	# 	else:
	# 		return 0.0
	# def regularizer_noise(self, alpha=0.0):
	# 	if alpha>0:
	# 		return alpha * self.alpha_beta * self.residual_noise
	# 	else:
	# 		return 0.0
