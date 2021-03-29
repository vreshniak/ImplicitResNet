import time
import warnings
from collections import deque
from abc import ABCMeta, abstractmethod

import torch
from ..solvers import nsolve, linsolve
from ..utils import calc


###############################################################################
###############################################################################
# global parameters


_nsolver    = 'lbfgs'
_lin_solver = 'gmres' #scipy_lgmres

_TOL = 1.e-8
_max_iters = 200
_max_lin_iters = 20

_collect_stat = True
_collect_rhs_stat = False
_debug = False


###############################################################################
###############################################################################


class linsolve_backprop(torch.autograd.Function):
	@staticmethod
	def forward(ctx, self, y, fpx, fpy):
		ctx.mark_non_differentiable(y)
		ctx.save_for_backward(y, fpy)
		ctx.self = self
		return fpx

	@staticmethod
	def backward(ctx, dy):
		y, y_fp,  = ctx.saved_tensors
		batch_dim = dy.size(0)

		start = time.time()

		if _debug:
			pre_grad = {}
			for name, param in ctx.self.named_parameters():
				pre_grad[name] = param.grad

		# 'Matrix-vector' product of the linear operator
		def matvec(v):
			v0 = v.view_as(y)
			Av = v0 - torch.autograd.grad(y_fp, y, grad_outputs=v0, create_graph=False, retain_graph=True, only_inputs=True)[0]
			return Av.reshape((batch_dim,-1))

		dx, error, lin_iters, flag = linsolve( matvec, dy.reshape((batch_dim,-1)), dy.reshape((batch_dim,-1)), _lin_solver, max_iters=_max_lin_iters)
		dx = dx.view_as(dy)

		assert not torch.isnan(error), "NaN value in the backprop error of the linear solver for ode %s"%(_nsolver, ctx.self.name)
		if _debug:
			resid1 = matvec(dy).sum()
			resid2 = matvec(dy).sum()
			assert resid1==resid2, "spectral normalization not frozen in backprop, delta_residual=%.2e"%((resid1-resid2).abs())
			for name, param in ctx.self.named_parameters():
				assert param.grad is None or (param.grad-pre_grad[name]).sum()==0, "linsolver propagated gradients to parameters"
		if flag>0: warnings.warn("%s in backprop didn't converge for ode %s, error is %.2E"%(_lin_solver, ctx.self.name, error))


		stop = time.time()

		if _collect_stat:
			ctx.self._stat['backward/steps']        = ctx.self._stat.get('backward/steps',0)        + 1
			ctx.self._stat['backward/walltime']     = ctx.self._stat.get('backward/walltime',0)     + (stop-start)
			ctx.self._stat['backward/lin_residual'] = ctx.self._stat.get('backward/lin_residual',0) + error
			ctx.self._stat['backward/lin_iters']    = ctx.self._stat.get('backward/lin_iters',0)    + lin_iters

		return None, None, dx, None



###############################################################################
###############################################################################



class ode_solver(torch.nn.Module, metaclass=ABCMeta):
	def __init__(self, rhs, T, num_steps, t_out=None, ind_out=None):
		super().__init__()

		# ODE count in a network as a class property
		self.__class__.ode_count = getattr(self.__class__,'ode_count',-1) + 1
		self.name = str(self.__class__.ode_count)+".ode"

		self.rhs = rhs # this is registered as a submodule
		self.register_buffer('_T', torch.tensor(T))
		self.register_buffer('_num_steps', torch.tensor(num_steps))
		self.register_buffer('_h', torch.tensor(T/num_steps))

		# self.evolution = False
		self._t = deque([], maxlen=num_steps+1)
		self._y = deque([], maxlen=num_steps+1)
		interp_coef = None
		if t_out is not None:
			assert torch.is_tensor(t_out) and t_out.ndim==1, "t_out must a 1d tensor, got %s"%(t_out)
			if t_out.numel()>1: assert torch.amin(t_out[1:]-t_out[:-1])>0, "t_out must be increasing sequence"
			assert t_out[0]>=0 and t_out[-1]<=T, "t_out must have values in [0,T]"
			ind_out     = (t_out/self._h).long()
			interp_coef = t_out/self._h - ind_out
		if ind_out is not None:
			assert t_out is None, "either t_out or ind_out can be given, not both"
			assert torch.is_tensor(ind_out) and ind_out.ndim==1, "ind_out must a 1d tensor, got %s"%(ind_out)
			if ind_out.numel()>1: assert torch.amin(ind_out[1:]-ind_out[:-1])>0, "ind_out must be increasing sequence"
			assert ind_out[0]>=0 and ind_out[-1]<=num_steps, "ind_out must have values in [0,num_steps]"
		self.register_buffer('_ind_out',     ind_out)
		self.register_buffer('_interp_coef', interp_coef)
		self._stat = {}

	########################################

	@property
	def num_steps(self):
		return self._num_steps
	@num_steps.setter
	def num_steps(self, num_steps):
		if self._interp_coef is not None:
			t_out = self._h * (self._ind_out + self._interp_coef)
		elif self._ind_out is not None:
			h = self._h
		self._num_steps.fill_(num_steps)
		self._h = self.T / num_steps
		self._t = deque([], maxlen=num_steps+1)
		self._y = deque([], maxlen=num_steps+1)
		if self._interp_coef is not None:
			self._ind_out     = (t_out/self._h).long()
			self._interp_coef = t_out/self._h - ind_out
		elif self._ind_out is not None:
			self._ind_out = ((h/self._h)*self._ind_out).long()

	@property
	def T(self):
		return self._T
	@T.setter
	def T(self, T):
		if self._interp_coef is not None:
			t_out = self._h * (self._ind_out + self._interp_coef)
		elif self._ind_out is not None:
			h = self._h
		self._T.fill_(T)
		self._h = T / self.num_steps
		if self._interp_coef is not None:
			self._ind_out     = (t_out/self._h).long()
			self._interp_coef = t_out/self._h - ind_out
		elif self._ind_out is not None:
			self._ind_out = ((h/self._h)*self._ind_out).long()

	@property
	def h(self):
		return self._h

	@property
	def ind_out(self):
		return self._ind_out
	@ind_out.setter
	def ind_out(self, ind_out):
		self._ind_out = ind_out

	@property
	def t_out(self):
		if self._interp_coef is not None:
			t_out = self._h * (self._ind_out + self._interp_coef)
		elif self._ind_out is not None:
			t_out = self._h * self._ind_out
		else:
			return None
		return t_out
	@t_out.setter
	def t_out(self, t_out):
		self._ind_out     = (t_out/self._h).long()
		self._interp_coef = t_out/self._h - ind_out

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

	########################################


	def forward(self, y0, t0=0, return_t=False):
		if self.training or self._ind_out is not None:
			# evaluate solution at the grid points
			self._t.append(t0+0*self.h) # to make it a tensor
			self._y.append(y0)
			for step in range(1,self.num_steps+1):
				self._y.append(self.ode_step(self._t[-1], self._y[-1]))
				self._t.append(t0+step*self.h)

			# evaluate solution at the given points
			if self._interp_coef is not None:
				# returns tensor of shape (batch size, time points, hidden dim)
				y = [ ((1-c[i])*self._y[i] if c[i]<1 else 0) + (c[i]*self._y[i+1] if c[i]>0 else 0) for i, c in zip(self._ind_out, self._interp_coef) ]
				y = torch.stack(y, 1)
				if return_t:
					t = [ ((1-c[i])*self._t[i] if c[i]<1 else 0) + (c[i]*self._t[i+1] if c[i]>0 else 0) for i, c in zip(self._ind_out, self._interp_coef) ]
					t = torch.stack(t) if t[0].ndim==0 else torch.stack(t, 1)
			elif self._ind_out is not None:
				y = [ self._y[i] for i in self._ind_out ]
				y = torch.stack(y, 1)
				if return_t:
					t = [ self._t[i] for i in self._ind_out ]
					t = torch.stack(t) if t[0].ndim==0 else torch.stack(t, 1)
			else:
				t = self._t[-1]
				y = self._y[-1]

			if return_t:
				return t, y
			else:
				return y
		else:
			y = y0
			for step in range(self.num_steps):
				y = self.ode_step(t0+step*self.h, y)
			return y



###############################################################################
###############################################################################



class theta_solver(ode_solver):
	def __init__(self, rhs, T, num_steps, theta, t_out=None, ind_out=None, tol=_TOL):
		super().__init__(rhs, T, num_steps, t_out, ind_out)

		self.register_buffer('_theta', torch.tensor(theta))
		self.tol = tol

	########################################

	@property
	def theta(self):
		return self._theta
	@theta.setter
	def theta(self, theta):
		self._theta.fill_(theta)

	@property
	def statistics(self, reset=True):
		stat = super().statistics
		stat['hparams/%s_theta'%(self.name)] = self.theta
		return stat

	########################################


	def step_fun(self, t, x, y):
		# if theta==1, use left endpoint
		t = (t + self.theta * self.h) if self.theta<1 else t
		z = ((1-self.theta)*x if self.theta<1 else 0) + (self.theta*y if self.theta>0 else 0)
		return self.h * self.rhs(t, z)


	def residual(self, t, x, y, fval=None, param=None):
		batch_dim = x.size(0)
		return ( y - x - self.step_fun(t,x,y) ).pow(2).reshape((batch_dim,-1)).sum(dim=1)


	def ode_step(self, t, x):
		iters = resid = flag = 0

		start = time.time()
		if self.theta>0:
			residual_fn = lambda z: self.residual(t, x.detach(), z)

			# check initial residual and, as a side effect, perform spectral normalization or other forward hooks
			init_resid = residual_fn(x).amax().detach()

			if init_resid<=self.tol:
				y = x.detach()
			else:
				# no need to freeze parameters, i.e. self.requires_grad_(False) as this should be taken care of in the nsolver
				# spectral normalization has to be performed only once per forward pass, so freeze it here
				self.rhs.eval() # note that self.training remains unchanged

				# solve nonlinear system
				y, resid, iters, flag = nsolve( residual_fn, x, _nsolver, tol=self.tol, max_iters=_max_iters )

				# assert not torch.isnan(resid), "NaN value in the residual of the %s nsolver for layer %s at t=%d"%(_nsolver, self.name, t)
				if _debug:
					assert init_resid==residual_fn(x).amax().detach(), "spectral normalization not frozen, delta_residual=%.2e"%((init_resid-residual_fn(x).amax()).abs())
					if self.training:
						for param in self.parameters():
							assert param.grad is None or param.grad.sum()==0, "nsolver propagated gradients to parameters"

				# unfreeze spectral normalization
				self.rhs.train(mode=self.training)

			if flag>0: warnings.warn("%s nonlinear solver didn't converge for layer %s at t=%d, error is %.2E"%(_nsolver, self.name, t, resid))

			# NOTE: two evaluations of step_fun are requried anyway!
			y.requires_grad_(True)
			y = linsolve_backprop.apply(self, y, x + self.step_fun(t, x, y.detach()), self.step_fun(t, x.detach(), y))
		else:
			y = x + self.step_fun(t, x, None)
		stop = time.time()

		if _collect_stat:
			mode = 'forward_train/' if self.training else 'forward_valid/'
			#######################
			self._stat[mode+'steps']    = self._stat.get(mode+'steps',0)    + 1
			self._stat[mode+'walltime'] = self._stat.get(mode+'walltime',0) + (stop-start)
			self._stat[mode+'iters']    = self._stat.get(mode+'iters',0)    + iters
			self._stat[mode+'residual'] = self._stat.get(mode+'residual',0) + resid
		return y



###############################################################################
###############################################################################



def theta_stability_fun(theta, x):
	return (1+(1-theta)*x) / (1-theta*x)

def theta_inv_stability_fun(theta, y):
	y = max(y, 1 - 1.0/(theta+1.e-12) + 1.e-6)
	return (1-y) / ((1-y)*theta-1)

def trapz(y, dx=1.0):
	res = 0.5 * y[0]
	for i in range(1,len(y)-1):
		res = res + y[i]
	res = res + 0.5 * y[-1]
	return res * dx


# forward hook to evaluate regularizers and statistics
def compute_regularizers_and_statistics(solver, input, output):
	reg   = {}
	stat  = {}
	if solver.training:
		rhs   = solver.rhs
		name  = solver.name+'_'
		alpha = solver.alpha

		n     = len(solver._t)
		steps = n - 1
		dim   = input[0].numel() / input[0].size(0)
		dx    = 1 / steps / dim

		# spectral normalization has to be performed only once per forward pass, so freeze here
		rhs.eval()

		# contribution of divergence along the trajectory
		c = (((t+1)/n)**solver.p for t in range(n))

		# evaluate divergence and jacobian along the trajectory
		if alpha['div']!=0 or alpha['jac']!=0 or solver.collect_rhs_stat:
			divjac = [ calc.trace_and_jacobian( lambda x: rhs(t, x), y, n=solver.mciters) for t, y in zip(solver._t, solver._y) ]

		if solver.collect_rhs_stat:
			stat['rhs/'+name+'div'] = trapz([divt[0] for divt in divjac], dx)
			stat['rhs/'+name+'jac'] = trapz([jact[1] for jact in divjac], dx=dx/dim)
			stat['rhs/'+name+'f']   = trapz([rhs(t,y).pow(2).sum() for t, y in zip(solver._t, solver._y)], dx)

		# divergence
		if alpha['div']!=0:
			reg[name+'div'] = alpha['div'] * torch.sigmoid( trapz([ct*divt[0] for ct, divt in zip(c, divjac)], dx) )

		# jacobian
		if alpha['jac']!=0:
			reg[name+'jac'] = alpha['jac'] * (stat['rhs/'+name+'jac'] if solver.collect_rhs_stat else trapz([jact[1] for jact in divjac], dx=dx/dim))

		# magnitude
		if alpha['f']!=0:
			reg[name+'f'] = alpha['f'] * (stat['rhs/'+name+'f'] if solver.collect_rhs_stat else trapz([rhs(t,y).pow(2).sum() for t, y in zip(solver._t, solver._y)], dx))

		# residual
		if alpha['resid']!=0:
			for step in range(n-1):
				x, y = solver._y[step].detach(), solver._y[step+1].detach()
				reg[name+'residual'] = reg.get(name+'residual',0) + solver.residual(solver._t[step], x, y)
			reg[name+'residual'] = (alpha['resid'] / steps) * reg[name+'residual']

		# 'Total variation'
		if alpha['TV']!=0 and len(rhs.F)>1:
			w1 = torch.nn.utils.parameters_to_vector(rhs.F[0].parameters())
			for t in range(len(rhs.F)-1):
				w2 = torch.nn.utils.parameters_to_vector(rhs.F[t+1].parameters())
				reg[name+'TV'] = reg.get(name+'TV',0) + ( w2 - w1 ).pow(2).sum()
				w1 = w2
			reg[name+'TV'] = (alpha['TV'] / steps) * reg[name+'TV']

		# note that solver.training has not been changed by rhs.eval()
		rhs.train(mode=solver.training)

		if not hasattr(solver, 'regularizer'):
			setattr(solver, 'regularizer', reg)
		else:
			for key, val in reg.items():
				solver.regularizer[key] = val
		if solver.collect_rhs_stat:
			stat['rhs/'+name+'jac'] = stat['rhs/'+name+'jac'].sqrt()
			stat['rhs/'+name+'f']   = stat['rhs/'+name+'f'].sqrt()
			setattr(rhs, 'statistics', stat)
	return None


def regularized_ode_solver(solver, alpha={}, stability_limits=None, mciters=1, p=2, collect_rhs_stat=_collect_rhs_stat, augmentation_loss=None, perturbation=None, perturbation_loss=None):
	if 'div'   not in alpha: alpha['div']   = 0.0
	if 'jac'   not in alpha: alpha['jac']   = 0.0
	if 'f'     not in alpha: alpha['f']     = 0.0
	if 'resid' not in alpha: alpha['resid'] = 0.0
	if 'TV'    not in alpha: alpha['TV']    = 0.0
	setattr(solver, 'alpha', alpha)
	setattr(solver, 'mciters', mciters)
	setattr(solver, 'p', p)
	setattr(solver, 'collect_rhs_stat', collect_rhs_stat)
	if stability_limits is not None:
		min_eig = max(-10, theta_inv_stability_fun(solver.theta, stability_limits[0]) )
		max_eig = min( 10, theta_inv_stability_fun(solver.theta, stability_limits[2]) )
		ini_eig = theta_inv_stability_fun(solver.theta, stability_limits[1])
		solver.rhs.set_spectral_limits([min_eig,ini_eig,max_eig])
	solver.register_forward_hook(compute_regularizers_and_statistics)
	return solver
