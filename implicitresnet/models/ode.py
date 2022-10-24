import time
import warnings
from collections import deque
from abc import ABCMeta, abstractmethod
from typing import Union, List, Optional
from types import MethodType

import torch
from ..solvers import nsolve, linsolve
from ..utils import calc
from ..utils.spectral import spectral_norm
from .rhs import RHS, restrict_theta_stability



_TNum = Union[int, float]


###############################################################################
###############################################################################
# global parameters


_nsolver    = 'lbfgs'
_lin_solver = 'gmres' #scipy_lgmres

_TOL = 1.e-8
_max_iters = 100
_max_lin_iters = 20

_collect_stat = True
_collect_rhs_stat = False
_debug = False


###############################################################################
###############################################################################
# helper functions


def addprop(inst, name, method):
	r'''Add property to existing instance of a class
	https://stackoverflow.com/questions/2954331/dynamically-adding-property-in-python
	'''
	cls = type(inst)
	if not hasattr(cls, '__perinstance'):
		cls = type(cls.__name__, (cls,), {})
		cls.__perinstance = True
		inst.__class__ = cls
	setattr(cls, name, property(method))


def to_range(input: Union[_TNum, List[_TNum]]) -> List[torch.Tensor]:
	'''Convert `input` to range of the form [bound_1,initial,bound_2]'''
	if isinstance(input, list):
		if len(input)<1 or len(input)>3:
			raise ValueError(f"length of input list should be 1, 2 or 3, got {str(input)}")
		if len(input)==1:
			output = 3*input
		elif len(input)==2:
			output = [input[0],input[0],input[1]]
		else:
			output = input
	else:
		output = 3*[input]
	return torch.tensor(output)


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

		assert not torch.isnan(error), "NaN value in the backprop error of the linear solver for ode %s"%(ctx.self.name)
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
			#ctx.self._stat['backward/dy_norm']      = ctx.self._stat.get('backward/dy_norm',0)      + dy.reshape((batch_dim,-1)).norm(dim=1).amax()

		return None, None, dx, None



###############################################################################
###############################################################################


class ode_solver(torch.nn.Module, metaclass=ABCMeta):
	def __init__(self, rhs: RHS, T:_TNum, num_steps: int, cache_path: bool = False,
		t_out: Optional[torch.Tensor] = None, ind_out: Optional[torch.Tensor] = None) -> None:
		'''
		Parameters
		----------
		  rhs:        vector field of the solver
		  T:          final time
		  num_steps:  number of steps on the time grid
		  cache_path: optional, flag to keep intermediate steps of the solver
		  t_out:      optional, time instances for the output, by default return only final value
		  ind_out:    optional, time grid indices for the output
		'''
		super().__init__()

		# define ode count in a network as a class property
		self.__class__.ode_count = getattr(self.__class__,'ode_count',-1) + 1
		self.name = f"{self.__class__.ode_count}.ode"

		# register rhs as a submodule
		self.rhs = rhs

		# register internal buffers describing time grid
		self.register_buffer('_T',         torch.tensor(T))
		self.register_buffer('_num_steps', torch.tensor(num_steps))
		self.register_buffer('_h',         torch.tensor(T/num_steps))

		# `_t` and `_y` are circular buffers
		self._t = deque([], maxlen=num_steps+1)
		self._y = deque([], maxlen=num_steps+1)

		# set parameters that control which values to output
		self.cache_path = cache_path
		self.t_out      = t_out
		self.ind_out    = ind_out

		# by default, there are no regularizers
		self.has_regularizers = False

		# is this adjoint solver? By default, it is not
		self.adjoint = False

		#
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
		if not self.adjoint:
			return self._h
		else:
			if hasattr(self,'_adj_h'):
				return -self._adj_h
			else:
				return -self._h

	@property
	def ind_out(self):
		if hasattr(self,'_ind_out'):
			return self._ind_out
		else:
			return None
	@ind_out.setter
	def ind_out(self, ind_out):
		if ind_out is not None:
			self.cache_path = True
			if not (torch.is_tensor(ind_out) and ind_out.ndim==1):
				raise TypeError(f"`ind_out` must a 1d tensor, got ind_out = {ind_out}")
			if ind_out.numel()>1 and torch.amin(torch.diff(ind_out))<=0:
				raise ValueError("`ind_out` must be increasing sequence")
			if ind_out[0]<0 or ind_out[-1]>self.num_steps:
				raise ValueError("`ind_out` must have values in [0,num_steps]")
			self._ind_out = ind_out

	@property
	def t_out(self):
		if hasattr(self,'_interp_coef'):
			t = self._h * (self.ind_out + self._interp_coef)
		elif hasattr(self,'_ind_out'):
			t = self._h * self.ind_out
		else:
			t = None
		return t
	@t_out.setter
	def t_out(self, t_out):
		if t_out is not None:
			self.cache_path = True
			if not (torch.is_tensor(t_out) and t_out.ndim==1):
				raise TypeError(f"`t_out` must be a 1d tensor, got t_out = {t_out}")
			if t_out.numel()>1 and torch.amin(torch.diff(t_out))<=0:
				raise ValueError("`t_out` must be increasing sequence")
			if t_out[0]<0 or t_out[-1]>self.T:
				raise ValueError("`t_out` must have values in [0,T]")
			# use `ind_out` to index the time grid interval that contains `t_out`
			self.ind_out = (t_out/self._h).long()
			# `interp_coef` are coefficients for linear interpolation of the time grid to arbitrary time instances
			# `interp_coef` are defined for each `t_out` relative to the time grid interval that contains it
			# For example, if `h=0.4` and `t_out=2.3`, then `ind_out=int(2.3/0.4)=5`, i.e, `t_out` is in the 6th interval
			# Hence `interp_coef=2.3/0.4-5=0.75` and `y[t=2.3] = 0.25*y[i=5] + 0.75*y[i=6]`
			self._interp_coef = t_out/self._h - self.ind_out

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

	@abstractmethod
	def solve_adjoint(self, yT):
		pass

	@abstractmethod
	def stability_function(self, z):
		pass

	@abstractmethod
	def inv_stability_function(self, z):
		pass

	@abstractmethod
	def restrict_stability(self, stability_center, lipschitz_constant):
		pass

	@abstractmethod
	def regularize(self, alpha):
		pass

	@abstractmethod
	def compute_regularizers(self):
		pass

	########################################

	def trapezoidal_quadrature(self, fun, weight_fun=None):
		'''Trapezoidal quadrature along the cached solver trajectory'''

		if not self.cache_path:
			raise RuntimeError("When using path integration, we need `cache_path=True`")

		# quadrature points
		qt, qy = self._t, self._y

		# quadrature weight
		qw = [1] * (self.num_steps+1) if weight_fun is None else [weight_fun(t,x) for t,x in zip(qt,qy)]
		qw[0], qw[-1] = 0.5*qw[0], 0.5*qw[-1]

		# function values along the trajectory
		qf = [fun(t,x) for t,x in zip(qt,qy)]

		return self.h * sum(wi*yi for wi,yi in zip(qw,qy))


	def rectangular_quadrature(self, fun, weight_fun=None, theta=0.0):
		'''Rectangular quadrature along the cached solver trajectory'''

		if not self.cache_path:
			raise RuntimeError("When using path integration, we need `cache_path=True`")

		# quadrature points
		# note that generator comprehensions ( ... for ... ) will not work when qt,qy need to be iterated multiple times, hence use lists
		# if theta==1, use left endpoint instead of right (same as step_fun in theta_solver)
		qt = [t+theta*self.h if theta<1 else t for t in self._t[:-1]]
		qy = [(1-theta)*self._y[i]+theta*self._y[i+1] for i in range(self.num_steps)]
		# qy = [solver._y[i+1] for i in range(solver.num_steps)]
		# qy = [solver._y[i] for i in range(solver.num_steps)]

		# quadrature weight
		qw = [1] * self.num_steps if weight_fun is None else [weight_fun(t,x) for t,x in zip(qt,qy)]

		# function values along the trajectory
		qf = [fun(t,x) for t,x in zip(qt,qy)]

		return self.h * sum(wi*yi for wi,yi in zip(qw,qy))

	def path_integral(self, fun, weight_fun=None, rule='trapezoidal', *args, **kwargs):
		if rule=='trapezoidal':
			return self.trapezoidal_quadrature(fun, weight_fun)
		elif rule=='rectangular':
			return self.rectangular_quadrature(fun, weight_fun, *args, **kwargs)
		else:
			raise NotImplementedError(f"Unknown quadrature rule {rule}")


	def trajectory(self, y0, t=None):
		'''Evaluate trajectory of the solver'''
		if t is None:
			# Evaluate solution at grid points
			ind_out = self._ind_out
			self.ind_out = torch.arange(self.num_steps+1)
			odesol = self.forward(y0, return_t=False)
			self._ind_out = ind_out
		else:
			# Evaluate solution at time instances in `t`
			ind_out     = self._ind_out
			interp_coef = self._interp_coef
			self.t_out  = t
			odesol = self.forward(y0, return_t=False)
			self._ind_out     = ind_out
			self._interp_coef = interp_coef
		return odesol


	def forward(self, y0, t0=0, return_t=False):
		# cache trajectory if needed
		if self.cache_path:
			# evaluate solution at the grid points
			self._t.append(t0+0*self.h) # to make it a tensor
			self._y.append(y0)
			for step in range(1,self.num_steps+1):
				self._y.append(self.ode_step(self._t[-1], self._y[-1]))
				self._t.append(t0+step*self.h)
		else:
			y = y0
			for step in range(self.num_steps):
				y = self.ode_step(t0+step*self.h, y)
			t = self.T

		# evaluate solution at the given points
		if hasattr(self,'_interp_coef'):
			# returns tensor of shape (batch size, time points, hidden dim)
			y = [ ((1-c)*self._y[i] if c<1 else 0) + (c*self._y[i+1] if c>0 else 0) for i,c in zip(self._ind_out,self._interp_coef) ]
			y = torch.stack(y, 1)
			if return_t:
				t = [ ((1-c)*self._t[i] if c<1 else 0) + (c*self._t[i+1] if c>0 else 0) for i,c in zip(self._ind_out,self._interp_coef) ]
				t = torch.stack(t) if t[0].ndim==0 else torch.stack(t, 1)
		elif hasattr(self,'_ind_out'):
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



###############################################################################
###############################################################################



class theta_solver(ode_solver):
	def __init__(self, rhs: RHS, T: _TNum, num_steps: int, theta: _TNum, cache_path: bool = False, tol: float=_TOL,
		t_out: Optional[torch.Tensor] = None, ind_out: Optional[torch.Tensor] = None) -> None:
		'''
		Parameters
		----------
		  rhs:        vector field of the solver
		  T:          final time
		  num_steps:  number of steps on the time grid
		  theta:      implicitness of the method
		  cache_path: optional, keep intermediate steps of the solver
		  tol:        optional, tolerance of the nonlinear solver, ignored for `theta=0`
		  t_out:      optional, time instances for the output, by default output only final value
		  ind_out:    optional, indices on the time grid for the output
		'''
		super().__init__(rhs=rhs, T=T, num_steps=num_steps, t_out=t_out, ind_out=ind_out, cache_path=cache_path)

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

	def stability_function(self, z):
		return (1+(1-self.theta)*z) / (1-self.theta*z)


	def inv_stability_function(self, z, min_eig=-20, max_eig=20):
		# horizontal asymptote of the stability function separating two branches; we need to remain on the upper branch
		branch_switch_asymptote = torch.tensor(1.0 - 1.0/(self.theta+1.e-12) + 1.e-6, dtype=torch.float)
		# restrict to correct branch
		if z<=0: z = torch.maximum(z, branch_switch_asymptote)
		# restrict range of spectrum
		eigenvalue = torch.clamp((1-z)/((1-z)*self.theta-1), min=min_eig, max=max_eig)
		return eigenvalue

	def restrict_stability(self, stability_center: Union[_TNum,List[_TNum]], lipschitz_constant: Union[_TNum,List[_TNum]]) -> None:
		'''Restrict stability of the solver

		Parameters
		----------
		  stability_center:   stability center   of the restricted rhs
		  lipschitz_constant: lipschitz constant of the restricted rhs
		'''
		if self.T!=self.num_steps:
			raise ValueError(f"if we explicitly control stability of the solver, we need `h=1` and hence `T=num_steps`, got T={T} and num_steps={num_steps}")

		stab_range = to_range(stability_center)
		lips_range = to_range(lipschitz_constant)

		sigscale = 2.0

		# note that `restrict_theta_stability` modifies `rhs` modules in-place, it doesn't need to be registered again
		for rhs_i in self.rhs:
			# restrict_theta_stability(rhs_i, self.theta, stability_center, lipschitz_constant)

			#######################################################################
			# initialize params so that sigmoid(param) = ini_sigmoid
			ini_lip = (lips_range[1]-lips_range[0]) / (lips_range[2]-lips_range[0]+1.e-8) + 0.0010
			ini_cnt = (stab_range[1]-stab_range[0]) / (stab_range[2]-stab_range[0]+1.e-8) + 0.0011
			inv_sigmoid = lambda ini_sigmoid: torch.log(ini_sigmoid/(1-ini_sigmoid)) / sigscale
			# new parameters for center and radius (note that not all parameters might be used)
			rhs_i.register_parameter('cnt_var', torch.nn.parameter.Parameter(inv_sigmoid(ini_cnt), requires_grad=(stab_range[-1]!=stab_range[0]).item()))
			rhs_i.register_parameter('lip_var', torch.nn.parameter.Parameter(inv_sigmoid(ini_lip), requires_grad=(lips_range[-1]!=lips_range[0]).item()))

			# add utility functions
			def freeze_spectral_circle(obj):
				obj.cnt_var.requires_grad_(False)
				obj.lip_var.requires_grad_(False)
			def unfreeze_spectral_circle(obj):
				obj.cnt_var.requires_grad_(True)
				obj.lip_var.requires_grad_(True)
			def get_spectral_circle(obj):
				return obj.center, obj.radius
			rhs_i.freeze_spectral_circle   = MethodType(freeze_spectral_circle,   rhs_i)
			rhs_i.unfreeze_spectral_circle = MethodType(unfreeze_spectral_circle, rhs_i)
			rhs_i.get_spectral_circle      = MethodType(get_spectral_circle,      rhs_i)

			#######################################################################
			# dynamically add properties to the object of the "per-instance class"
			def stability_center(obj):
				return stab_range[0] + torch.sigmoid(sigscale*obj.cnt_var) * ( stab_range[-1] - stab_range[0] )

			def lipschitz_constant(obj):
				return lips_range[0] + torch.sigmoid(sigscale*obj.lip_var) * ( lips_range[-1] - lips_range[0] )

			def stability_radius(obj):
				return torch.nn.functional.relu(obj.lipschitz_constant-obj.stability_center)
				# return F.relu(self.lipschitz_constant-self.stability_center.detach())

			def center(obj):
				return self.inv_stability_function(obj.stability_center)

			def radius(obj):
				eig_lipschitz = self.inv_stability_function(obj.lipschitz_constant)
				eig_center    = self.inv_stability_function(obj.stability_center)
				eig_radius    = torch.nn.functional.relu(eig_lipschitz-eig_center)
				#eig_radius    = F.relu(eig_lipschitz-eig_center.detach())
				#eig_radius    = F.softplus(eig_lipschitz-eig_center,beta=20)
				return eig_radius

			addprop(rhs_i, 'center', center)
			addprop(rhs_i, 'radius', radius)
			addprop(rhs_i, 'stability_center', stability_center)
			addprop(rhs_i, 'stability_radius', stability_radius)
			addprop(rhs_i, 'lipschitz_constant', lipschitz_constant)

			#######################################################################
			# perform spectral normalization for linear layers

			for f in rhs_i.modules():
				if isinstance(f, torch.nn.modules.conv._ConvNd) or isinstance(f, torch.nn.Linear) or isinstance(f, torch.nn.modules.batchnorm._BatchNorm):
					spectral_norm(f, name='weight')
					# # perform dummy initial spectral normalization
					# x = torch.ones(1,*input_shape)
					# for _ in range(5):
					# 	rhs_i(x)

			#######################################################################
			# register forward hooks

			def stability_restriction_hook(m, input, output):
				eig_lipsch = self.inv_stability_function(m.lipschitz_constant)
				eig_center = self.inv_stability_function(m.stability_center)
				eig_radius = torch.nn.functional.relu(eig_lipsch-eig_center)
				# eig_radius = F.relu(eig_lipsch-eig_center.detach())
				# eig_radius = F.softplus(eig_lipsch-eig_center,beta=20)
				# print(m.cnt_var)
				return eig_center * input[0] + eig_radius * output

			rhs_i.register_forward_hook(stability_restriction_hook)

		return self


	def regularize(self, alpha, quadrature_rule='trapezoidal', collect_rhs_stat=_collect_rhs_stat):
		'''Apply regularization to the solver'''

		self.has_regularizers = True
		self.cache_path = True

		# copy to avoid unexpected behavior if alpha is changed after passing to this method
		alpha = alpha.copy()

		# default regularizers
		if 'tr0'  not in alpha: alpha['tr0']  = 0.0
		if 'jac'  not in alpha: alpha['jac']  = 0.0
		if 'f'    not in alpha: alpha['f']    = 0.0
		##########
		if 'df'         not in alpha: alpha['df']         = 0.0
		if 'df^eps'     not in alpha: alpha['df^eps']     = 1.e-2
		if 'df^theta'   not in alpha: alpha['df^theta']   = None
		if 'df^h'       not in alpha: alpha['df^h']       = None
		if 'df^steps'   not in alpha: alpha['df^steps']   = 0
		if 'df^samples' not in alpha: alpha['df^samples'] = 1
		if 'df^ini'     not in alpha: alpha['df^ini']     = 0
		if 'df^max'     not in alpha: alpha['df^max']     = 10
		##########
		if 'dfdn' not in alpha: alpha['dfdn'] = 0.0
		if 'dfdn^eps'     not in alpha: alpha['dfdn^eps']     = 1.e-3
		if 'dfdn^samples' not in alpha: alpha['dfdn^samples'] = 1
		##########
		if 'div'  not in alpha: alpha['div']  = 0.0
		if 'rad'  not in alpha: alpha['rad']  = 0.0
		if 'cnt'  not in alpha: alpha['cnt']  = 0.0
		if 'lip'  not in alpha: alpha['lip']  = 0.0
		if 'tv'   not in alpha: alpha['tv']   = 0.0
		if 'stabcnt' not in alpha: alpha['stabcnt'] = 0.0
		if 'stabrad' not in alpha: alpha['stabrad'] = 0.0
		##########
		if 'stabdiv' not in alpha: alpha['stabdiv'] = 0.0
		if 'stabdiv^samples' not in alpha: alpha['stabdiv^samples'] = 0
		if 'stabdiv^eps' not in alpha: alpha['stabdiv^eps'] = 1.e-3

		self.r_integrate = lambda f: self.path_integral(f, rule=quadrature_rule, theta=self.theta) / self.T
		self.r_alpha = alpha
		self.r_collect_rhs_stat = collect_rhs_stat

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
		fevals = 1

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
				y, resid, iters, fevals, flag = nsolve( residual_fn, x, _nsolver, tol=self.tol, max_iters=_max_iters )
				assert not torch.isnan(resid), "NaN value in the residual of the nonlinear solver for ode %s"%(self.name)

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
			self._stat[mode+'fevals']   = self._stat.get(mode+'fevals',0)   + fevals
			self._stat[mode+'residual'] = self._stat.get(mode+'residual',0) + resid
		return y



###############################################################################
###############################################################################


#######################################
# # stability function and its inverse
# def theta_stability_fun(theta, x):
# 	return (1+(1-theta)*x) / (1-theta*x)

# def theta_inv_stability_fun(theta, y):
# 	y = max(y, 1 - 1.0/(theta+1.e-12) + 1.e-6)
# 	return (1-y) / ((1-y)*theta-1)
#######################################


#######################################
# quadrature rules
def trapz(y, dx=1.0):
	res = 0.5 * (y[0] + y[-1]) + sum(y[1:-1])
	return dx * res

# def trapz(y, dx=1.0):
# 	res = 0.5 * y[0]
# 	for i in range(1,len(y)-1):
# 		res = res + y[i]
# 	res = res + 0.5 * y[-1]
# 	return res * dx

def rect(y, dx=1.0):
	return dx * sum(y)
#######################################


# forward hook to evaluate regularizers and statistics
def old_compute_regularizers_and_statistics(solver, input, output):
	reg   = {}
	stat  = {}
	if solver.training:
		rhs        = solver.rhs
		name       = solver.name+'_'
		alpha      = solver.alpha
		theta      = solver.theta
		steps      = solver.num_steps
		quadrature = solver.quadrature
		stabval    = solver.stabval
		batch_dim  = input[0].size(0)
		dim        = input[0].numel() / batch_dim
		dx         = 1 / steps / dim

		# temporal quadrature rule
		if quadrature=='trapezoidal':
			integrate = trapz
			# contribution of divergence along the trajectory
			c = ( ((t+1)/(steps+1))**solver.p for t in range(steps+1) )
		elif quadrature=='rectangle':
			integrate = rect
			c = ( ((t+1)/(steps))**solver.p for t in range(steps) )
		def quadrature_points():
			if quadrature=='trapezoidal':
				q_t = solver._t
				q_y = solver._y
			elif quadrature=='rectangle':
				# if theta==1, use left endpoint
				q_t = ( solver._t[i-1]+theta*solver.h if theta<1 else solver._t[i-1] for i in range(1,solver.num_steps+1) )
				q_y = ( (1-theta)*solver._y[i-1]+theta*solver._y[i] for i in range(1,solver.num_steps+1) )
			return q_t, q_y


		# spectral normalization has to be performed only once per forward pass, so freeze here
		rhs.eval()

		# evaluate divergence and jacobian along the trajectory
		if alpha['div']!=0 or alpha['jac']!=0 or solver.collect_rhs_stat:
			# if not hasattr(solver, 'trace_weight'):
			# 	solver.trace_weight = torch.ones_like(input[0])
			div = []
			for t, y in zip(*quadrature_points()):
				jacdiag = calc.jacobian_diag( lambda x: rhs(t, x), y, n=solver.mciters )
				if stabval is None:
					weight = 1
				else:
					weight = jacdiag>theta_inv_stability_fun(theta,stabval) #torch.heaviside( jacdiag - theta_inv_stability_fun(theta,stabval), torch.tensor([0.0]) ).detach()
					dx = 1.0 / steps / (weight.sum()/batch_dim)
				div.append( (weight*jacdiag).sum()/batch_dim )
				if stabval is not None and weight.sum()<0.8*input[0].numel():
					solver.reduce = True
			# jacdiag = [ calc.jacobian_diag( lambda x: rhs(t, x), y, n=solver.mciters) for t, y in zip(*quadrature_points()) ]
			# divjac = [ calc.trace_and_jacobian( lambda x: rhs(t, x), y, n=solver.mciters, min_eig=theta_inv_stability_fun(solver.theta,0.0)) for t, y in zip(solver._t, solver._y) ]
			# divjac = [ calc.partial_trace( lambda x: rhs(t, x), y, n=solver.mciters, fraction=0.8) for t, y in zip(solver._t, solver._y) ]
			# buf = 0
			# for dj in divjac:
			# 	# buf += dj[2]
			# 	if dj[2]<0.8*input[0].numel(): solver.reduce = True
			# w = buf / input[0].numel() / len(solver._t)

		if alpha['f']!=0 or solver.collect_rhs_stat:
			vfield = [ rhs(t,y).pow(2).sum() for t, y in zip(*quadrature_points()) ]

		# divergence
		if alpha['div']!=0:
			reg[name+'div'] = alpha['div'] * torch.sigmoid( integrate([ct*divt for ct, divt in zip(c, div)], dx) )



		if solver.collect_rhs_stat:
			stat['rhs/'+name+'div'] = integrate([divt for divt in div], dx)
			# stat['rhs/'+name+'jac'] = integrate([jact[1] for jact in divjac], dx=dx/dim)
			# stat['rhs/'+name+'f']   = integrate(vfield, dx)

		# # divergence
		# if alpha['div']!=0:
		# 	reg[name+'div'] = alpha['div'] * torch.sigmoid( integrate([ct*divt[0] for ct, divt in zip(c, divjac)], dx) )

		# # jacobian
		# if alpha['jac']!=0:
		# 	reg[name+'jac'] = alpha['jac'] * (stat['rhs/'+name+'jac'] if solver.collect_rhs_stat else integrate([jact[1] for jact in divjac], dx=dx/dim))

		# # magnitude
		# if alpha['f']!=0:
		# 	reg[name+'f'] = alpha['f'] * (stat['rhs/'+name+'f'] if solver.collect_rhs_stat else integrate([rhs(t,y).pow(2).sum() for t, y in zip(solver._t, solver._y)], dx))

		# # residual
		# if alpha['resid']!=0:
		# 	for step in range(steps):
		# 		x, y = solver._y[step].detach(), solver._y[step+1].detach()
		# 		reg[name+'residual'] = reg.get(name+'residual',0) + solver.residual(solver._t[step], x, y)
		# 	reg[name+'residual'] = (alpha['resid'] / steps) * reg[name+'residual']

		# # data augmentation
		# if alpha['daugm']!=0:
		# 	reg[name+'daugm'] = alpha['daugm'] * solver.augmentation_loss(solver._y[-1])

		# # hidden layer perturbation
		# if alpha['perturb']!=0:
		# 	x_perturb = solver_y[0] + solver.perturbation(solver_y[0])
		# 	y_perturb = solver.forward(x_perturb.detach())
		# 	reg[name+'perturb'] = alpha['perturb'] * solver.perturbation_loss(y_perturb, solver._y[-1])

		# # 'Total variation'
		# if alpha['TV']!=0 and len(rhs.F)>1:
		# 	w1 = torch.nn.utils.parameters_to_vector(rhs.F[0].parameters())
		# 	for t in range(len(rhs.F)-1):
		# 		w2 = torch.nn.utils.parameters_to_vector(rhs.F[t+1].parameters())
		# 		reg[name+'TV'] = reg.get(name+'TV',0) + ( w2 - w1 ).pow(2).sum()
		# 		w1 = w2
		# 	reg[name+'TV'] = (alpha['TV'] / steps) * reg[name+'TV']

		# note that solver.training has not been changed by rhs.eval()
		rhs.train(mode=solver.training)

		if not hasattr(solver, 'regularizer'):
			setattr(solver, 'regularizer', reg)
		else:
			for key, val in reg.items():
				solver.regularizer[key] = val
		if solver.collect_rhs_stat:
			# stat['rhs/'+name+'jac'] = stat['rhs/'+name+'jac'].sqrt()
			# stat['rhs/'+name+'f']   = stat['rhs/'+name+'f'].sqrt()
			setattr(rhs, 'statistics', stat)
	return None


def old_regularized_ode_solver(solver, alpha={}, stability_limits=None, mciters=1, p=2, quadrature='rectangle', stability_target=None, collect_rhs_stat=_collect_rhs_stat, augmentation_loss=None, perturbation=None, perturbation_loss=None):
	if 'div'   not in alpha: alpha['div']   = 0.0
	if 'jac'   not in alpha: alpha['jac']   = 0.0
	if 'f'     not in alpha: alpha['f']     = 0.0
	if 'resid' not in alpha: alpha['resid'] = 0.0
	if 'TV'    not in alpha: alpha['TV']    = 0.0
	# if 'daugm' not in alpha:
	# 	alpha['daugm'] = 0.0
	# else:
	# 	if augmentation_loss is None: augmentation_loss = lambda x: (x[:x.size(0)//2,...]-x[x.size(0)//2:,...]).pow(2).sum()
	# 	setattr(solver, 'augmentation_loss', augmentation_loss)
	# if 'perturb' not in alpha:
	# 	alpha['perturb'] = 0.0
	# else:
	# 	if alpha['perturb']!=0: assert perturbation is not None, "if alpha['perturb'] != 0, perturbation must be given"
	# 	if perturbation_loss is None: perturbation_loss = lambda x,y: (x-y).pow(2).sum()
	# 	setattr(solver, 'perturbation',      perturbation)
	# 	setattr(solver, 'perturbation_loss', perturbation_loss)
	setattr(solver, 'alpha', alpha)
	setattr(solver, 'mciters', mciters)
	setattr(solver, 'p', p)
	setattr(solver, 'quadrature', quadrature)
	setattr(solver, 'stabval', stability_target)
	setattr(solver, 'collect_rhs_stat', collect_rhs_stat)
	if stability_limits is not None:
		if stability_target is not None:
			minstab_lim = theta_inv_stability_fun(solver.theta, stability_target)
			stab_lim    = theta_inv_stability_fun(solver.theta, stability_limits[0])
			min_eig = max(-20, minstab_lim - min(2.0,minstab_lim-stab_lim) )
		else:
			min_eig = max(-20, theta_inv_stability_fun(solver.theta, stability_limits[0]) )
		max_eig = min( 20, theta_inv_stability_fun(solver.theta, stability_limits[2]) )
		ini_eig = theta_inv_stability_fun(solver.theta, stability_limits[1])
		solver.rhs.set_spectral_limits([min_eig,ini_eig,max_eig])
	if stability_target is not None:
		center_final = theta_inv_stability_fun(solver.theta, stability_target)
		solver.rhs.set_circle_limits( center_limits=[0.0, center_final], radius_limits=[1.0, 0.1] )
	solver.register_forward_hook(compute_regularizers_and_statistics)
	return solver



# forward hook to evaluate regularizers and statistics
def compute_regularizers_and_statistics(solver, input, output):
	reg   = {}
	stat  = {}
	if solver.training:
		# rhs        = solver.rhs
		name       = solver.name+'_'
		# alpha      = solver.r_alpha
		# theta      = solver.theta
		# steps      = solver.num_steps
		# stabval    = solver.stabval
		batch_dim  = input[0].size(0)
		dim        = input[0].numel() / batch_dim

		# spectral normalization has to be performed only once per forward pass, so freeze here
		solver.rhs.eval()


		#######################################################################
		# evaluate quantities for regularizers
		# quadrature points to evaluate temporal integrals
		qt, qy = solver.r_quadrature_points()

		if solver.r_alpha['div']!=0 or solver.r_collect_rhs_stat:
			# int_div = solver.r_integrate( lambda t,y: calc.jacobian_diag(lambda x: solver.rhs(t, x), y).sum() / batch_dim ) / dim
			int_div = solver.r_integrate([ calc.jacobian_diag(lambda x: solver.rhs(t, x), y).sum()/batch_dim for t,y in zip(qt,qy) ]) / dim

		if solver.r_alpha['cnt']!=0 or solver.r_alpha['rad']!=0 or solver.r_collect_rhs_stat:
			center, radius   = solver.rhs.center,  solver.rhs.radius
			stabmin, stabmax = solver.rhs.stabmin, solver.rhs.stabmax
			regcenter = center if stabmin is None else stabmin
			regradius = radius if stabmin is None or stabmax is None else torch.maximum(torch.tensor(0.1).to(int_div.device),stabmax-stabmin.detach())
			# stabcenter, radius = solver.rhs.get_spectral_circle(stability_circle=True)

		#######################################################################
		# regularizers
		# divergence regularizer
		if solver.r_alpha['div']!=0:
			reg[name+'div'] = solver.r_alpha['div'] * torch.sigmoid(int_div)

		# spectral center regularizer
		if solver.r_alpha['cnt']!=0:
			reg[name+'cnt'] = solver.r_alpha['cnt'] * regcenter

		# spectral radius regularizer
		if solver.r_alpha['rad']!=0:
			reg[name+'rad'] = solver.r_alpha['rad'] * regradius

		#######################################################################
		# statistics
		if solver.r_collect_rhs_stat:
			# stabcenter, radius = solver.rhs.get_stability_circle()
			center, radius = solver.rhs.get_spectral_circle()
			stat['rhs/'+name+'div'] = int_div
			stat['rhs/'+name+'cnt'] = center
			stat['rhs/'+name+'rad'] = radius
			if stabmin is not None: stat['rhs/'+name+'stabmin'] = stabmin
			if stabmax is not None: stat['rhs/'+name+'stabmax'] = stabmax



		# note that solver.training has not been changed by rhs.eval()
		solver.rhs.train(mode=solver.training)

		if not hasattr(solver, 'regularizer'):
			setattr(solver, 'regularizer', reg)
		else:
			for key, val in reg.items():
				solver.regularizer[key] = val
		if solver.r_collect_rhs_stat:
			setattr(solver.rhs, 'statistics', stat)
	return None


def regularized_ode_solver(solver, alpha={}, p=0, quadrature='rectangle', collect_rhs_stat=_collect_rhs_stat):
	alpha = alpha.copy()
	# disable all regularizers by default
	if 'div' not in alpha: alpha['div'] = 0.0
	if 'jac' not in alpha: alpha['jac'] = 0.0
	if 'f'   not in alpha: alpha['f']   = 0.0
	if 'rad' not in alpha: alpha['rad'] = 0.0
	if 'cnt' not in alpha: alpha['cnt'] = 0.0
	if 'tv'  not in alpha: alpha['tv']  = 0.0

	# temporal quadrature rule,
	# contribution of divergence along the trajectory, and
	# (t,y) quadrature_points
	if quadrature=='trapezoidal':
		qc = [((t+1)/(solver.num_steps+1))**p for t in range(solver.num_steps+1)]
		# integrate = lambda fun: solver.h * (
		# 	0.5 * ( qc[0] * fun(solver._t[0],solver._y[0]) + qc[-1] * fun(solver._t[-1],solver._y[-1]) )
		# 	+ sum(c*fun(t,y) for c,t,y in zip(qc[1:-1],solver._t[1:-1],solver._y[1:-1]))
		# 	)
		integrate = lambda y: (0.5 * (qc[0]*y[0] + qc[-1]*y[-1]) + sum(ci*yi for ci,yi in zip(qc[1:-1],y[1:-1]))) / solver.num_steps
		def quadrature_points():
			return solver._t, solver._y
	elif quadrature=='rectangle':
		qc = [((t+1)/(solver.num_steps))**p for t in range(solver.num_steps)]
		# # if theta==1, use left endpoint instead of right (same as step_fun in theta_solver)
		# qt = (solver._t[i]+solver.theta*solver.h if solver.theta<1 else solver._t[i] for i in range(solver.num_steps))
		# integrate = lambda fun: solver.h *
		# 	sum( c*fun(t,y) for c,t,y in zip(
		# 		qc,
		# 		(solver._t[i]+solver.theta*solver.h if solver.theta<1 else solver._t[i] for i in range(solver.num_steps)),
		# 		((1-solver.theta)*solver._y[i]+solver.theta*solver._y[i+1] for i in range(solver.num_steps))
		# 		)
		# 	)
		# print(solver.h, 1/solver.num_steps)
		# exit()
		integrate = lambda y: sum(ci*yi for ci,yi in zip(qc,y)) / solver.num_steps
		def quadrature_points():
			# note that generator comprehensions ( ... for ... ) will not work
			# when qt, qy need to be iterated multiple times, hence use lists
			# if theta==1, use left endpoint instead of right (same as step_fun in theta_solver)
			qt = [solver._t[i]+solver.theta*solver.h if solver.theta<1 else solver._t[i] for i in range(solver.num_steps)]
			qy = [(1-solver.theta)*solver._y[i]+solver.theta*solver._y[i+1] for i in range(solver.num_steps)]
			return qt, qy
	else:
		exit('unknown quadrature rule')

	setattr(solver, 'r_integrate', integrate)
	setattr(solver, 'r_quadrature_points', quadrature_points)
	setattr(solver, 'r_alpha', alpha)
	# setattr(solver, 'r_mciters', mciters)
	# setattr(solver, 'p', p)
	# setattr(solver, 'quadrature', quadrature)
	setattr(solver, 'r_collect_rhs_stat', collect_rhs_stat)

	solver.register_forward_hook(compute_regularizers_and_statistics)
	return solver
