from abc import ABCMeta, abstractmethod
from typing import Union, List
from types import MethodType
from copy import deepcopy

import math

import torch
import torch.nn.functional as F
from .misc import ParabolicPerceptron, HamiltonianPerceptron, HollowMLP, MLP, PreActConv2d
from ..utils.spectral import spectral_norm



_TNum = Union[int, float]


###############################################################################
###############################################################################


# class rhs_base_old(torch.nn.Module, metaclass=ABCMeta):
# 	def __init__(self, shape, T, num_steps,
# 		learn_spectral_limits=False, spectral_limits=None,
# 		learn_spectral_circle=False, center_limits=None, radius_limits=None):
# 		super().__init__()
# 		self.shape = shape
# 		self.h     = T / num_steps

# 		# self.learn_scales = learn_scales
# 		assert (learn_spectral_limits and learn_spectral_circle)==False, 'either spectral_limits or spectral_circle can be learned, not both'
# 		self.learn_limits = learn_spectral_limits
# 		self.learn_circle = learn_spectral_circle

# 		#######################################################################
# 		# Parameters (note that not all parameters might be used)
# 		# initialize params so that sigmoid(param) = ini_sigmoid
# 		ini_sigmoid = 0.99
# 		inv_sigmoid = math.log(ini_sigmoid/(1-ini_sigmoid))
# 		#
# 		self.eigmin_var = torch.nn.parameter.Parameter( torch.tensor(inv_sigmoid), requires_grad=learn_spectral_limits)
# 		self.eigmax_var = torch.nn.parameter.Parameter( torch.tensor(inv_sigmoid), requires_grad=learn_spectral_limits)
# 		#
# 		self.center_var = torch.nn.parameter.Parameter( torch.tensor(inv_sigmoid), requires_grad=learn_spectral_circle)
# 		self.radius_var = torch.nn.parameter.Parameter( torch.tensor(inv_sigmoid), requires_grad=learn_spectral_circle)

# 		#######################################################################
# 		# Buffers (initialize with default values, also note that not all might be used)
# 		#
# 		self.register_buffer('eigmin',  torch.tensor(-1.0))
# 		self.register_buffer('eiginit', torch.tensor(0.0))
# 		self.register_buffer('eigmax',  torch.tensor(1.0))
# 		#
# 		self.register_buffer('center_init',  torch.tensor(0.0))
# 		self.register_buffer('center_final', torch.tensor(0.0))
# 		self.register_buffer('radius_init',  torch.tensor(1.0))
# 		self.register_buffer('radius_final', torch.tensor(1.0))

# 		# # initialize scales so that sigmoid(scales) = ini_sigmoid
# 		# if learn_scales:
# 		# 	ini_sigmoid = 0.5 if learn_scales else 0.99
# 		# 	inv_sigmoid = math.log(ini_sigmoid/(1-ini_sigmoid))
# 		# 	self.scales = torch.nn.parameter.Parameter( inv_sigmoid * torch.ones(1,*shape), requires_grad=learn_scales)
# 		# else:
# 		# 	ini_sigmoid = 0.99
# 		# 	inv_sigmoid = math.log(ini_sigmoid/(1-ini_sigmoid))
# 		# 	self.scales = torch.tensor(inv_sigmoid)
# 		# 	# self.scales = torch.nn.parameter.Parameter( inv_sigmoid * torch.ones(1,*shape), requires_grad=learn_scales)

# 		if learn_spectral_limits: self.set_spectral_limits(spectral_limits)
# 		if learn_spectral_circle: self.set_circle_limits(center_limits, radius_limits)

# 	def set_spectral_limits(self, spectral_limits):
# 		assert isinstance(spectral_limits,list) or (spectral_limits is None), "spectral_limits should be a list or None"
# 		# if spectral_limits is None:
# 		# 	# self.eigmin, self.eiginit, self.eigmax = ( -1.0, 0.0, 1.0 )
# 		# 	self.register_buffer('eigmin',  torch.tensor(-1.0))
# 		# 	self.register_buffer('eiginit', torch.tensor(0.0))
# 		# 	self.register_buffer('eigmax',  torch.tensor(1.0))
# 		if spectral_limits is not None:
# 			if len(spectral_limits)==2:
# 				# self.eigmin, self.eigmax = spectral_limits
# 				# self.eiginit = 0.5*(self.eigmin+self.eigmax)
# 				self.register_buffer('eigmin',  torch.tensor(spectral_limits[0]))
# 				self.register_buffer('eigmax',  torch.tensor(spectral_limits[1]))
# 				self.register_buffer('eiginit', torch.tensor(0.5*(spectral_limits[0]+spectral_limits[1])))
# 				assert self.eigmin<self.eigmax, "eigmin < eigmax must be given, got spectral_limits = "+str(spectral_limits)
# 			elif len(spectral_limits)==3:
# 				# self.eigmin, self.eiginit, self.eigmax = spectral_limits
# 				self.register_buffer('eigmin',  torch.tensor(spectral_limits[0]))
# 				self.register_buffer('eiginit', torch.tensor(spectral_limits[1]))
# 				self.register_buffer('eigmax',  torch.tensor(spectral_limits[2]))
# 				assert self.eigmin<self.eiginit and self.eiginit<self.eigmax, "eigmin < eiginit < eigmax must be given, got spectral_limits = "+str(spectral_limits)

# 		ini_sigmoid_a = 0.01
# 		ini_sigmoid_b = ini_sigmoid_a * (self.eigmax-self.eiginit)/(self.eiginit-self.eigmin) # balance initial shifta and shiftb
# 		# ini_sigmoid_b = 0.99
# 		# ini_sigmoid_a = ini_sigmoid_b * (self.eiginit-self.eigmin)/(self.eigmax-self.eiginit) # balance initial shifta and shiftb

# 		a = math.log(ini_sigmoid_a/(1-ini_sigmoid_a))
# 		b = math.log(ini_sigmoid_b/(1-ini_sigmoid_b))

# 		torch.nn.init.constant_(self.eigmin_var, a)
# 		torch.nn.init.constant_(self.eigmax_var, b)

# 	def set_circle_limits(self, center_limits, radius_limits):
# 		assert isinstance(center_limits,list) or (center_limits is None), "center_limits should be a list or None"
# 		assert isinstance(radius_limits,list) or (radius_limits is None), "radius_limits should be a list or None"

# 		if center_limits is not None:
# 			self.register_buffer('center_init',  torch.tensor(center_limits[0]))
# 			self.register_buffer('center_final', torch.tensor(center_limits[1]))

# 		if radius_limits is not None:
# 			self.register_buffer('radius_init',  torch.tensor(radius_limits[0]))
# 			self.register_buffer('radius_final', torch.tensor(radius_limits[1]))

# 		ini_sigmoid_c = 0.01
# 		ini_sigmoid_r = 0.01

# 		c = math.log(ini_sigmoid_c/(1-ini_sigmoid_c))
# 		r = math.log(ini_sigmoid_r/(1-ini_sigmoid_r))

# 		torch.nn.init.constant_(self.center_var, c)
# 		torch.nn.init.constant_(self.radius_var, r)

# 	def get_spectral_limits(self):
# 		if self.learn_limits:
# 			a = self.eiginit + torch.sigmoid(self.eigmin_var) * ( self.eigmin - self.eiginit )
# 			b = self.eiginit + torch.sigmoid(self.eigmax_var) * ( self.eigmax - self.eiginit )
# 		elif self.learn_circle:
# 			c = self.center_init + torch.sigmoid(self.center_var) * ( self.center_final - self.center_init )
# 			r = self.radius_init + torch.sigmoid(self.radius_var) * ( self.radius_final - self.radius_init )
# 			a = c-r
# 			b = c+r
# 		return a, b

# 	def get_spectral_circle(self):
# 		if self.learn_limits:
# 			a = self.eiginit + torch.sigmoid(self.eigmin_var) * ( self.eigmin - self.eiginit )
# 			b = self.eiginit + torch.sigmoid(self.eigmax_var) * ( self.eigmax - self.eiginit )
# 			c = (a+b)/2
# 			r = (b-a)/2
# 		elif self.learn_circle:
# 			c = self.center_init + torch.sigmoid(self.center_var) * ( self.center_final - self.center_init )
# 			# r = self.radius_init + torch.sigmoid(self.radius_var) * ( self.radius_final - self.radius_init )
# 			r = torch.abs(c)
# 		else:
# 			c = 0.0
# 			r = 1.0
# 		return c, r

# 	def freeze_spectral_circle(self):
# 		if self.learn_limits:
# 			self.eigmin_var.requires_grad_(False)
# 			self.eigmax_var.requires_grad_(False)
# 		elif self.learn_circle:
# 			self.center_var.requires_grad_(False)
# 			self.radius_var.requires_grad_(False)

# 	def unfreeze_spectral_circle(self):
# 		if self.learn_limits:
# 			self.eigmin_var.requires_grad_(True)
# 			self.eigmax_var.requires_grad_(True)
# 		elif self.learn_circle:
# 			self.center_var.requires_grad_(True)
# 			self.radius_var.requires_grad_(True)

# 	def initialize(self):
# 		for name, weight in self.F.named_parameters():
# 			if 'weight' in name:
# 				# torch.nn.init.xavier_normal_(weight, gain=1.e-1)
# 				# torch.nn.init.xavier_uniform_(weight)
# 				# torch.nn.init.xavier_uniform_(weight, gain=1./weight.detach().norm())
# 				torch.nn.init.uniform_(weight,-1.e-3,1.e-3)
# 				# torch.nn.init.uniform_(weight,-1.,1.)
# 			else:
# 				torch.nn.init.zeros_(weight)
# 		# perform dummy initial spectral normalization if any
# 		x = torch.ones(1,*self.shape)
# 		for _ in range(5):
# 			for m in self.F:
# 				m(x)

# 	def t2ind(self, t):
# 		if torch.is_tensor(t):
# 			assert t.ndim<2, "t must be either a scalar or a vector"
# 			return torch.clamp( (t/self.h).int(), max=len(self.F)-1 )
# 		else:
# 			return min(int(t/self.h), len(self.F)-1)

# 	def forward(self, t, y):
# 		ind = self.t2ind(t)
# 		if torch.is_tensor(ind) and ind.ndim>0:
# 			assert ind.size(0)==y.size(0), "if t is tensor, it must have the same batch dimension as y"
# 			# need to sacrifice full batch parallelization here
# 			f = [ self.F[i](y[batch,...]) for batch, i in enumerate(ind) ]
# 			f = torch.stack(f)
# 			# this doesn't work. why?
# 			# f = [ self.F[i](y[i==ind,...]) for i in torch.unique(ind) ]
# 			# f = torch.cat(f,0)
# 		else:
# 			f = self.F[ind](y)

# 		# f = torch.sigmoid(self.scales) * f
# 		c, r = self.get_spectral_circle()

# 		return r * f + c * y
# 		# a = self.eiginit + torch.sigmoid(self.shifta) * ( self.eigmin - self.eiginit )
# 		# b = self.eiginit + torch.sigmoid(self.shiftb) * ( self.eigmax - self.eiginit )

# 		# return 0.5 * ((b-a)*f + (a+b)*y)


# class rhs_base(torch.nn.Module, metaclass=ABCMeta):
# 	def __init__(self, T, num_steps, mode='spectrum', center=None, radius=None, stabmin=None, stabmax=None, theta=None):
# 		"""
# 		Modes
# 		-----
# 		'spectrum' : center and radius of the circle with all eigenvalues
# 			either numbers or arrays of length 2 specifying the range of values
# 		'stability' : minimum and maximum of the stability function
# 			either numbers or arrays of length 2 specifying the range of values
# 		"""
# 		super().__init__()
# 		self.mode = mode
# 		self.h    = T / num_steps
# 		self.mode = mode

# 		# convert to eigenvalues from stability values if given
# 		if mode in ['stability', 'stabcenter']:
# 			assert theta is not None and theta>=0 and theta<=1, "if mode is 'stability', theta must be in [0,1], got theta="+str(theta)
# 			self.theta = theta
# 			# self.register_buffer('theta', torch.tensor(theta, dtype=torch.float))
# 			# center = self.inverse_stability_fun(theta, center)
# 			# spread = self.inverse_stability_fun(theta, spread)

# 		assert mode in ['spectrum', 'stability', 'stabcenter'], 'unknown mode %s'%(mode)
# 		if mode in ['spectrum', 'stabcenter']:
# 			self.center = center
# 			self.radius = radius
# 			# self.set_center(center)
# 			# self.set_radius(radius)
# 		elif mode=='stability':
# 			self.stabmin = stabmin
# 			self.stabmax = stabmax
# 			# self.set_stabmin(stabmin)
# 			# self.set_stabmax(stabmax)

# 		self.sigmscale = 2.0

# 		#######################################################################
# 		# Parameters (note that not all parameters might be used)
# 		# initialize params so that sigmoid(param) = ini_sigmoid
# 		ini_sigmoid = 0.01
# 		inv_sigmoid = math.log(ini_sigmoid/(1-ini_sigmoid)) / self.sigmscale
# 		#
# 		self.var1 = torch.nn.parameter.Parameter( torch.tensor(inv_sigmoid), requires_grad=True)
# 		self.var2 = torch.nn.parameter.Parameter( torch.tensor(inv_sigmoid), requires_grad=True)

# 	def stability_fun(self, x):
# 		return (1+(1-self.theta)*x) / (1-self.theta*x)

# 	def inverse_stability_fun(self, y):
# 		if y<=0: y = torch.maximum(y, 1.0 - 1.0/(self.theta+1.e-12) + 1.e-6)
# 		return torch.clamp((1-y)/((1-y)*self.theta-1), min=-20, max=20)
# 	# @staticmethod
# 	# def inverse_stability_fun(theta, y):
# 	# 	if y is None:
# 	# 		return y
# 	# 	elif isinstance(y,list):
# 	# 		y = [ max(yi, 1 - 1.0/(theta+1.e-12) + 1.e-6) for yi in y ]
# 	# 		return [ min(20, max(-20, (1-yi) / ((1-yi)*theta-1))) for yi in y ]
# 	# 	else:
# 	# 		y = max(y, 1.0 - 1.0/(theta+1.e-12) + 1.e-6)
# 	# 		return min(20, max(-20, (1-y) / ((1-y)*theta-1)))

# 	# def set_spectral_spread(self, spread):
# 	# 	spread = [-1.0,0.0,1.0] if spread is None else spread
# 	# 	assert isinstance(spread, list) and len(spread)==3, "spread should be None or list of length 3"
# 	# 	assert spread[0]<=spread[1] and spread[1]<=spread[2], "spread[0] < spread[1] < spread[2] must be given, got spread = "+str(spread)
# 	# 	self.register_buffer('spread', torch.tensor(spread, dtype=torch.float))

# 	@staticmethod
# 	def to_tensor(input):
# 		assert isinstance(input, int) or isinstance(input, float) or (isinstance(input, list) and all(isinstance(i,float) or isinstance(i,int) for i in input)), "input should be a number or a list of numbers, got "+str(input)
# 		if isinstance(input, list):
# 			assert len(input)==1 or len(input)==2, "length of input list should be 1 or 2, got "+str(input)
# 			output = 2*input if len(input)==1 else input
# 		else:
# 			output = 2*[input]
# 		return torch.tensor(output, dtype=torch.float)

# 	@property
# 	def hparam1(self):
# 		return self._hparam1
# 	@hparam1.setter
# 	def hparam1(self, hparam1):
# 		self._hparam1 = self.to_tensor(hparam1)
# 		# self.register_buffer('_hparam1', self.to_tensor(hparam1))

# 	@property
# 	def hparam2(self):
# 		return self._hparam2
# 	@hparam2.setter
# 	def hparam2(self, hparam2):
# 		self._hparam2 = self.to_tensor(hparam2)
# 		# self.register_buffer('_param2', self.to_tensor(param2))

# 	@property
# 	def center(self):
# 		p = self.hparam1
# 		c = p[0] + torch.sigmoid(self.sigmscale*self.var1) * ( p[1] - p[0] )
# 		return c if self.mode=='spectrum' else self.inverse_stability_fun(c)
# 		# if self.mode=='spectrum':
# 		# 	return c
# 		# else: #self.mode in ['stability','stabcenter']:
# 		# 	return self.inverse_stability_fun(c)
# 	@center.setter
# 	def center(self, center):
# 		if center is None: center = 0.0
# 		self.hparam1 = center

# 	@property
# 	def radius(self):
# 		if self.mode=='stability':
# 			# p1, p2 = self.hparam1, self.hparam2
# 			# stabmin = p1[0] + torch.sigmoid(self.sigmscale*self.var1) * ( p1[1] - p1[0] )
# 			# stabmax = p2[0] + torch.sigmoid(self.sigmscale*self.var2) * ( p2[1] - p2[0] )
# 			eigmin = self.inverse_stability_fun(self.stabmin)
# 			eigmax = self.inverse_stability_fun(self.stabmax)
# 			return torch.nn.functional.relu(eigmax-eigmin.detach())
# 		else: #self.mode in ['spectrum','stabcenter']:
# 			p = self.hparam2
# 			return p[0] + torch.sigmoid(self.sigmscale*self.var2) * ( p[1] - p[0] )
# 	@radius.setter
# 	def radius(self, radius):
# 		if radius is None: radius = 1.0
# 		assert all(r>=0 for r in (radius if isinstance(radius,list) else [radius])), "radius should be nonnegative, got "+str(radius)
# 		self.hparam2 = radius

# 	@property
# 	def stabmin(self):
# 		if self.mode in ['stability','stabcenter']:
# 			p = self.hparam1
# 			c = p[0] + torch.sigmoid(self.sigmscale*self.var1) * ( p[1] - p[0] )
# 			return c
# 		else:
# 			return None
# 	@stabmin.setter
# 	def stabmin(self, stabmin):
# 		if stabmin is None: stabmin = 1.0
# 		assert all(s>=0 for s in (stabmin if isinstance(stabmin,list) else [stabmin])), "stabmin should be nonnegative, got "+str(stabmin)
# 		self.hparam1 = stabmin

# 	@property
# 	def stabmax(self):
# 		if self.mode=='stability':
# 			p = self.hparam2
# 			return p[0] + torch.sigmoid(self.sigmscale*self.var2) * ( p[1] - p[0] )
# 		else:
# 			return None
# 	@stabmax.setter
# 	def stabmax(self, stabmax):
# 		if stabmax is None: stabmax = 1.0
# 		assert all(s>=0 for s in (stabmax if isinstance(stabmax,list) else [stabmax])), "stabmax should be nonnegative, got "+str(stabmax)
# 		self.hparam2 = stabmax


# 	# def set_center(self, center):
# 	# 	center = 0.0 if center is None else center
# 	# 	if isinstance(center, list):
# 	# 		assert len(center)==1 or len(center)==2, "center should be None, number or list of length 1 or 2, got "+str(center)
# 	# 		if len(center)==2:
# 	# 			self.register_buffer('center', torch.tensor(center, dtype=torch.float))
# 	# 		else:
# 	# 			self.register_buffer('center', torch.tensor(2*center, dtype=torch.float))
# 	# 	else:
# 	# 		self.register_buffer('center', torch.tensor([center,center], dtype=torch.float))

# 	# def set_radius(self, radius):
# 	# 	radius = 1.0 if radius is None else radius
# 	# 	if isinstance(radius, list):
# 	# 		assert len(radius)==1 or len(radius)==2 and all(r>=0 for r in radius), "radius should be None, number, or list of length 1 or 2 with nonnegative elements, got "+str(radius)
# 	# 		if len(radius)==2:
# 	# 			self.register_buffer('radius', torch.tensor(radius, dtype=torch.float))
# 	# 		else:
# 	# 			self.register_buffer('radius', torch.tensor(2*radius, dtype=torch.float))
# 	# 	else:
# 	# 		assert radius>0, "radius should be None, number, or list of length 2 with positive elements"
# 	# 		self.register_buffer('radius', torch.tensor([radius,radius], dtype=torch.float))

# 	def get_spectral_spread(self):
# 		# if self.mode=='spread':
# 		# 	a = self.spread[1] + torch.sigmoid(self.var1) * ( self.spread[0] - self.spread[1] )
# 		# 	b = self.spread[1] + torch.sigmoid(self.var2) * ( self.spread[2] - self.spread[1] )
# 		# elif self.mode=='circle' or self.mode=='stabcircle':
# 		c, r = self.get_spectral_circle()
# 		a = c - r
# 		b = c + r
# 		# elif self.mode=='center':
# 		# 	c, r = self.get_spectral_circle()
# 		# 	a = c - r
# 		# 	b = c + r
# 		# else:
# 		# 	a = -1.0
# 		# 	b = 1.0
# 		return a, b

# 	def get_spectral_circle(self, stability_circle=False):
# 		return self.center, self.radius
# 		# c = self.center[0] + torch.sigmoid(self.var1) * ( self.center[1] - self.center[0] )
# 		# r = self.radius[0] + torch.sigmoid(self.var2) * ( self.radius[1] - self.radius[0] )
# 		# if self.theta is not None:
# 		# 	stabc = c
# 		# 	eigc  = self.inverse_stability_fun(c)
# 		# else:
# 		# 	eigc  = c
# 		# 	stabc = self.stability_fun(c)
# 		# if self.mode=='circle':
# 		# 	r = self.radius[0] + torch.sigmoid(self.var2) * ( self.radius[1] - self.radius[0] )
# 		# elif self.mode=='center':
# 		# 	r = torch.minimum(torch.abs(eigc), self.radius[0])
# 		# elif self.mode=='stabcircle':
# 		# 	stabr = self.radius[0] + torch.sigmoid(self.var2) * ( self.radius[1] - self.radius[0] )
# 		# 	# r = torch.clamp(self.inverse_stability_fun(stabr) - eigc.detach(), min=0.1)
# 		# 	r = torch.nn.functional.relu(self.inverse_stability_fun(stabr) - eigc.detach())
# 		# # elif self.mode=='spread':
# 		# # 	a, b = self.get_spectral_spread()
# 		# # 	c = (a+b)/2
# 		# # 	r = (b-a)/2
# 		# if stability_circle:
# 		# 	return stabc, torch.nn.functional.relu(stabr-stabc.detach())
# 		# else:
# 		# 	return eigc, r

# 	# def get_stability_circle(self):
# 	# 	c, r = self.get_spectral_circle()
# 	# 	return self.stability_fun(c), r

# 	def freeze_spectral_circle(self):
# 		self.var1.requires_grad_(False)
# 		self.var2.requires_grad_(False)

# 	def unfreeze_spectral_circle(self):
# 		self.var1.requires_grad_(True)
# 		self.var2.requires_grad_(True)

# 	def initialize(self, shape):
# 		for name, weight in self.F.named_parameters():
# 			if 'weight' in name:
# 				# torch.nn.init.xavier_normal_(weight, gain=1.e-1)
# 				# torch.nn.init.xavier_uniform_(weight)
# 				# torch.nn.init.xavier_uniform_(weight, gain=1./weight.detach().norm())
# 				torch.nn.init.uniform_(weight,-1.e-3,1.e-3)
# 				# torch.nn.init.uniform_(weight,-1.,1.)
# 			else:
# 				torch.nn.init.zeros_(weight)
# 		# perform dummy initial spectral normalization if any
# 		x = torch.ones(1,*shape)
# 		for _ in range(5):
# 			for m in self.F:
# 				m(x)

# 	def t2ind(self, t):
# 		if torch.is_tensor(t):
# 			assert t.ndim<2, "t must be either a scalar or a vector"
# 			return torch.clamp( (t/self.h).int(), max=len(self.F)-1 )
# 		else:
# 			return min(int(t/self.h), len(self.F)-1)

# 	def forward(self, t, y):
# 		ind = self.t2ind(t)
# 		if torch.is_tensor(ind) and ind.ndim>0:
# 			assert ind.size(0)==y.size(0), "if t is tensor, it must have the same batch dimension as y"
# 			# need to sacrifice full batch parallelization here
# 			f = [ self.F[i](y[batch,...]) for batch, i in enumerate(ind) ]
# 			f = torch.stack(f)
# 			# this doesn't work. why?
# 			# f = [ self.F[i](y[i==ind,...]) for i in torch.unique(ind) ]
# 			# f = torch.cat(f,0)
# 		else:
# 			f = self.F[ind](y)

# 		return self.radius * f + self.center * y
# 		# f = torch.sigmoid(self.scales) * f
# 		# c, r = self.get_spectral_circle()

# 		# return r * f + c * y



###############################################################################
###############################################################################

# def inverse_stability_theta_fun(y, theta, eigmin=-20, eigmax=20):
# 	if y<0: y = torch.maximum(y, 1.0 - 1.0/(theta+1.e-12) + 1.e-6)
# 	return torch.clamp((1-y)/((1-y)*theta-1), min=eigmin, max=eigmax)


# def spectral_localization_hook(m, input, output):
# 	center = m.center_range[1] + torch.sigmoid(m.sigscale*m.cnt_var) * ( m.center_range[1] - m.center_range[0] )
# 	radius = m.radius_range[1] + torch.sigmoid(m.sigscale*m.rad_var) * ( m.radius_range[1] - m.radius_range[0] )
# 	return center * input + radius * output
# def stability_localization_theta_hook(m, input, output):
# 	lipschitz  = m.lipschitz_range[1]  + torch.sigmoid(m.sigscale*m.lip_var) * ( m.lipschitz_range[1]  - m.lipschitz_range[0] )
# 	bottleneck = m.bottleneck_range[1] + torch.sigmoid(m.sigscale*m.btl_var) * ( m.bottleneck_range[1] - m.bottleneck_range[0] )
# 	eig_lipschitz = inverse_stability_theta_fun(lipschitz,  m.theta)
# 	eig_center    = inverse_stability_theta_fun(bottleneck, m.theta)
# 	eig_radius    = F.relu(eig_lipschitz-eig_bottleneck.detach())
# 	return eig_center * input + eig_radius * output
# class spectral_localization_hook(object):
# 	def __init__(self, sigscale=1):
# 		self.sigscale = sigscale

# 	def __call__(self, m, input, output):
# 		center = m.center_range[1] + torch.sigmoid(self.sigscale*m.cnt_var) * ( m.center_range[1] - m.center_range[0] )
# 		radius = m.radius_range[1] + torch.sigmoid(self.sigscale*m.rad_var) * ( m.radius_range[1] - m.radius_range[0] )
# 		return center * input + radius * output



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


def to_range(input: Union[_TNum, List[_TNum]]) -> List[_TNum]:
	if isinstance(input, list):
		if len(input)<1 or len(input)>3:
			raise ValueError(f"length of input list should be 1, 2 or 3, got {str(input)}")
		if len(input)==1:
			return 3*input
		elif len(input)==2:
			return [input[0],input[0],input[1]]
		else:
			return input
	else:
		return 3*[input]


def localize_spectrum(rhs, input_shape, center=0, radius=1, power_iters=1):
	# perform spectral normalization for all weights in the module
	rhs.apply(lambda m: spectral_norm( m, name='weight', input_shape=input_shape, n_power_iterations=power_iters, eps=1e-12, dim=None))

	sigscale = 2.0
	#######################################################################
	# new parameters for center and radius (note that not all parameters might be used)
	# initialize params so that sigmoid(param) = ini_sigmoid
	ini_sigmoid = 0.01
	inv_sigmoid = math.log(ini_sigmoid/(1-ini_sigmoid)) / sigscale
	#
	rhs.register_parameter('cnt_var', torch.nn.parameter.Parameter( torch.tensor(inv_sigmoid), requires_grad=True))
	rhs.register_parameter('rad_var', torch.nn.parameter.Parameter( torch.tensor(inv_sigmoid), requires_grad=True))
	#
	rhs.center_range = to_range(center)
	rhs.radius_range = to_range(radius)
	#
	rhs.sigscale = sigscale

	rhs.register_forward_hook(spectral_localization_hook)

	return rhs


def restrict_theta_stability(rhs, theta, stability_center, lipschitz_constant, min_eig=-20., max_eig=20., power_iters=1, sigscale=2.0):
	stabcenter_range = to_range(stability_center)
	lipschitz_range  = to_range(lipschitz_constant)

	#######################################################################
	# horizontal asymptote of the stability function separating two branches
	# we need to remain on the upper branch
	branch_switch_asymptote = torch.tensor(1.0 - 1.0/(theta+1.e-12) + 1.e-6, dtype=torch.float)
	def inverse_theta_stability_fun(y):
		# restrict to correct branch
		if y<=0: y = torch.maximum(y, branch_switch_asymptote)
		# restrict range of spectrum
		eigenvalue = torch.clamp((1-y)/((1-y)*theta-1), min=min_eig, max=max_eig)
		return eigenvalue

	#######################################################################
	# initialize weights
	for name, param in rhs.named_parameters():
		if isinstance(param, torch.nn.parameter.UninitializedParameter):
			if 'weight' in name and param.dim()>1:
				torch.nn.init.xavier_uniform_(param)
				# torch.nn.init.xavier_normal_(param, gain=1.e-1)
				# torch.nn.init.xavier_uniform_(param, gain=1./param.detach().norm())
				# torch.nn.init.uniform_(param,-1.e-3,1.e-3)
				# torch.nn.init.uniform_(param,-1.,1.)
			elif 'bias' in name:
				torch.nn.init.zeros_(param)

	#######################################################################
	# new parameters for center and radius (note that not all parameters might be used)
	# initialize params so that sigmoid(param) = ini_sigmoid
	ini_lip = (lipschitz_range[1]-lipschitz_range[0])   / (lipschitz_range[2]-lipschitz_range[0]+1.e-8)   + 0.0010
	ini_cnt = (stabcenter_range[1]-stabcenter_range[0]) / (stabcenter_range[2]-stabcenter_range[0]+1.e-8) + 0.0011
	inv_sigmoid = lambda ini_sigmoid: math.log(ini_sigmoid/(1-ini_sigmoid)) / sigscale
	#
	rhs.register_parameter('cnt_var', torch.nn.parameter.Parameter( torch.tensor(inv_sigmoid(ini_cnt)), requires_grad=(stabcenter_range[-1]!=stabcenter_range[0])))
	rhs.register_parameter('lip_var', torch.nn.parameter.Parameter( torch.tensor(inv_sigmoid(ini_lip)), requires_grad=(lipschitz_range[-1]!=lipschitz_range[0])))

	# add utility functions
	def freeze_spectral_circle(self):
		self.cnt_var.requires_grad_(False)
		self.lip_var.requires_grad_(False)
	def unfreeze_spectral_circle(self):
		self.cnt_var.requires_grad_(True)
		self.lip_var.requires_grad_(True)
	def get_spectral_circle(self):
		return self.center, self.radius
	rhs.freeze_spectral_circle   = MethodType(freeze_spectral_circle,   rhs)
	rhs.unfreeze_spectral_circle = MethodType(unfreeze_spectral_circle, rhs)
	rhs.get_spectral_circle      = MethodType(get_spectral_circle,      rhs)

	#######################################################################
	# dynamically add properties to the object of "per-instance class"
	def stability_center(self):
		return stabcenter_range[0] + torch.sigmoid(sigscale*self.cnt_var) * ( stabcenter_range[-1] - stabcenter_range[0] )

	def lipschitz_constant(self):
		return lipschitz_range[0]  + torch.sigmoid(sigscale*self.lip_var) * ( lipschitz_range[-1]  - lipschitz_range[0]  )

	def stability_radius(self):
		return F.relu(self.lipschitz_constant-self.stability_center)
		# return F.relu(self.lipschitz_constant-self.stability_center.detach())

	def center(self):
		return inverse_theta_stability_fun(self.stability_center)

	def radius(self):
		eig_lipschitz = inverse_theta_stability_fun(self.lipschitz_constant)
		eig_center    = inverse_theta_stability_fun(self.stability_center)
		eig_radius    = F.relu(eig_lipschitz-eig_center)
		#eig_radius    = F.relu(eig_lipschitz-eig_center.detach())
		#eig_radius    = F.softplus(eig_lipschitz-eig_center,beta=20)
		return eig_radius

	addprop(rhs, 'center', center)
	addprop(rhs, 'radius', radius)
	addprop(rhs, 'stability_center', stability_center)
	addprop(rhs, 'stability_radius', stability_radius)
	addprop(rhs, 'lipschitz_constant', lipschitz_constant)

	#######################################################################
	# perform spectral normalization for linear layers

	for f in rhs.modules():
		if isinstance(f, torch.nn.modules.conv._ConvNd) or isinstance(f, torch.nn.Linear) or isinstance(f, torch.nn.modules.batchnorm._BatchNorm):
			spectral_norm(f, name='weight', n_power_iterations=power_iters)
			# # perform dummy initial spectral normalization
			# x = torch.ones(1,*input_shape)
			# for _ in range(5):
			# 	rhs(x)

	#######################################################################
	# register forward hooks

	def theta_stability_restriction_hook(m, input, output):
		eig_lipsch = inverse_theta_stability_fun(m.lipschitz_constant)
		eig_center = inverse_theta_stability_fun(m.stability_center)
		eig_radius = F.relu(eig_lipsch-eig_center)
		# eig_radius = F.relu(eig_lipsch-eig_center.detach())
		# eig_radius = F.softplus(eig_lipsch-eig_center,beta=20)
		return eig_center * input[0] + eig_radius * output

	rhs.register_forward_hook(theta_stability_restriction_hook)

	return rhs



###############################################################################
###############################################################################


def make_rhs(model, T=1, num_steps=1):
	return rhs(model, T, num_steps)

class rhs(torch.nn.ModuleList):
	def __init__(self, model, T=1, num_steps=1):
		super().__init__(modules=[deepcopy(model) for _ in range(num_steps)])
		self.h = T / num_steps

	def t2ind(self, t):
		if torch.is_tensor(t):
			assert t.ndim<2, "t must be either a scalar or a vector"
			return torch.clamp( (t/self.h).int(), max=len(self)-1 )
		else:
			return min(int(t/self.h), len(self)-1)

	def forward(self, t, y):
		ind = self.t2ind(t)
		if torch.is_tensor(ind) and ind.ndim>0:
			assert ind.size(0)==y.size(0), "if t is tensor, it must have the same batch dimension as y"
			# need to sacrifice full batch parallelization here
			f = [ self[i](y[batch,...]) for batch, i in enumerate(ind) ]
			f = torch.stack(f)
			# this doesn't work. why?
			# f = [ self.F[i](y[i==ind,...]) for i in torch.unique(ind) ]
			# f = torch.cat(f,0)
		else:
			f = self[ind](y)

		return f


# class rhs(torch.nn.Module):
# 	def __init__(self, fn, T=1, num_steps=1, *args, **kwargs):
# 		super().__init__()
# 		self.F = torch.nn.ModuleList([fn(*args, **kwargs) for _ in range(num_steps)])
# 		self.h = T / num_steps

# 	def t2ind(self, t):
# 		if torch.is_tensor(t):
# 			assert t.ndim<2, "t must be either a scalar or a vector"
# 			return torch.clamp( (t/self.h).int(), max=len(self.F)-1 )
# 		else:
# 			return min(int(t/self.h), len(self.F)-1)

# 	def forward(self, t, y):
# 		ind = self.t2ind(t)
# 		if torch.is_tensor(ind) and ind.ndim>0:
# 			assert ind.size(0)==y.size(0), "if t is tensor, it must have the same batch dimension as y"
# 			# need to sacrifice full batch parallelization here
# 			f = [ self.F[i](y[batch,...]) for batch, i in enumerate(ind) ]
# 			f = torch.stack(f)
# 			# this doesn't work. why?
# 			# f = [ self.F[i](y[i==ind,...]) for i in torch.unique(ind) ]
# 			# f = torch.cat(f,0)
# 		else:
# 			f = self.F[ind](y)

# 		return f


# class rhs_mlp(rhs_base):
# 	# def __init__(self, dim, width, depth, T, num_steps, activation='relu', final_activation=None, power_iters=0, spectral_limits=None, center_limits=None, radius_limits=None, learn_spectral_limits=False, learn_spectral_circle=False):
# 	# 	super().__init__((dim,), T, num_steps, learn_spectral_limits, spectral_limits, learn_spectral_circle, center_limits, radius_limits)
# 	def __init__(self, dim, width, depth, T=1, num_steps=1, activation='relu', final_activation=None, bias=True, power_iters=0, **kwargs): # mode='circle', center=None, radius=None, spread=None):
# 		super().__init__(T, num_steps, **kwargs)

# 		if final_activation is None:
# 			final_activation = activation

# 		self.F = torch.nn.ModuleList( [ MLP(in_dim=dim, out_dim=dim, width=width, depth=depth, activation=activation, final_activation=final_activation, power_iters=power_iters) for _ in range(num_steps) ] )

# 		# intialize rhs
# 		self.initialize((dim,))

# class rhs_hamiltonian_mlp(rhs_base):
# 	def __init__(self, dim, width, T, num_steps, activation='relu', power_iters=0, spectral_limits=None, center_limits=None, radius_limits=None, learn_spectral_limits=False, learn_spectral_circle=False):
# 		super().__init__((dim,), T, num_steps, learn_spectral_limits, spectral_limits, learn_spectral_circle, center_limits, radius_limits)

# 		self.F = torch.nn.ModuleList( [ HamiltonianPerceptron(dim=dim, width=width, activation=activation, power_iters=power_iters) for _ in range(num_steps) ] )

# 		# intialize rhs
# 		self.initialize()

# class rhs_parabolic_mlp(rhs_base):
# 	def __init__(self, dim, width, T, num_steps, activation='relu', power_iters=0, spectral_limits=None, center_limits=None, radius_limits=None, learn_spectral_limits=False, learn_spectral_circle=False):
# 		super().__init__((dim,), T, num_steps, learn_spectral_limits, spectral_limits, learn_spectral_circle, center_limits, radius_limits)

# 		self.F = torch.nn.ModuleList( [ ParabolicPerceptron(dim=dim, width=width, activation=activation, power_iters=power_iters) for _ in range(num_steps) ] )

# 		# intialize rhs
# 		self.initialize()


# class rhs_conv2d(rhs_base):
# 	# def __init__(self, input_shape, kernel_size, depth, T, num_steps, activation='relu', bias=True, power_iters=0, spectral_limits=None, center_limits=None, radius_limits=None, learn_spectral_limits=False, learn_spectral_circle=False):
# 		# super().__init__(input_shape, T, num_steps, learn_spectral_limits, spectral_limits, learn_spectral_circle, center_limits, radius_limits)
# 	def __init__(self, input_shape, kernel_size, depth, T=1, num_steps=1, activation='relu', bias=True, power_iters=0, **kwargs): # mode='circle', center=None, radius=None, spread=None):
# 		super().__init__(T, num_steps, **kwargs)
# 		# super().__init__(T, num_steps, mode, center, radius, spread)

# 		# define rhs
# 		self.F = torch.nn.ModuleList( [ PreActConv2d(input_shape, depth=depth, kernel_size=kernel_size, activation=activation, power_iters=power_iters, bias=bias) for _ in range(num_steps) ] )

# 		# intialize rhs
# 		self.initialize(input_shape)



###############################################################################
###############################################################################

# class rhs_sum(rhs_base):
# 	def __init__(self, rhs_list):
# 		assert isinstance(rhs_list, list), "rhs_list must be a list"
# 		self.rhs_list = torch.nn.ModuleList(rhs_list)

# 	def forward(self, t, y):
# 		ind = self.t2ind(t)
# 		f = 0
# 		if torch.is_tensor(ind) and ind.ndim>0:
# 			assert ind.size(0)==y.size(0), "if t is tensor, it must have the same batch dimension as y"
# 			# need to sacrifice full batch parallelization here
# 			for rhs_i in self.rhs_list:
# 				f_i = [ rhs_i.F[i](y[batch,...]) for batch, i in enumerate(ind) ]
# 				f = f + torch.stack(f_i)
# 		else:
# 			for rhs_i in self.rhs_list:
# 				f = rhs_i.F[ind](y)

# 		f = torch.sigmoid(self.scales) * f
# 		a = self.eiginit + torch.sigmoid(self.shifta) * ( self.eigmin - self.eiginit )
# 		b = self.eiginit + torch.sigmoid(self.shiftb) * ( self.eigmax - self.eiginit )

# 		return 0.5 * ((b-a)*f + (a+b)*y)