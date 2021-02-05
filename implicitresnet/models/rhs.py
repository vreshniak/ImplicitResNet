from abc import ABCMeta, abstractmethod

import math

import torch
from .misc import ParabolicPerceptron, HamiltonianPerceptron, HollowMLP, MLP, PreActConv2d





###############################################################################
###############################################################################


_collect_stat = True


###############################################################################
###############################################################################


class rhs_base(torch.nn.Module, metaclass=ABCMeta):
	def __init__(self, shape, T, num_steps, spectral_limits=None, learn_scales=False, learn_shift=False):
		super().__init__()
		self.shape = shape
		self.h     = T / num_steps
		self.learn_scales = learn_scales
		self.learn_shift  = learn_shift

		# initialize shifts so that sigmoid(shift) = ini_sigmoid
		ini_sigmoid = 0.99
		inv_sigmoid = math.log(ini_sigmoid/(1-ini_sigmoid))
		self.shifta = torch.nn.parameter.Parameter( torch.tensor(inv_sigmoid), requires_grad=learn_shift)
		self.shiftb = torch.nn.parameter.Parameter( torch.tensor(inv_sigmoid), requires_grad=learn_shift)

		# initialize scales so that sigmoid(scales) = ini_sigmoid
		ini_sigmoid = 0.5 if learn_scales else 0.99
		inv_sigmoid = math.log(ini_sigmoid/(1-ini_sigmoid))
		self.scales = torch.nn.parameter.Parameter( inv_sigmoid * torch.ones(1,*shape), requires_grad=learn_scales)

		self.set_spectral_limits(spectral_limits)

	def set_spectral_limits(self, spectral_limits):
		assert isinstance(spectral_limits,list) or (spectral_limits is None), "spectral_limits should be a list or None"
		if spectral_limits is None:
			self.eigmin, self.eiginit, self.eigmax = ( -1.0, 0.0, 1.0 )
		elif len(spectral_limits)==2:
			self.eigmin, self.eigmax = spectral_limits
			self.eiginit = 0.5*(self.eigmin+self.eigmax)
		elif len(spectral_limits)==3:
			self.eigmin, self.eiginit, self.eigmax = spectral_limits

		if self.learn_shift:
			ini_sigmoid_a = 0.1
			ini_sigmoid_b = ini_sigmoid_a * (self.eigmax-self.eiginit)/(self.eiginit-self.eigmin) # balance initial shifta and shiftb
			b = math.log(ini_sigmoid_a/(1-ini_sigmoid_a))
			a = math.log(ini_sigmoid_b/(1-ini_sigmoid_b))
			torch.nn.init.constant_(self.shifta, a)
			torch.nn.init.constant_(self.shiftb, b)


	def initialize(self):
		for name, weight in self.F.named_parameters():
			if 'weight' in name:
				# torch.nn.init.xavier_normal_(weight, gain=1.e-1)
				torch.nn.init.xavier_uniform_(weight)
				torch.nn.init.xavier_uniform_(weight, gain=1./weight.detach().norm())
				# torch.nn.init.uniform_(weight,-1.e-5,1.e-5)
			else:
				torch.nn.init.zeros_(weight)
		# perform dummy initial spectral normalization if any
		x = torch.ones(1,*self.shape)
		for _ in range(5):
			for m in self.F:
				m(x)

	def t2ind(self, t):
		if torch.is_tensor(t):
			assert t.ndim<2, "t must be either a scalar or a vector"
			return torch.clamp( (t/self.h).int(), max=len(self.F)-1 )
		else:
			return min(int(t/self.h), len(self.F)-1)


	def forward(self, t, y):
		ind = self.t2ind(t)
		if torch.is_tensor(ind):
			# need to sacrifice full batch parallelization here
			f = [ self.F[i](y[batch,...]) for batch, i in enumerate(ind) ]
			f = torch.stack(f)
			# this doesn't work. why?
			# f = [ self.F[i](y[i==ind,...]) for i in torch.unique(ind) ]
			# f = torch.cat(f,0)
		else:
			f = self.F[ind](y)

		f = torch.sigmoid(self.scales) * f
		a = self.eiginit + torch.sigmoid(self.shifta) * ( self.eigmin - self.eiginit )
		b = self.eiginit + torch.sigmoid(self.shiftb) * ( self.eigmax - self.eiginit )

		return 0.5 * ((b-a)*f + (a+b)*y)



###############################################################################
###############################################################################



class rhs_mlp(rhs_base):
	def __init__(self, dim, width, depth, T, num_steps, activation='relu', final_activation=None, power_iters=0, spectral_limits=None, learn_scales=False, learn_shift=False):
		super().__init__((dim,), T, num_steps, spectral_limits, learn_scales, learn_shift)

		if final_activation is None:
			final_activation = activation

		self.F = torch.nn.ModuleList( [ MLP(in_dim=dim, out_dim=dim, width=width, depth=depth, activation=activation, final_activation=final_activation, power_iters=power_iters) for _ in range(num_steps) ] )

		# intialize rhs
		self.initialize()

class rhs_hamiltonian_mlp(rhs_base):
	def __init__(self, dim, width, T, num_steps, activation='relu', power_iters=0, spectral_limits=None, learn_scales=False, learn_shift=False):
		super().__init__((dim,), T, num_steps, spectral_limits, learn_scales, learn_shift)

		self.F = torch.nn.ModuleList( [ HamiltonianPerceptron(dim=dim, width=width, activation=activation, power_iters=power_iters) for _ in range(num_steps) ] )

		# intialize rhs
		self.initialize()

class rhs_parabolic_mlp(rhs_base):
	def __init__(self, dim, width, T, num_steps, activation='relu', power_iters=0, spectral_limits=None, learn_scales=False, learn_shift=False):
		super().__init__((dim,), T, num_steps, spectral_limits, learn_scales, learn_shift)

		self.F = torch.nn.ModuleList( [ ParabolicPerceptron(dim=dim, width=width, activation=activation, power_iters=power_iters) for _ in range(num_steps) ] )

		# intialize rhs
		self.initialize()


class rhs_conv2d(rhs_base):
	def __init__(self, input_shape, kernel_size, depth, T, num_steps, activation='relu', power_iters=0, spectral_limits=None, learn_scales=False, learn_shift=False):
		super().__init__(input_shape, T, num_steps, spectral_limits, learn_scales, learn_shift)

		# define rhs
		self.F = torch.nn.ModuleList( [ PreActConv2d(input_shape, depth=depth, kernel_size=kernel_size, activation=activation, power_iters=power_iters) for _ in range(num_steps) ] )

		# intialize rhs
		self.initialize()
