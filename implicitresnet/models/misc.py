import time
import math

from typing import Union

import torch
from torch.nn import Linear, ReLU, Conv2d, Module, Sequential, Parameter, ModuleList, BatchNorm1d
from torch.nn.functional import linear

from ..utils import spectral_norm




###############################################################################
###############################################################################


def choose_activation(activation):
	if isinstance(activation,str):
		if activation=='linear':
			return torch.nn.Identity()
		elif activation=='relu':
			return torch.nn.ReLU()
		elif activation=='silu':
			return torch.nn.SiLU()
		elif activation=='elu':
			return torch.nn.ELU()
		elif activation=='leakyrelu':
			return torch.nn.LeakyReLU()
		elif activation=='gelu':
			return torch.nn.GELU()
		elif activation=='celu':
			return torch.nn.CELU()
		elif activation=='tanh':
			return torch.nn.Tanh()
		elif activation=='sigmoid':
			return torch.nn.Sigmoid()
		elif activation=='tanhshrink':
			return torch.nn.Tanhshrink()
		elif activation=='softsign':
			return torch.nn.Softsign()
	elif isinstance(activation,Module):
		return activation


###############################################################################################
# Perceptrons

class MLP(Sequential):
	def __init__(self, in_dim: int, out_dim: int, width: Union[int,list], depth: int,
		activation: Union[str,list] = 'relu', bias: Union[bool,list] = True, spectral_norm=False, batch_norm=False, power_iters=1) -> None:
		r""" Multilayer perceptron implemented on top of torch.nn.Sequential.
		The `depth` is the number of hidden layers, i.e., shallow network has depth 0 and the total number of layers is `depth+2`.
		If `batch_norm` is True, it is applied to all but last layer.
		If `spectral_norm` is True, it is applied to all layers including batch normalization layers.
		"""

		# activation functions of the hidden layers
		if isinstance(activation,str):
			sigma = [choose_activation(activation)]*(depth+1)
		elif len(activation)!=(depth+1):
			raise ValueError(f"`activation' must have length depth+1={depth+1}, got {activation}")

		# widths of the hidden layers
		if isinstance(width,int):
			width = [width]*(depth+1)
		elif len(width)!=(depth+1):
			raise ValueError(f"`width' must have length depth+1={depth+1}, got {width}")

		# widths of the hidden layers
		if isinstance(bias,bool):
			bias = [bias]*(depth+2)
		elif len(bias)!=(depth+1):
			raise ValueError(f"`bias' must have length depth+2={depth+2}, got {bias}")

		# batch normalization layers
		if batch_norm:
			bias  = [False]*(depth+1) + [bias[-1]]
			bnorm = [ BatchNorm1d(in_dim) ] + [ BatchNorm1d(width[d]) for d in range(depth) ]

		# linear layers
		linear = [Linear(in_dim, width[0], bias=bias[0])] + [Linear(width[d], width[d+1], bias=bias[d+1]) for d in range(depth)] + [Linear(width[depth], out_dim, bias=bias[depth+1])]

		# spectral normalization
		if spectral_norm:
			linear = [spectral_norm(l, n_power_iterations=power_iters) for l in linear]
			if batch_norm:
				bnorm = [spectral_norm(b, n_power_iterations=power_iters) for b in bnorm]

		# batch normalization
		if batch_norm:
			net = [bnorm[0],linear[0]] + [val for triple in zip(sigma[:-1],bnorm[1:],linear[1:-1]) for val in triple] + [sigma[-1],linear[-1]]
		else:
			net = [linear[0]] + [val for pair in zip(sigma,linear[1:]) for val in pair]

		super().__init__(*net)


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
		self.weight = Parameter(torch.Tensor(width, dim))
		self.bias   = Parameter(torch.Tensor(width))

		# intialize weights
		torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
		fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
		bound = 1 / math.sqrt(fan_in)
		torch.nn.init.uniform_(self.bias, -bound, bound)

		# spectral normalization
		if power_iters>0:
			spectral_norm( self, name='weight', input_shape=(dim,), n_power_iterations=power_iters, eps=1e-12, dim=None)

	def forward(self, x):
		return -linear( self.sigma(linear(x, self.weight, self.bias)), self.weight.t(), None )



class HamiltonianPerceptron(Module):
	def __init__(self, dim, width, activation='relu', power_iters=0):
		super().__init__()
		assert dim%2==0,   'dim must be power of 2 for HamiltonianPerceptron'
		assert width%2==0, 'width must be power of 2 for HamiltonianPerceptron'

		# activation function
		self.sigma = choose_activation(activation)

		# parameters
		self.weight1 = Parameter(torch.Tensor(width//2, dim//2))
		self.weight2 = Parameter(torch.Tensor(width//2, dim//2))
		self.bias1   = Parameter(torch.Tensor(width//2))
		self.bias2   = Parameter(torch.Tensor(width//2))

		# intialize weights
		torch.nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
		torch.nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
		fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight1)
		bound = 1 / math.sqrt(fan_in)
		torch.nn.init.uniform_(self.bias1, -bound, bound)
		torch.nn.init.uniform_(self.bias2, -bound, bound)

		# spectral normalization
		if power_iters>0:
			spectral_norm( self, name='weight1', input_shape=(dim//2,), n_power_iterations=power_iters, eps=1e-12, dim=None)
			spectral_norm( self, name='weight2', input_shape=(dim//2,), n_power_iterations=power_iters, eps=1e-12, dim=None)

	def forward(self, x):
		batch_dim, x_dim = x.shape
		x1, x2 = x[:,:x_dim//2], x[:,x_dim//2:]
		y1 =  linear( self.sigma(linear(x2, self.weight1, self.bias1)), self.weight1.t(), None )
		y2 = -linear( self.sigma(linear(x1, self.weight2, self.bias2)), self.weight2.t(), None )
		return torch.cat( (y1, y2), 1 )



class LinearParabolic(Linear):
	def __init__(self, dim, bias=True):
		super().__init__(dim,dim,bias)

	def forward(self, x):
		return functional.linear(x, -self.weight.t()@self.weight, self.bias)



class LinearHyperbolic(Linear):
	def __init__(self, dim, bias=True):
		super().__init__(dim,dim,bias)

	def forward(self, x):
		return linear(x, self.weight.t()-self.weight, self.bias)



###############################################################################################

class PreActConv2d(Module):
	def __init__(self, im_shape, depth, kernel_size, activation='relu', power_iters=0, bias=True):
		super().__init__()

		channels = im_shape[0]

		# activation function of hidden layers
		sigma = choose_activation(activation)

		# conv layers
		conv_hid = [ Conv2d(channels, channels, kernel_size, padding=kernel_size//2, bias=bias) for _ in range(depth) ]

		# spectral normalization
		if power_iters>0:
			conv_hid = [ spectral_norm(conv, name='weight', input_shape=im_shape, n_power_iterations=power_iters, eps=1e-12, dim=None) for conv in conv_hid ]

		# normalization layers
		# norm_hid = [ torch.nn.BatchNorm2d(channels) for _ in range(depth) ]

		# network
		# self.net = torch.nn.Sequential(*[val for triple in zip(norm_hid,[sigma]*depth,conv_hid) for val in triple])
		self.net = Sequential(*[val for pair in zip([sigma]*depth,conv_hid) for val in pair])

	def forward(self, x):
		return self.net(x)

