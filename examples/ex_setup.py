import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from abc import ABCMeta, abstractmethod

import argparse
from random import randint
import numpy as np
import math

import torch
import utils
import layers


def parse_args():
	parser = argparse.ArgumentParser()
	####################
	parser.add_argument("--prefix",   default=None )
	parser.add_argument("--mode",     type=str,   default="train", choices=["init", "train", "plot", "test"] )
	parser.add_argument("--seed",     type=int,   default=randint(0,10000))
	####################
	# resnet params
	# parser.add_argument("--method",   type=str,   default="inner", choices=["inner", "outer"])
	parser.add_argument("--theta",    type=float, default=0.0   )
	parser.add_argument("--tol",      type=float, default=1.e-6 )
	parser.add_argument("--T",        type=float, default=1     )
	parser.add_argument("--steps",    type=int,   default=-1    )
	####################
	# rhs params
	parser.add_argument("--codim",    type=int,   default=1   )
	parser.add_argument("--width",    type=int,   default=2   )
	parser.add_argument("--depth",    type=int,   default=3   )
	parser.add_argument("--sigma",    type=str,   default="relu", nargs='+', choices=["relu", "gelu", "celu", "tanh"])
	# parser.add_argument("--sigma",    type=str,   default="relu", choices=["relu", "gelu", "celu", "tanh"])
	# rhs spectral properties
	parser.add_argument("--scales",   type=str,   default="equal", choices=["equal", "learn"])
	parser.add_argument("--piters",   type=int,   default=0   )
	parser.add_argument("--eigs",     type=float, default=[math.nan,math.nan], nargs='+')
	# parser.add_argument("--minrho",   type=float, default=0   )
	# parser.add_argument("--maxrho",   type=float, default=-1  )
	####################
	# training params
	parser.add_argument("--datasize",  type=int,   default=500 )
	parser.add_argument("--datasteps", type=int,   default=-1  )
	parser.add_argument("--batch",     type=int,   default=-1  )
	parser.add_argument("--init",      type=str,   default="rnd", choices=["init", "cont", "rnd", "zero"])
	parser.add_argument("--epochs",    type=int,   default=1000)
	parser.add_argument("--lr",        type=float, default=0.01)
	####################
	# regularizers
	parser.add_argument("--wdecay", type=float, default=0   )
	parser.add_argument("--aTV",    type=float, default=0   )
	parser.add_argument("--adiv",   type=float, default=0   )
	parser.add_argument("--ajac",   type=float, default=0   )
	parser.add_argument("--ajdiag", type=float, default=0   )
	parser.add_argument("--diaval", type=float, default=0   )
	# parser.add_argument("--atan",   type=float, default=0   )
	parser.add_argument("--af",     type=float, default=0   )
	parser.add_argument("--aresid", type=float, default=0   )
	parser.add_argument("--mciters",type=int,   default=1   )
	# parser.add_argument("--ax",     type=float, default=0   )

	args = parser.parse_args()
	if args.steps<=0:
		assert int(args.T)==args.T
		args.steps = int(args.T)
	if args.datasteps<0:
		del args.datasteps

	return args



def option_type(option):
	opt2type = {'prefix': str,
				'mode': str,
				'seed': int,
				'method': str,
				'theta': float,
				'tol': float,
				'T': int,
				'codim': int,
				'width': int,
				'depth': int,
				'sigma': str,
				'scales': str,
				'piters': int,
				'minrho': float,
				'maxrho': float,
				'init': str,
				'epochs': int,
				'lr': float,
				'datasize': int,
				'datasteps': int,
				'batch': int,
				'wdecay': float,
				'aTV': float,
				'adiv': float,
				'ajac': float,
				'atan': float,
				'af': float,
				'aresid': float,
				}
	return opt2type[option]


def make_name(args):
	sep   = '|'
	opsep = '_'
	print("\n-------------------------------------------------------------------")
	# file_name = str(args.prefix)+sep if args.prefix is not None and args.prefix!=opsep else ''
	file_name = ''
	max_len = 0
	for arg in vars(args):
		length  = len(arg)
		max_len = length if length>max_len else max_len
	max_len += 1
	for arg, value in vars(args).items():
		if value is not None:
			print("{0:>{length}}: {1}".format(arg,str(value),length=max_len))
			if arg!='prefix' and arg!='sigma' and arg!='mode' and arg!='method' and arg!='steps' and arg!='eigs' and arg!='tol' and arg!='ajdiag' and arg!='diaval': # and arg!='minrho' and arg!='maxrho':
				file_name += arg+opsep+str(value)+sep
			if arg=='steps' and value>0:
				file_name += arg+opsep+str(value)+sep
			if arg=='sigma':
				file_name += arg
				for sigma in value:
					file_name += opsep+str(sigma)
				file_name += sep
			if arg=='ajdiag':
				file_name += arg+opsep+str(value)+opsep+str(args.diaval)+sep
			# if arg!='prefix' and arg!='mode' and arg!='theta' and arg!='method' and arg!='min_rho' and arg!='max_rho':
			# 	file_name += arg+opsep+str(value)+sep
			# if arg=='theta':
			# 	file_name += ('theta_in' if args.method=='inner' else 'theta_out')+opsep+str(value)+sep
	if args.piters>0:
		# file_name += 'rho'+opsep+str(args.minrho)+opsep+str(args.maxrho)+sep
		file_name += 'eigs'+opsep+str(args.eigs[0])+opsep+str(args.eigs[1])+sep
	print("-------------------------------------------------------------------")
	return file_name[:-len(sep)]


def get_options_from_name(name):
	sep   = '|'
	opsep = '_'

	options = {}

	opts = name.split(sep)
	for opt in opts:
		opt_name, opt_val = opt.split(opsep)
		options[opt_name] = option_type(opt_name)(opt_val)
	return options




###############################################################################
###############################################################################




class rhs_base(torch.nn.Module, metaclass=ABCMeta):
	def __init__(self, args):
		super().__init__()
		self.args = args

		# divergence and Jacobian regularizer
		self.divjacreg  = utils.TraceJacobianReg(n=args.mciters)
		self.jacdiagreg = utils.JacDiagReg(n=args.mciters, value=args.diaval)

		# self.maxshift = 1./max(0.2,self.args.theta) if math.isnan(self.args.eigs[1]) else np.abs(self.args.eigs[1])
		# self.maxrho   = 1.e2 if math.isnan(self.args.eigs[0]) else np.abs(self.args.eigs[0])
		# self.maxrho   = 1./max(0.02,np.abs(self.args.theta-0.5)) if math.isnan(self.args.eigs[0]) else np.abs(self.args.eigs[0])
		# self.maxrho  *= 0.5
		# self.maxrho  += self.maxshift

		self.min_eig = -1./max(0.01,np.abs(self.args.theta-0.5)) if math.isnan(self.args.eigs[0]) else self.args.eigs[0]
		self.max_eig =  1./max(0.01,self.args.theta)             if math.isnan(self.args.eigs[1]) else self.args.eigs[1]
		assert self.max_eig>self.min_eig
		self.max_rho = (self.max_eig - self.min_eig) / 2


	def initialize(self):
		for name, weight in self.rhs.named_parameters():
			if 'weight' in name:
				# torch.nn.init.xavier_normal_(weight, gain=1.e-1)
				# torch.nn.init.xavier_normal_(weight)
				torch.nn.init.xavier_uniform_(weight)
				torch.nn.init.xavier_uniform_(weight, gain=1./weight.detach().norm())
				# torch.nn.init.uniform_(weight,-1.e-5,1.e-5)
			else:
				torch.nn.init.zeros_(weight)


	# divergence and jacobian of the vector field
	def divjac(self, t, y):
		# return self.divjacreg( self.F(t), y )
		return self.divjacreg( lambda x: self.forward(t,x), y )

	# divergence and jacobian of the vector field
	def jacdiag(self, t, y):
		# return self.jacdiagreg( self.F(t), y )
		return self.jacdiagreg( lambda x: self.forward(t,x), y )


	# # derivative of the tangential component in the tangential direction
	# def Ftan(self, t, y):
	# 	batch_dim = y.size(0)
	# 	fun   = lambda z: self.F(t)(z).reshape((batch_dim,-1)).pow(2).sum(dim=1, keepdim=True)
	# 	Fnorm = torch.clamp( self.F(t)(y).reshape((batch_dim,-1)).norm(dim=1, keepdim=True), min=1.e-16 )
	# 	return 0.5 * (utils.directional_derivative( fun, y, F ) / Fnorm).sum() / batch_dim


	@property
	def regularizer(self):
		# Total variation like regularizer
		reg = {}
		if self.args.aTV<=0 or len(self.rhs)==1:
			return reg
		for t in range(len(self.rhs)-1):
			for w2, w1 in zip(self.rhs[t+1].parameters(), self.rhs[t].parameters()):
				reg['TV'] = reg.get('TV',0) + ( w2 - w1 ).pow(2).sum() #* (t+1)**2
		reg['TV'] = (self.args.aTV / self.args.steps) * reg['TV']
		return reg


	@property
	def scales(self):
		if self.args.scales=='learn' and self.args.piters>0:
			return self.max_rho * torch.sigmoid( self._scales )
			# return self.maxrho * torch.sigmoid( self._scales )
			# 0.5 because eigs of F(y)-y are in [-2,0]
		else:
			return self._scales


	@property
	def eigshift(self):
		return self._eigshift
		# if self.args.scales=='learn' and self.maxshift>0:
		# 	return self.maxshift * torch.sigmoid( self._eigshift )
		# else:
		# 	return self._eigshift


	def spectral_normalization(self, y, iters=1):
		if self.args.piters>0:
			for _ in range(iters):
				for t in range(len(self.rhs)):
					self.rhs[t](y)
					# self.rhs[t](torch.ones((1,self.dim)))


	# @abstractmethod
	# def t2ind(self, t):
	# 	pass
	def t2ind(self, t):
		args = self.args
		h = args.T / args.steps
		if args.aTV>=0:
			return int(t/h) if t<args.T else args.steps-1
		else:
			return 0

	def spectrum(self, t, data):
		data = torch.tensor(data).requires_grad_(True)
		batch_dim = data.size(0)
		data_dim  = data.numel() // batch_dim
		jacobian  = utils.jacobian( self.forward(t,data), data, True ).reshape( batch_dim, data_dim, batch_dim, data_dim ).detach().numpy()
		eigvals   = np.linalg.eigvals([ jacobian[i,:,i,:] for i in range(batch_dim) ]).reshape((-1,1))
		return np.hstack(( np.real(eigvals), np.imag(eigvals) ))


	def forward(self, t, y, p=None):
		# th = max(self.args.theta,0.2)
		# if self.args.prefix=='par' or self.args.prefix=='ham':
		# 	alpha = 1 / (self.args.maxrho*th)
		# else:
		# 	alpha = 1 / (self.args.maxrho*th) - 1
		# return self.scales * ( alpha * y + self.F(t)(y) )
		ind = self.t2ind(t)
		# return self.scales * (self.rhs[ind](y) - y) + self.eigshift * y
		return self.scales * self.rhs[ind](y) + self._eigshift * y




class rhs_mlp(rhs_base):
	def __init__(self, data_dim, args, final_activation=None):
		super().__init__(args)

		# sigmas = args.sigma.split('_')
		if len(args.sigma)==1:
			sigma = args.sigma[0]
			final_activation = None
		elif len(args.sigma)==2:
			sigma, final_activation = args.sigma
		# final_activation = args.sigma if final_activation is None else final_activation

		# dimension of the state space
		dim = data_dim+args.codim
		self.dim = dim

		###############################
		# scales of each dimension
		# if args.scales=='learn':
		# 	if self.maxrho>0:
		# 		# initrho = 1.0 if self.maxrho>1 else self.maxrho
		# 		# initrho = 0.1*self.maxrho
		# 		# self._scales = torch.nn.parameter.Parameter( np.log(initrho/(self.maxrho-initrho)) * torch.ones((1,dim), dtype=torch.float), requires_grad=True)
		# 	else:
		# 		self._scales = torch.tensor([0.0], dtype=torch.float)

		# 	if self.maxshift>0:
		# 		# initshift = 0.5 if self.maxshift>1 else 0.5*self.maxshift
		# 		initshift = 0.1*self.maxshift
		# 		self._eigshift = torch.nn.parameter.Parameter( np.log(initshift/(self.maxshift-initshift)) * torch.ones((1,dim), dtype=torch.float), requires_grad=True)
		# 	else:
		# 		self._eigshift = torch.tensor([0.0], dtype=torch.float)
		# elif args.scales=='equal':
		# 	self._scales   = torch.tensor([1.0]) if math.isnan(self.args.eigs[0]) else torch.tensor([self.maxrho],   dtype=torch.float)
		# 	self._eigshift = torch.tensor([1.0]) if math.isnan(self.args.eigs[1]) else torch.tensor([self.maxshift], dtype=torch.float)
		if args.scales=='learn' and args.piters>0:
			if self.max_rho>0:
				initrho = 0.1*self.max_rho
				self._scales = torch.nn.parameter.Parameter( np.log(initrho/(self.max_rho-initrho)) * torch.ones((1,dim), dtype=torch.float), requires_grad=True)
			else:
				self._scales = torch.tensor([0.0], dtype=torch.float)
			self._eigshift = torch.tensor([(self.min_eig+self.max_eig)/2], dtype=torch.float)
		else:
			self._scales   = torch.tensor([1.0])
			self._eigshift = torch.tensor([0.0])
		# elif args.scales=='equal':
		# 	self._scales   = torch.tensor([1.0]) if math.isnan(self.args.eigs[0]) else torch.tensor([self.maxrho],   dtype=torch.float)
		# 	self._eigshift = torch.tensor([0.0]) if math.isnan(self.args.eigs[1]) else torch.tensor([self.maxshift], dtype=torch.float)


		###############################
		F_depth = args.steps if args.aTV>=0 else 1

		# structured rhs
		structure = args.prefix if args.prefix is not None else 'mlp'
		if structure=='par':
			self.rhs = torch.nn.ModuleList( [ layers.ParabolicPerceptron( dim=dim, width=args.width, activation=sigma, power_iters=args.piters) for _ in range(F_depth) ] )
		elif structure=='ham':
			self.rhs = torch.nn.ModuleList( [ layers.HamiltonianPerceptron( dim=dim, width=args.width, activation=sigma, power_iters=args.piters) for _ in range(F_depth) ] )
		elif structure=='hol':
			self.rhs = torch.nn.ModuleList( [ layers.HollowMLP(dim=dim, width=args.width, depth=args.depth, activation=sigma, final_activation=final_activation, power_iters=args.piters) for _ in range(F_depth) ] )
		elif structure=='mlp':
			self.rhs = torch.nn.ModuleList( [ layers.MLP(in_dim=dim, out_dim=dim, width=args.width, depth=args.depth, activation=sigma, final_activation=final_activation, power_iters=args.piters) for _ in range(F_depth) ] )


		###############################
		# intialization
		self.initialize()
		# perform initial spectral normalization
		if args.piters>0:
			self.spectral_normalization(torch.ones((1,dim)), 10)




class rhs_conv2d(rhs_base):
	def __init__(self, channels, args):
		super().__init__(args)

		if args.scales=='learn' and args.piters>0:
			if self.max_rho>0:
				initrho = 0.1*self.max_rho
				self._scales = torch.nn.parameter.Parameter( np.log(initrho/(self.max_rho-initrho)) * torch.ones((1,channels,1,1), dtype=torch.float), requires_grad=True)
			else:
				self.register_buffer('_scales', torch.tensor([0.0], dtype=torch.float))
			self.register_buffer('_eigshift', torch.tensor([(self.min_eig+self.max_eig)/2], dtype=torch.float))
		else:
			self.register_buffer('_scales',   torch.tensor([1.0], dtype=torch.float))
			self.register_buffer('_eigshift', torch.tensor([0.0], dtype=torch.float))
		# elif args.scales=='equal':
		# 	self._scales   = torch.tensor([1.0]) if math.isnan(self.args.eigs[0]) else torch.tensor([self.maxrho],   dtype=torch.float)
		# 	self._eigshift = torch.tensor([0.0]) if math.isnan(self.args.eigs[1]) else torch.tensor([self.maxshift], dtype=torch.float)


		###############################
		F_depth = args.steps if args.aTV>=0 else 1

		# rhs
		self.rhs = torch.nn.ModuleList( [ layers.PreActConv2d(channels, depth=args.depth, kernel_size=3, activation='relu', power_iters=0) for _ in range(F_depth) ] )


		###############################
		# intialization
		self.initialize()
		# perform initial spectral normalization
		if args.piters>0:
			self.spectral_normalization(torch.ones((1,channels,5,5)), 10)


		# optimizer = torch.optim.SGD(self.parameters(), lr=1.e-2, momentum=0.5)
		# y = torch.ones((1,channels,5,5))
		# for i in range(100):
		# 	div = 0
		# 	for t in range(args.steps):
		# 		divt, _ = self.divjac(t, y)
		# 		div = div + divt
		# 	(0-div).backward()
		# 	self.spectral_normalization(y, 1)
		# 	optimizer.step()






###############################################################################
###############################################################################





class ode_block_base(torch.nn.Module):
	def __init__(self, args):
		super().__init__()
		self.args = args

	def forward(self, y0, t0=0, evolution=False):
		if self.training and self.args.piters>0:
			self.ode.rhs.spectral_normalization(y=y0)
		self.ode.rhs.eval()
		self.ode_out = self.ode(y0, t0)
		self.ode.rhs.train(mode=self.training)
		return self.ode_out if evolution else self.ode_out[-1]

	@property
	def regularizer(self):
		reg  = {}
		rhs  = self.ode.rhs
		name = self.ode.name+'_'
		args = self.args

		# spectral normalization has to be performed only once per forward pass, so freeze here
		rhs.eval()

		p = 2

		# trapezoidal rule
		y0, yT = self.ode_out[0].detach(), self.ode_out[-1].detach()
		if args.adiv>0 or args.ajac>0:
			div0, jac0 = rhs.divjac(0,y0)
			divT, jacT = rhs.divjac(args.T,yT)
			if args.adiv>0: reg[name+'div'] = 0.5 * ( div0/(args.steps+1)**p + divT)
			if args.ajac>0: reg[name+'jac'] = 0.5 * ( jac0 + jacT )
		if args.ajdiag>0: reg[name+'jdiag'] = 0.5 * ( rhs.jacdiag(0,y0) + rhs.jacdiag(args.T,yT) )
		if args.af>0:     reg[name+'f']     = 0.5 * ( rhs(0,y0).pow(2).sum() + rhs(args.T,yT).pow(2).sum() )
		# if args.atan!=0:  reg[name+'tan']   = 0.5 * ( rhs.Ftan(0,y0) + rhs.Ftan(args.T,yT) )
		for t in range(1,args.steps):
			y = self.ode_out[t].detach()
			if args.adiv>0 or args.ajac>0:
				divt, jact = rhs.divjac(t,y)
				if args.adiv>0: reg[name+'div'] = reg[name+'div']   + ((t+1)/(args.steps+1))**p * divt #- ((args.atan * rhs.Ftan(t,y)) if args.atan>0 else 0)
				if args.ajac>0: reg[name+'jac'] = reg[name+'jac']   + jact
			if args.ajdiag>0: reg[name+'jdiag'] = reg[name+'jdiag'] + rhs.jacdiag(t,y)
			if args.af>0:    reg[name+'f']      = reg[name+'f']     + rhs(t,y).pow(2).sum()
			# if args.atan!=0: reg[name+'tan'] = reg[name+'tan'] + rhs.Ftan(t,y)

		dim = y0.numel() / y0.size(0)
		if args.af>0:     reg[name+'f']     = (args.af     / args.steps / dim)    * reg[name+'f']
		if args.ajac>0:   reg[name+'jac']   = (args.ajac   / args.steps / dim**2) * reg[name+'jac']
		if args.adiv>0:   reg[name+'div']   = (args.adiv   / args.steps / dim)    * reg[name+'div'] #( reg[name+'div'] + (args.T * args.max_rho if args.power_iters>0 else 0) )
		if args.ajdiag>0: reg[name+'jdiag'] = (args.ajdiag / args.steps / dim)    * reg[name+'jdiag']
		# if args.atan!=0:  reg[name+'tan']   = (args.atan   / args.steps / dim)    * reg[name+'tan']

		if args.aresid>0:
			for step in range(args.steps):
				x, y = self.ode_out[step].detach(), self.ode_out[step+1].detach()
				reg[name+'residual'] = reg.get(name+'residual',0) + self.ode.residual(step, [x,y])
			reg[name+'residual'] = (args.aresid / args.steps) * reg[name+'residual']

		rhs.train(mode=self.training)

		return reg

	@property
	def statistics(self):
		# with torch.no_grad():
		stat = {}
		# rhs  = self.ode.rhs
		# name = 'rhs/'+self.ode.name+'_'
		# args = self.args

		# # spectral normalization has to be performed only once per forward pass, so freeze here
		# rhs.eval()

		# # # trapezoidal rule
		# # y0, yT = self.ode_out[0], self.ode_out[-1]
		# # div0, jac0 = rhs.divjac(0,y0)
		# # divT, jacT = rhs.divjac(args.T,yT)
		# # stat[name+'div']   = 0.5 * ( div0 + divT)
		# # stat[name+'jdiag'] = 0.5 * ( rhs.jacdiag(0,y0) + rhs.jacdiag(args.T,yT) )
		# # stat[name+'jac']   = 0.5 * ( jac0 + jacT )
		# # stat[name+'f']     = 0.5 * ( rhs(0,y0).pow(2).sum() + rhs(args.T,yT).pow(2).sum() )
		# # for t in range(1,args.steps):
		# # 	y = self.ode_out[t]
		# # 	divt, jact = rhs.divjac(t,y)
		# # 	stat[name+'div']   = stat[name+'div']   + divt
		# # 	stat[name+'jdiag'] = stat[name+'jdiag'] + rhs.jacdiag(t,y)
		# # 	stat[name+'jac']   = stat[name+'jac']   + jact
		# # 	stat[name+'f']     = stat[name+'f']     + rhs(t,y).pow(2).sum()

		# # dim = y0.numel() / y0.size(0)
		# # stat[name+'f']     = stat[name+'f']     / args.steps / dim
		# # stat[name+'jdiag'] = stat[name+'jdiag'] / args.steps / dim
		# # stat[name+'jac']   = stat[name+'jac']   / args.steps / dim**2
		# # stat[name+'div']   = stat[name+'div']   / args.steps / dim
		# # trapezoidal rule
		# y0, yT = self.ode_out[0].detach(), self.ode_out[-1].detach()
		# div0, jac0 = rhs.divjac(0,y0)
		# divT, jacT = rhs.divjac(args.T,yT)
		# stat[name+'div']   = 0.5 * ( div0 + divT)
		# # stat[name+'jdiag'] = 0.5 * ( rhs.jacdiag(0,y0) + rhs.jacdiag(args.T,yT) )
		# stat[name+'jac']   = 0.5 * ( jac0 + jacT )
		# stat[name+'f']     = 0.5 * ( rhs(0,y0).pow(2).sum() + rhs(args.T,yT).pow(2).sum() )
		# for t in range(1,args.steps):
		# 	y = self.ode_out[t].detach()
		# 	divt, jact = rhs.divjac(t,y)
		# 	stat[name+'div']   = stat[name+'div']   + divt
		# 	# stat[name+'jdiag'] = stat[name+'jdiag'] + rhs.jacdiag(t,y)
		# 	stat[name+'jac']   = stat[name+'jac']   + jact
		# 	stat[name+'f']     = stat[name+'f']     + rhs(t,y).pow(2).sum()

		# dim = y0.numel() / y0.size(0)
		# stat[name+'f']     = stat[name+'f'].detach()     / args.steps / dim
		# # stat[name+'jdiag'] = stat[name+'jdiag'].detach() / args.steps / dim
		# stat[name+'jac']   = stat[name+'jac'].detach()   / args.steps / dim**2
		# stat[name+'div']   = stat[name+'div'].detach()   / args.steps / dim

		# rhs.train(mode=self.training)

		return stat












