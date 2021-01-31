import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from abc import ABCMeta, abstractmethod

import argparse
import yaml
from random import randint
import numpy as np
import math

from pathlib import Path

from collections import deque

import torch
from src import layers, utils
from src.utilities import calc
from src.utilities import TraceJacobianReg
# import utils
# import layers




###############################################################################
###############################################################################


_gpu   = torch.device('cuda')
_cpu   = torch.device('cpu')
_dtype = torch.float

_collect_stat = True


###############################################################################
###############################################################################




def parse_args():
	parser = argparse.ArgumentParser()
	####################
	parser.add_argument("--optdir",   default="options.yml" )
	####################
	parser.add_argument("--prefix",   default=None )
	parser.add_argument("--name",     default=None )
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
	####################
	# rhs spectral properties
	# parser.add_argument("--scales",   type=str,   default="equal", choices=["equal", "learn"])
	parser.add_argument("--piters",   type=int,   default=0   )
	parser.add_argument("--eiglim",   type=float, default=[0,0],   nargs='+')
	parser.add_argument("--stablim",  type=float, default=[0,1.5], nargs='+')
	parser.add_argument("--stabinit", type=float, default=0.75)
	parser.add_argument("--eigscale", type=str,   default="fixed", choices=["fixed", "learn"])
	parser.add_argument("--eigshift", type=str,   default="fixed", choices=["fixed", "learn"])
	# parser.add_argument("--eigmap",   type=str,   default=None, choices=["shift", "rational"])
	# parser.add_argument("--minrho",   type=float, default=0   )
	# parser.add_argument("--maxrho",   type=float, default=-1  )
	####################
	# data params
	parser.add_argument("--datasize",  type=int,   default=500 )
	parser.add_argument("--datasteps", type=int,   default=-1  )
	parser.add_argument("--datanoise", type=str,   default="gaussian" )
	parser.add_argument("--noisesize", type=float, default=0.0 )
	####################
	# training params
	parser.add_argument("--init",      type=str,   default="rnd", choices=["init", "cont", "rnd", "zero"])
	parser.add_argument("--lr",        type=float, default=0.01)
	parser.add_argument("--batch",     type=int,   default=-1  )
	parser.add_argument("--epochs",    type=int,   default=1000)
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
	if os.path.exists(Path(args.optdir)):
		with open(args.optdir, 'r') as f:
			opts = yaml.load(f, Loader=yaml.Loader)
			for key, value in opts.items():
				args.__dict__[key] = value

	if args.steps<=0:
		assert int(args.T)==args.T
		args.steps = int(args.T)
	if args.datasteps<0:
		del args.datasteps

	assert args.eiglim[0]<=args.eiglim[1]
	assert args.stablim[0]<args.stablim[1]
	assert args.stabinit>=args.stablim[0] and args.stabinit<=args.stablim[1]
	if args.theta>0:
		assert args.stablim[0]>(args.theta-1)/args.theta, "lower bound of stability function for theta=%.2f should be greater than %.2e, got args.stablim[0] = %.2e"%(args.theta, (args.theta-1)/args.theta, args.stablim[0])
	# assert args.stablim[1]>=1, "upper bound of stability function must be greater than 1, got args.stablim[1] = "+str(args.stablim[1])
	return args



def option_type(option):
	opt2type = {'prefix': str,
				'name': str,
				'mode': str,
				'seed': int,
				'method': str,
				'theta': float,
				'tol': float,
				'T': float,
				'steps': int,
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
				'ajdiag': float,
				'ajac': float,
				'atan': float,
				'af': float,
				'aresid': float,
				'mciters': int
				}
	return opt2type[option]


def make_name(args, verbose=True):
	sep   = '|'
	opsep = '_'
	if verbose: print("\n-------------------------------------------------------------------")
	# file_name = str(args.prefix)+sep if args.prefix is not None and args.prefix!=opsep else ''
	file_name = ''
	max_len = 0
	for arg in vars(args):
		length  = len(arg)
		max_len = length if length>max_len else max_len
	max_len += 1
	ignore = ['prefix', 'sigma', 'mode', 'steps', 'eiglim', 'tol', 'ajdiag', 'diaval']
	for arg, value in vars(args).items():
		if value is not None:
			if verbose: print("{0:>{length}}: {1}".format(arg,str(value),length=max_len))
			# if arg!='prefix' and arg!='sigma' and arg!='mode' and arg!='method' and arg!='steps' and arg!='eigs' and arg!='tol' and arg!='ajdiag' and arg!='diaval': # and arg!='minrho' and arg!='maxrho':
			if arg not in ignore:
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
		file_name += 'eiglim'+opsep+str(args.eiglim[0])+opsep+str(args.eiglim[1])+sep
	if args.prefix is not None:
		file_name = str(args.prefix) + file_name
	if verbose: print("-------------------------------------------------------------------")
	return args.name if args.name is not None else file_name[:-len(sep)]


def get_options_from_name(name):
	sep   = '|'
	opsep = '_'

	options = {}

	opts = name.split(sep)
	for opt in opts:
		opts = opt.split(opsep)
		if len(opts)==2:
			opt_name, opt_val = opts
			opt_val = option_type(opt_name)(opt_val)
		else:
			opt_name = opts[0]
			opt_val  = [ option_type(opt_name)(op) for op in opts[1:] ]
		options[opt_name] = opt_val
		# print(opt_name)
		# opt_name, opt_val = opt.split(opsep)
		# options[opt_name] = option_type(opt_name)(opt_val)
	return options




###############################################################################
###############################################################################





def get_optimizer(name, model, lr, wdecay=0):
	if name=='adam':
		return torch.optim.Adam(model.parameters(),    lr=lr, weight_decay=wdecay)
	elif name=='rms':
		return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=wdecay)
	elif name=='sgd':
		return torch.optim.SGD(model.parameters(),     lr=lr, weight_decay=wdecay, momentum=0.5)
	elif name=='lbfgs':
		return torch.optim.LBFGS(model.parameters(),   lr=1., max_iter=100, max_eval=None, tolerance_grad=1.e-9, tolerance_change=1.e-9, history_size=100, line_search_fn='strong_wolfe')



def create_paths(args):
	# subdir = "mlp" if args.prefix is None else args.prefix
	# if args.mode=="init": subdir += "/init"

	Path("checkpoints","init").mkdir(parents=True, exist_ok=True)
	Path("checkpoints","epoch_0").mkdir(parents=True, exist_ok=True)
	Path("checkpoints","epoch_last").mkdir(parents=True, exist_ok=True)
	Path("output","images").mkdir(parents=True, exist_ok=True)
	Path("output","data").mkdir(parents=True, exist_ok=True)

	paths = {
		'initialization': Path('checkpoints','init'),
		'checkpoints_0':  Path('checkpoints','epoch_0'),
		'checkpoints':    Path('checkpoints','epoch_last'),
		'output':         Path('output'),
		'output_images':  Path("output","images"),
		'output_data':    Path("output","data")
	}

	return paths
	# checkpoint_dir_0 = Path("checkpoints",subdir,'epoch0',file_name)
	# checkpoint_dir   = Path("checkpoints",subdir,file_name)
	# checkpoint_dir_0 = Path("checkpoints",'epoch0')
	# checkpoint_dir   = Path("checkpoints")
	# out_dir = Path("out")
	# return checkpoint_dir_0, checkpoint_dir, out_dir


def load_model(model, args, device=_cpu, location=None):
	mod = model.to(device=device)

	if location is not None:
		load_dir = location
	else:
		paths     = create_paths(args)
		file_name = make_name(args, verbose=False)

		if args.mode=='init':
			return mod
		if args.mode=='train':
			if args.init=='rnd':
				return mod
			elif args.init=="init":
				load_dir = Path(paths['initialization'],'%4.2f'%(args.theta))
				# import re
				# load_dir = Path( checkpoint_dir, re.sub('theta_\d*.\d*','theta_'+str(args.theta),file_name) )
			elif args.init=="cont":
				load_dir = Path(paths['checkpoints'], file_name)
		if args.mode=='plot' or args.mode=='test':
			load_dir = Path(paths['checkpoints'], file_name)

		# if args.init=='rnd' and args.mode!='plot' and args.mode!='test':
		# 	return mod
		# else:
		# 	paths     = create_paths(args)
		# 	file_name = make_name(args)

		# 	# initialize model
		# 	if args.mode=='train':
		# 		if args.init=="init":
		# 			load_dir = Path(paths['initialization'],'%4.2f'%(args.theta))
		# 			# import re
		# 			# load_dir = Path( checkpoint_dir, re.sub('theta_\d*.\d*','theta_'+str(args.theta),file_name) )
		# 		elif args.init=="cont":
		# 			load_dir = Path(paths['checkpoints'], file_name)
		# 	else:
		# 		load_dir = Path(paths['checkpoints'], file_name)

	missing_keys, unexpected_keys = mod.load_state_dict(torch.load(load_dir, map_location=device))
	mod.apply(lambda m: setattr(m,'theta',args.theta))
	print('Mode: ', args.mode)
	print('Load model from: ',load_dir)
	print('\tmissing_keys:    ', missing_keys)
	print('\tunexpected_keys: ', unexpected_keys)
	return mod






###############################################################################
###############################################################################




class rhs_base(torch.nn.Module, metaclass=ABCMeta):
	def __init__(self, shape, args):
		super().__init__()
		self.args = args

		# divergence and Jacobian regularizers
		# self.divjacreg  = TraceJacobianReg(n=args.mciters)
		# self.jacdiagreg = utils.JacDiagReg(n=args.mciters, value=args.diaval)

		###############################
		# scales

		# self.maxshift = 1./max(0.2,self.args.theta) if math.isnan(self.args.eigs[1]) else np.abs(self.args.eigs[1])
		# self.maxrho   = 1.e2 if math.isnan(self.args.eigs[0]) else np.abs(self.args.eigs[0])
		# self.maxrho   = 1./max(0.02,np.abs(self.args.theta-0.5)) if math.isnan(self.args.eigs[0]) else np.abs(self.args.eigs[0])
		# self.maxrho  *= 0.5
		# self.maxrho  += self.maxshift
		# min_eig = -1./max(0.01,np.abs(args.theta-0.5)) if math.isnan(args.eigs[0]) else args.eigs[0]

		# if spectrum not given, stability function (1+(1-theta)*z)/(1-theta*z) should be in [-1,inf]
		# min_eig = -1./max(0.1,0.5-args.theta) if math.isnan(args.eiglim[0]) else args.eiglim[0]
		# if spectrum not given, stability function (1+(1-theta)*z)/(1-theta*z) should be in [0,inf]
		# min_eig = -1./max(0.1,1-args.theta)   if math.isnan(args.eiglim[0]) else args.eiglim[0]
		# min_eig = 1./min(1.0/args.eiglim[0],args.theta-1)
		# max_eig = 1./max(0.1,args.theta)     if math.isnan(args.eiglim[1]) else args.eiglim[1]

		# choose eiglims such that stability function (1+(1-theta)*z)/(1-theta*z) is in stablims
		if args.eiglim[0]==args.eiglim[1]:
			min_eig = max(-10, (1-args.stablim[0])/((1-args.stablim[0])*args.theta-1) )
			max_eig = min( 10, (1-args.stablim[1])/((1-args.stablim[1])*args.theta-1) )
		else:
			min_eig, max_eig = args.eiglim
		assert max_eig>min_eig
		# if not math.isnan(args.eiglim[0]): min_eig = max(min_eig, args.eiglim[0])
		# if not math.isnan(args.eiglim[1]): max_eig = min(max_eig, args.eiglim[1])
		# assert max_eig>min_eig

		# if args.scales=='learn' and args.piters>0:
		# 	if self.max_rho>0:
		# 		initrho = 0.1*self.max_rho
		# 		self._scales = torch.nn.parameter.Parameter( np.log(initrho/(self.max_rho-initrho)) * self.ones_like_input(), requires_grad=True)
		# 	else:
		# 		self.register_buffer('_scales', torch.tensor([0.0], dtype=torch.float))
		# 	self.register_buffer('_eigshift', torch.tensor([(min_eig+max_eig)/2], dtype=torch.float))
		# else:
		# 	self.register_buffer('_scales',   torch.tensor([1.0], dtype=torch.float))
		# 	self.register_buffer('_eigshift', torch.tensor([0.0], dtype=torch.float))

		############################

		self.shape = shape

		self.eigscale = args.eigscale
		self.eigshift = args.eigshift
		# self.eigmap   = args.eigmap
		self.eigmin   = min_eig
		self.eigmax   = max_eig
		self.eiginit  = (1-args.stabinit)/((1-args.stabinit)*args.theta-1)
		# self.eigdif   = (max_eig - min_eig) / 2
		# self.eigsum   = (max_eig + min_eig) / 2
		# self.eigprod  =  min_eig * max_eig

		# self.register_buffer('_eigmin', min_eig)
		# self.register_buffer('_eigmax', max_eig)
		# self.register_buffer('_eiginit', (1-args.stabinit)/((1-args.stabinit)*args.theta-1))

		if self.eigscale=='learn':
			assert args.piters>0, "if eigscale=='learn', spectral normalization should be performed and hence piters must be positive"
			# initialize _scales so that sigmoid(_scales) = a
			a = 0.5
			self._scales = torch.nn.parameter.Parameter( np.log(a/(1-a)) * torch.ones(1,*shape), requires_grad=True)
		else:
			self.register_parameter('_scales', None)

		if self.eigshift=='learn':
			assert args.piters>0, "if eigshift=='learn', spectral normalization should be performed and hence piters must be positive"
			# initialize _shifta, _shiftb so that sigmoid(_shifta) = sigmoid(_shiftb) = a
			a = 0.1
			self._shifta = torch.nn.parameter.Parameter( torch.tensor(np.log(a/(1-a))), requires_grad=True)
			self._shiftb = torch.nn.parameter.Parameter( torch.tensor(np.log(a/(1-a))), requires_grad=True)
		else:
			self.register_parameter('_shifta', None)
			self.register_parameter('_shiftb', None)

		# if args.eigmap=='shift':
		# 	self.register_buffer('_eigshift', torch.tensor((min_eig+max_eig)/2, dtype=torch.float))
		# elif args.eigmap=='rational':
		# 	a = min_eig + max_eig
		# 	b = min_eig - max_eig
		# 	self.eigmap = lambda x,F: 2*min_eig*max_eig*torch.sigmoid(self._scales)*F / (a*torch.sigmoid(self._scales)*F+b)

		# elif args.scales=='equal':
		# 	self._scales   = torch.tensor([1.0]) if math.isnan(self.args.eigs[0]) else torch.tensor([self.maxrho],   dtype=torch.float)
		# 	self._eigshift = torch.tensor([0.0]) if math.isnan(self.args.eigs[1]) else torch.tensor([self.maxshift], dtype=torch.float)


	# def scale(self, f):
	# 	if self.eigscale=='learn':
	# 		return torch.sigmoid(self._scales) * f
	# 	else:
	# 		return f

	# @abstractmethod
	# def ones_like_input(self):
	# 	pass


	def initialize(self):
		for name, weight in self.F.named_parameters():
			if 'weight' in name:
				# torch.nn.init.xavier_normal_(weight, gain=1.e-1)
				# torch.nn.init.xavier_normal_(weight)
				torch.nn.init.xavier_uniform_(weight)
				torch.nn.init.xavier_uniform_(weight, gain=1./weight.detach().norm())
				# torch.nn.init.uniform_(weight,-1.e-5,1.e-5)
			else:
				torch.nn.init.zeros_(weight)
		# perform initial spectral normalization
		if self.args.piters>0:
			self.spectral_normalization(torch.ones(1,*self.shape), 10)
			# self.spectral_normalization(self.ones_like_input(), 10)


	# divergence and jacobian of the vector field
	# def divjac(self, t, y):
	# 	# return self.divjacreg( self.F(t), y )
	# 	return self.divjacreg( lambda x: self.forward(t,x), y )

	# divergence and jacobian of the vector field
	# def jacdiag(self, t, y):
	# 	# return self.jacdiagreg( self.F(t), y )
	# 	return self.jacdiagreg( lambda x: self.forward(t,x), y )

	# # derivative of the tangential component in the tangential direction
	# def Ftan(self, t, y):
	# 	batch_dim = y.size(0)
	# 	fun   = lambda z: self.F(t)(z).reshape((batch_dim,-1)).pow(2).sum(dim=1, keepdim=True)
	# 	Fnorm = torch.clamp( self.F(t)(y).reshape((batch_dim,-1)).norm(dim=1, keepdim=True), min=1.e-16 )
	# 	return 0.5 * (utils.directional_derivative( fun, y, F ) / Fnorm).sum() / batch_dim

	# @property
	# def regularizer(self):
	# 	# Total variation like regularizer
	# 	reg = {}
	# 	if self.args.aTV<=0 or len(self.rhs)==1:
	# 		return reg
	# 	for t in range(len(self.rhs)-1):
	# 		for w2, w1 in zip(self.rhs[t+1].parameters(), self.rhs[t].parameters()):
	# 			reg['TV'] = reg.get('TV',0) + ( w2 - w1 ).pow(2).sum() #* (t+1)**2
	# 	reg['TV'] = (self.args.aTV / self.args.steps) * reg['TV']
	# 	return reg


	# @property
	# def scales(self):
	# 	return self._scales
	# 	# if self.args.scales=='learn' and self.args.piters>0:
	# 	# 	return self.max_rho * torch.sigmoid( self._scales )
	# 	# 	# return self.maxrho * torch.sigmoid( self._scales )
	# 	# 	# 0.5 because eigs of F(y)-y are in [-2,0]
	# 	# else:
	# 	# 	return self._scales


	# @property
	# def eigshift(self):
	# 	return self._eigshift
	# 	# if self.args.scales=='learn' and self.maxshift>0:
	# 	# 	return self.maxshift * torch.sigmoid( self._eigshift )
	# 	# else:
	# 	# 	return self._eigshift

	@property
	def spectrum_bounds(self):
		a = self.eigmin if self._shifta is None else self.eiginit + torch.sigmoid(self._shifta)*(self.eigmin-self.eiginit)
		b = self.eigmax if self._shiftb is None else self.eiginit + torch.sigmoid(self._shiftb)*(self.eigmax-self.eiginit)
		return (a, b)

	def spectral_normalization(self, y, iters=1):
		if self.args.piters>0:
			for _ in range(iters):
				for t in range(len(self.F)):
					self.F[t](y)
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

	def eigenvalues(self, t, data):
		from src.utilities.spectral import eigenvalues
		return eigenvalues( lambda x: self.forward(t,x), data)
		# # with torch.enable_grad():
		# data = torch.tensor(data).requires_grad_(True)
		# batch_dim = data.size(0)
		# data_dim  = data.numel() // batch_dim

		# # import scipy.sparse.linalg as sla
		# # from scipy.sparse.linalg import eigs, svds
		# # # using scipy sparse LinearOperator
		# # numpy_dtype = {torch.float: np.float, torch.float32: np.float32, torch.float64: np.float64}
		# # # 'Matrix-vector' product of the linear operator
		# # def matvec(v):
		# # 	v0 = torch.from_numpy(v).view_as(data).to(device=data.device, dtype=data.dtype)
		# # 	Av = torch.autograd.grad(self.forward(t,data), data, grad_outputs=v0, create_graph=False, retain_graph=True, only_inputs=True)[0]
		# # 	return Av.cpu().detach().numpy().ravel()
		# # A_dot = sla.LinearOperator(dtype=numpy_dtype[data.dtype], shape=(data.numel(),data.numel()), matvec=matvec)
		# #
		# # # using scipy sparse matrix
		# # A_dot  = utils.jacobian( self.forward(t,data), data, True ).reshape( data.numel(),data.numel() ).cpu().detach().numpy()
		# # eigvals  = eigs(A_dot, k=data.numel()-2, return_eigenvectors=False).reshape((-1,1))
		# # singvals = svds(A_dot, k=5, return_singular_vectors=False)

		# jacobian = utils.jacobian( self.forward(t,data), data, True ).reshape( batch_dim, data_dim, batch_dim, data_dim ).cpu().detach().numpy()
		# eigvals  = np.linalg.eigvals([ jacobian[i,:,i,:] for i in range(batch_dim) ]).reshape((-1,1))
		# # singvals = np.linalg.svd(jacobian.reshape( data.numel(),data.numel() ), compute_uv=False)
		# # singvals = np.linalg.svd([ jacobian[i,:,i,:] for i in range(batch_dim) ], compute_uv=False)
		# return np.hstack(( np.real(eigvals), np.imag(eigvals) ))


	def forward(self, t, y, p=None):
		# th = max(self.args.theta,0.2)
		# if self.args.prefix=='par' or self.args.prefix=='ham':
		# 	alpha = 1 / (self.args.maxrho*th)
		# else:
		# 	alpha = 1 / (self.args.maxrho*th) - 1
		# return self.scales * ( alpha * y + self.F(t)(y) )
		# ind = self.t2ind(t)
		# return self.scales * (self.rhs[ind](y) - y) + self.eigshift * y
		# f = self.scale(self.rhs[ind](y))
		# if self.eigmap is None:
		# 	return f
		# elif self.eigmap=="shift":
		# 	return self.eigdif * f + self.eigsum * y

		ind = self.t2ind(t)
		f, a, b = self.F[ind](y), self.eigmin, self.eigmax

		if self.eigscale=='learn':
			f = torch.sigmoid(self._scales) * f
		if self.eigshift=='learn':
			a = self.eiginit + torch.sigmoid(self._shifta)*(self.eigmin-self.eiginit)
			b = self.eiginit + torch.sigmoid(self._shiftb)*(self.eigmax-self.eiginit)
		# if not self.training:
		# 	print("%.2f(%.2f) %.2f(%.2f) %.2f %.2f"%(a.item(), self.eigmin, b.item(), self.eigmax, 1+a.item(),1+b.item()))
		# m = 0.0
		# if self.eigshift=='learn' and self.eigmin<-m and self.eigmax>m:
		# 	a = -m + torch.sigmoid(self._shifta)*(self.eigmin+m)
		# 	b =  m + torch.sigmoid(self._shiftb)*(self.eigmax-m)
		return 0.5 * ((b-a)*f + (a+b)*y)





class rhs_mlp(rhs_base):
	def __init__(self, data_dim, args, final_activation=None):
		# dimension of the state space
		self.dim = data_dim+args.codim
		super().__init__([shape,], args)

		# activation
		if len(args.sigma)==1:
			sigma, final_activation = args.sigma[0], None
		elif len(args.sigma)==2:
			sigma, final_activation = args.sigma
		else:
			assert False, 'args.sigma should have either 1 or 2 entries'

		# depth of rhs
		rhs_depth = args.steps if args.aTV>=0 else 1

		# structured rhs
		structure = args.prefix if args.prefix is not None else 'mlp'
		if structure=='par':
			self.F = torch.nn.ModuleList( [ layers.ParabolicPerceptron( dim=self.dim, width=args.width, activation=sigma, power_iters=args.piters) for _ in range(rhs_depth) ] )
		elif structure=='ham':
			self.F = torch.nn.ModuleList( [ layers.HamiltonianPerceptron( dim=self.dim, width=args.width, activation=sigma, power_iters=args.piters) for _ in range(rhs_depth) ] )
		elif structure=='hol':
			self.F = torch.nn.ModuleList( [ layers.HollowMLP(dim=self.dim, width=args.width, depth=args.depth, activation=sigma, final_activation=final_activation, power_iters=args.piters) for _ in range(rhs_depth) ] )
		elif structure=='mlp':
			self.F = torch.nn.ModuleList( [ layers.MLP(in_dim=self.dim, out_dim=self.dim, width=args.width, depth=args.depth, activation=sigma, final_activation=final_activation, power_iters=args.piters) for _ in range(rhs_depth) ] )

		# intialize rhs
		self.initialize()


	def ones_like_input(self):
		return torch.ones((1,self.dim), dtype=torch.float)




class rhs_conv2d(rhs_base):
	def __init__(self, input_shape, kernel_size, depth, power_iters, args):
		super().__init__(input_shape, args)

		self.shape = input_shape
		self.depth = depth

		# define rhs
		self.F = torch.nn.ModuleList( [ layers.PreActConv2d(input_shape, depth=depth, kernel_size=kernel_size, activation='relu', power_iters=power_iters) for _ in range(depth) ] )

		# intialize rhs
		self.initialize()





###############################################################################
###############################################################################


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
		p = 2 if alpha['TV']>0 else 0
		c = (((t+1)/n)**p for t in range(n))

		# evaluate divergence and jacobian along the trajectory
		if alpha['div']!=0 or alpha['jac']!=0 or _collect_stat:
			divjac = [ calc.trace_and_jacobian( lambda x: rhs(t, x), y) for t, y in zip(solver._t, solver._y) ]

		if _collect_stat:
			stat['rhs/'+name+'div'] = trapz([divt[0] for divt in divjac], dx)
			stat['rhs/'+name+'jac'] = trapz([jact[1] for jact in divjac], dx=dx/dim)
			stat['rhs/'+name+'f']   = trapz([rhs(t,y).pow(2).sum() for t, y in zip(solver._t, solver._y)], dx)

		# divergence
		if alpha['div']!=0:
			reg[name+'div'] = alpha['div'] * trapz([ct*divt[0] for ct, divt in zip(c, divjac)], dx)

		# jacobian
		if alpha['jac']!=0:
			reg[name+'jac'] = alpha['jac'] * (stat['rhs/'+name+'jac'] if _collect_stat else trapz([jact[1] for jact in divjac], dx=dx/dim))

		# magnitude
		if alpha['f']!=0:
			reg[name+'f'] = alpha['f'] * (stat['rhs/'+name+'f'] if _collect_stat else trapz([rhs(t,y).pow(2).sum() for t, y in zip(solver._t, solver._y)], dx))

		# residual
		if alpha['resid']!=0:
			for step in range(n-1):
				x, y = solver._y[step].detach(), solver._y[step+1].detach()
				reg[name+'residual'] = reg.get(name+'residual',0) + solver.residual(solver._t[step], x, y)
			reg[name+'residual'] = (alpha['resid'] / steps) * reg[name+'residual']

		# 'Total variation'
		if alpha['TV']!=0 and rhs.depth>1:
			w1 = torch.nn.utils.parameters_to_vector(rhs.F[0].parameters())
			for t in range(rhs.depth-1):
				w2 = torch.nn.utils.parameters_to_vector(rhs.F[t+1].parameters())
				reg[name+'TV'] = ( w2 - w1 ).pow(2).sum()
				w1 = w2
			reg[name+'TV'] = (alpha['TV'] / steps) * reg[name+'TV']

		# note that solver.training has not been changed by rhs.eval()
		rhs.train(mode=solver.training)

		setattr(solver, 'regularizer', reg)
		if _collect_stat:
			setattr(rhs, 'statistics', stat)
	return None


def regularized_ode_solver(solver, args):
	setattr(solver, 'alpha', {'div':args.adiv, 'jac':args.ajac, 'f':args.af, 'resid':args.aresid, 'TV':args.aTV})
	solver.register_forward_hook(compute_regularizers_and_statistics)
	return solver


###############################################################################
###############################################################################

# class ode_block_base(torch.nn.Module):
# 	def __init__(self, args):
# 		super().__init__()
# 		self.args = args
# 		# setattr(self, 'alpha', {'div':args.adiv, 'jac':args.ajac, 'f':args.af, 'resid':args.aresid, 'TV':args.aTV})

# 	# def forward(self, y0, t0=0):
# 	# 	return self.ode(y0, t0)
# 	# def forward(self, y0, t0=0, evolution=False):
# 	# 	ode_out = self.ode(y0, t0, evolution=evolution)

# 	# 	reg   = {}
# 	# 	rhs   = self.ode.rhs
# 	# 	name  = self.ode.name+'_'
# 	# 	alpha = self.alpha

# 	# 	n   = len(self.ode._t)
# 	# 	dim = y0.numel() / y0.size(0)

# 	# 	# spectral normalization has to be performed only once per forward pass, so freeze here
# 	# 	rhs.eval()

# 	# 	p = 2 if alpha['TV']>0 else 0
# 	# 	c = [((t+1)/n)**p for t in range(n)]

# 	# 	if alpha['div']!=0 or alpha['jac']!=0:
# 	# 		divjac = [rhs.divjac(t, y.detach()) for t, y in zip(self.ode._t, self.ode._y)]

# 	# 	# divergence regularizer
# 	# 	if alpha['div']!=0:
# 	# 		# reg[name+'div'] = alpha['div'] * trapz([ct*divt[0] for ct, divt in zip(c,divjac)], dx=1/(n-1)/dim)
# 	# 		reg[name+'div'] = 0.5 * ( c[0]*divjac[0][0] + c[-1]*divjac[-1][0])
# 	# 		for i in range(1,len(c)-1):
# 	# 			reg[name+'div'] = reg[name+'div'] + c[i]*divjac[i][0]
# 	# 		reg[name+'div'] = alpha['div'] *reg[name+'div']/(n-1)/dim

# 	# 	rhs.train(mode=self.ode.training)
# 	# 	setattr(self, 'regularizer', reg)

# 	# 	return ode_out
# 	def forward(self, y0, t0=0, evolution=False):
# 		# if self.training and self.args.piters>0:
# 		# 	self.ode.rhs.spectral_normalization(y=y0)
# 		# self.ode.rhs.eval()
# 		self.ode_out = self.ode(y0, t0, evolution=True)
# 		# self.ode.rhs.train(mode=self.training)
# 		return self.ode_out if evolution else self.ode_out[-1]

# 	@property
# 	def regularizer(self):
# 		reg  = {}
# 		rhs  = self.ode.rhs
# 		name = self.ode.name+'_'
# 		args = self.args

# 		# spectral normalization has to be performed only once per forward pass, so freeze here
# 		rhs.eval()

# 		if args.aTV>0:
# 			p = 2
# 		else:
# 			p = 0
# 		# p = 2

# 		# trapezoidal rule
# 		y0, yT = self.ode_out[0].detach(), self.ode_out[-1].detach()
# 		if args.adiv>0 or args.ajac>0:
# 			div0, jac0 = rhs.divjac(0,y0)
# 			divT, jacT = rhs.divjac(args.T,yT)
# 			if args.adiv>0: reg[name+'div'] = 0.5 * ( div0/(args.steps+1)**p + divT)
# 			if args.ajac>0: reg[name+'jac'] = 0.5 * ( jac0 + jacT )
# 		if args.ajdiag>0: reg[name+'jdiag'] = 0.5 * ( rhs.jacdiag(0,y0) + rhs.jacdiag(args.T,yT) )
# 		if args.af>0:     reg[name+'f']     = 0.5 * ( rhs(0,y0).pow(2).sum() + rhs(args.T,yT).pow(2).sum() )
# 		# if args.atan!=0:  reg[name+'tan']   = 0.5 * ( rhs.Ftan(0,y0) + rhs.Ftan(args.T,yT) )
# 		for t in range(1,args.steps):
# 			y = self.ode_out[t].detach()
# 			if args.adiv>0 or args.ajac>0:
# 				divt, jact = rhs.divjac(t,y)
# 				if args.adiv>0: reg[name+'div'] = reg[name+'div']   + ((t+1)/(args.steps+1))**p * divt #- ((args.atan * rhs.Ftan(t,y)) if args.atan>0 else 0)
# 				if args.ajac>0: reg[name+'jac'] = reg[name+'jac']   + jact
# 			if args.ajdiag>0: reg[name+'jdiag'] = reg[name+'jdiag'] + rhs.jacdiag(t,y)
# 			if args.af>0:    reg[name+'f']      = reg[name+'f']     + rhs(t,y).pow(2).sum()
# 			# if args.atan!=0: reg[name+'tan'] = reg[name+'tan'] + rhs.Ftan(t,y)

# 		dim = y0.numel() / y0.size(0)
# 		if args.af>0:     reg[name+'f']     = (args.af     / args.steps / dim)    * reg[name+'f']
# 		if args.ajac>0:   reg[name+'jac']   = (args.ajac   / args.steps / dim**2) * reg[name+'jac']
# 		if args.adiv>0:   reg[name+'div']   = (args.adiv   / args.steps / dim)    * reg[name+'div'] #( reg[name+'div'] + (args.T * args.max_rho if args.power_iters>0 else 0) )
# 		if args.ajdiag>0: reg[name+'jdiag'] = (args.ajdiag / args.steps / dim)    * reg[name+'jdiag']
# 		# if args.atan!=0:  reg[name+'tan']   = (args.atan   / args.steps / dim)    * reg[name+'tan']

# 		if args.aresid>0:
# 			for step in range(args.steps):
# 				x, y = self.ode_out[step].detach(), self.ode_out[step+1].detach()
# 				reg[name+'residual'] = reg.get(name+'residual',0) + self.ode.residual(step, [x,y])
# 			reg[name+'residual'] = (args.aresid / args.steps) * reg[name+'residual']

# 		rhs.train(mode=self.training)

# 		return reg

# 	@property
# 	def statistics(self):
# 		stat = {}
# 		if _collect_stat:
# 			rhs  = self.ode.rhs
# 			name = 'rhs/'+self.ode.name+'_'
# 			args = self.args

# 			# spectral normalization has to be performed only once per forward pass, so freeze here
# 			rhs.eval()

# 			# # trapezoidal rule
# 			# y0, yT = self.ode_out[0], self.ode_out[-1]
# 			# div0, jac0 = rhs.divjac(0,y0)
# 			# divT, jacT = rhs.divjac(args.T,yT)
# 			# stat[name+'div']   = 0.5 * ( div0 + divT)
# 			# stat[name+'jdiag'] = 0.5 * ( rhs.jacdiag(0,y0) + rhs.jacdiag(args.T,yT) )
# 			# stat[name+'jac']   = 0.5 * ( jac0 + jacT )
# 			# stat[name+'f']     = 0.5 * ( rhs(0,y0).pow(2).sum() + rhs(args.T,yT).pow(2).sum() )
# 			# for t in range(1,args.steps):
# 			# 	y = self.ode_out[t]
# 			# 	divt, jact = rhs.divjac(t,y)
# 			# 	stat[name+'div']   = stat[name+'div']   + divt
# 			# 	stat[name+'jdiag'] = stat[name+'jdiag'] + rhs.jacdiag(t,y)
# 			# 	stat[name+'jac']   = stat[name+'jac']   + jact
# 			# 	stat[name+'f']     = stat[name+'f']     + rhs(t,y).pow(2).sum()

# 			# dim = y0.numel() / y0.size(0)
# 			# stat[name+'f']     = stat[name+'f']     / args.steps / dim
# 			# stat[name+'jdiag'] = stat[name+'jdiag'] / args.steps / dim
# 			# stat[name+'jac']   = stat[name+'jac']   / args.steps / dim**2
# 			# stat[name+'div']   = stat[name+'div']   / args.steps / dim
# 			# trapezoidal rule
# 			y0, yT = self.ode_out[0].detach(), self.ode_out[-1].detach()
# 			div0, jac0 = rhs.divjac(0,y0)
# 			divT, jacT = rhs.divjac(args.T,yT)
# 			stat[name+'div']   = 0.5 * ( div0 + divT)
# 			# stat[name+'jdiag'] = 0.5 * ( rhs.jacdiag(0,y0) + rhs.jacdiag(args.T,yT) )
# 			stat[name+'jac']   = 0.5 * ( jac0 + jacT )
# 			stat[name+'f']     = 0.5 * ( rhs(0,y0).pow(2).sum() + rhs(args.T,yT).pow(2).sum() )
# 			for t in range(1,args.steps):
# 				y = self.ode_out[t].detach()
# 				divt, jact = rhs.divjac(t,y)
# 				stat[name+'div']   = stat[name+'div']   + divt
# 				# stat[name+'jdiag'] = stat[name+'jdiag'] + rhs.jacdiag(t,y)
# 				stat[name+'jac']   = stat[name+'jac']   + jact
# 				stat[name+'f']     = stat[name+'f']     + rhs(t,y).pow(2).sum()

# 			dim = y0.numel() / y0.size(0)
# 			stat[name+'f']     = stat[name+'f'].detach()     / args.steps / dim
# 			# stat[name+'jdiag'] = stat[name+'jdiag'].detach() / args.steps / dim
# 			stat[name+'jac']   = stat[name+'jac'].detach()   / args.steps / dim**2
# 			stat[name+'div']   = stat[name+'div'].detach()   / args.steps / dim

# 			rhs.train(mode=self.training)

# 		return stat












