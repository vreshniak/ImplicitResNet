import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from random import randint
import numpy as np

import torch
import utils
import layers


def make_parser():
	parser = argparse.ArgumentParser()
	####################
	parser.add_argument("--prefix",   default=None )
	parser.add_argument("--mode",     type=str,   default="train", choices=["init", "train", "plot", "test"] )
	parser.add_argument("--seed",     type=int,   default=randint(0,10000))
	####################
	# resnet params
	parser.add_argument("--method",   type=str,   default="inner", choices=["inner", "outer"])
	parser.add_argument("--theta",    type=float, default=0.0   )
	parser.add_argument("--tol",      type=float, default=1.e-6 )
	parser.add_argument("--T",        type=int,   default=1     )
	####################
	# rhs params
	parser.add_argument("--codim",    type=int,   default=1   )
	parser.add_argument("--width",    type=int,   default=2   )
	parser.add_argument("--depth",    type=int,   default=3   )
	parser.add_argument("--sigma",    type=str,   default="relu", choices=["relu", "gelu", "celu", "tanh"])
	# rhs spectral properties
	parser.add_argument("--scales",   type=str,   default="equal", choices=["equal", "learn", "ortho"])
	parser.add_argument("--piters",   type=int,   default=0   )
	parser.add_argument("--minrho",   type=float, default=1.e-5)
	parser.add_argument("--maxrho",   type=float, default=2    )
	####################
	# training params
	parser.add_argument("--init",     type=str,   default="rnd", choices=["warmup", "chkp", "rnd", "zero"])
	parser.add_argument("--epochs",   type=int,   default=1000)
	parser.add_argument("--lr",       type=float, default=0.01)
	parser.add_argument("--datasize", type=int,   default=500 )
	parser.add_argument("--batch",    type=int,   default=-1  )
	####################
	# regularizers
	parser.add_argument("--wdecay", type=float, default=0   )
	parser.add_argument("--aTV",    type=float, default=0   )
	parser.add_argument("--adiv",   type=float, default=0   )
	parser.add_argument("--ajac",   type=float, default=0   )
	parser.add_argument("--ajdiag", type=float, default=0   )
	parser.add_argument("--atan",   type=float, default=0   )
	parser.add_argument("--af",     type=float, default=0   )
	parser.add_argument("--aresid", type=float, default=0   )
	# parser.add_argument("--ax",     type=float, default=0   )

	return parser

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
			if arg!='prefix' and arg!='mode' and arg!='method' and arg!='minrho' and arg!='maxrho':
				file_name += arg+opsep+str(value)+sep
			# if arg!='prefix' and arg!='mode' and arg!='theta' and arg!='method' and arg!='min_rho' and arg!='max_rho':
			# 	file_name += arg+opsep+str(value)+sep
			# if arg=='theta':
			# 	file_name += ('theta_in' if args.method=='inner' else 'theta_out')+opsep+str(value)+sep
	if args.piters>0:
		file_name += 'rho'+opsep+str(args.minrho)+opsep+str(args.maxrho)+sep
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



class rhs_base(torch.nn.Module):
	def __init__(self, args):
		super().__init__()
		self.args = args

		# divergence and Jacobian regularizer
		self.divjacreg  = utils.TraceJacobianReg(n=3)
		self.jacdiagreg = utils.JacDiagReg(n=3, value=-3.0)


	def initialize(self):
		for name, weight in self.rhs.named_parameters():
			if 'weight' in name:
				# torch.nn.init.xavier_normal_(weight, gain=1.e-1)
				# torch.nn.init.xavier_normal_(weight)
				torch.nn.init.xavier_uniform_(weight)
				# torch.nn.init.uniform_(weight,-1.0,1.0)
			else:
				torch.nn.init.zeros_(weight)


	# divergence and jacobian of the vector field
	def divjac(self, t, y):
		return self.divjacreg( self.F(t), y )

	# divergence and jacobian of the vector field
	def jacdiag(self, t, y):
		return self.jacdiagreg( self.F(t), y )


	# derivative of the tangential component in the tangential direction
	def Ftan(self, t, y):
		batch_dim = y.size(0)
		fun   = lambda z: self.F(t)(z).reshape((batch_dim,-1)).pow(2).sum(dim=1, keepdim=True)
		Fnorm = torch.clamp( self.F(t)(y).reshape((batch_dim,-1)).norm(dim=1, keepdim=True), min=1.e-16 )
		return 0.5 * (utils.directional_derivative( fun, y, F ) / Fnorm).sum() / batch_dim


	@property
	def regularizer(self):
		# Total variation like regularizer
		reg = {}
		if self.args.aTV<=0 or len(self.rhs)==1:
			return reg
		for t in range(len(self.rhs)-1):
			for w2, w1 in zip(self.rhs[t+1].parameters(), self.rhs[t].parameters()):
				reg['TV'] = reg.get('TV',0) + ( w2 - w1 ).pow(2).sum()
		reg['TV'] = (self.args.aTV / self.args.T) * reg['TV']
		return reg


	def forward(self, t, y, p=None):
		scales = torch.clamp( self.scales, min=self.args.minrho, max=self.args.maxrho) if self.args.piters>0 and self.args.scales=='learn' else self.scales
		return scales * self.F(t)(y)



class rhs_mlp(rhs_base):
	def __init__(self, data_dim, args, final_activation=None):
		super().__init__(args)

		final_activation = args.sigma if final_activation is None else final_activation

		# dimension of the state space
		dim = data_dim+args.codim

		###############################
		# scales of each dimension
		if args.scales=='learn':
			self.scales = torch.nn.parameter.Parameter(torch.ones((1,dim), dtype=torch.float), requires_grad=True)
			# torch.nn.init.ones_(self.scales)
		elif args.scales=='equal':
			self.scales = 1.0
		elif args.scales=='ortho':
			self.scales = torch.tensor([[0,1]], dtype=torch.float)


		###############################
		F_depth = args.T if args.aTV>=0 else 1

		# structured rhs
		if args.prefix is not None:
			if args.prefix=='par':
				self.rhs = torch.nn.ModuleList( [ layers.ParabolicPerceptron( dim=dim, width=args.width, activation=args.sigma, power_iters=args.piters) for _ in range(F_depth) ] )
			elif args.prefix=='ham':
				self.rhs = torch.nn.ModuleList( [ layers.HamiltonianPerceptron( dim=dim, width=args.width, activation=args.sigma, power_iters=args.piters) for _ in range(F_depth) ] )
			elif args.prefix=='hol':
				self.rhs = torch.nn.ModuleList( [ layers.HollowMLP(dim=dim, width=args.width, depth=args.depth, activation=args.sigma, final_activation=final_activation, power_iters=args.piters) for _ in range(F_depth) ] )
		# unstructured rhs
		else:
			self.rhs = torch.nn.ModuleList( [ layers.MLP(in_dim=dim, out_dim=dim, width=args.width, depth=args.depth, activation=args.sigma, final_activation=final_activation, power_iters=args.piters) for _ in range(F_depth) ] )

		# rhs at a given "time"
		self.F = lambda t: self.rhs[(int(t) if t<args.T else args.T-1) if args.aTV>=0 else 0]


		###############################
		# intialization
		self.initialize()


	def spectrum(self, t, data):
		data = torch.tensor(data).requires_grad_(True)
		batch_dim = data.size(0)
		data_dim  = data.numel() // batch_dim
		jacobian  = utils.jacobian( self.forward(t,data), data, True ).reshape( batch_dim, data_dim, batch_dim, data_dim )
		eigvals = []
		# loop over the data points
		for i in range(batch_dim):
			# eigenvalues at the given point
			eigval = torch.eig(jacobian[i,:,i,:])[0].detach().numpy()
			# sort eigenvalues
			eigvals.append(eigval[np.argsort([ lmb[0]**2+lmb[1]**2 for lmb in eigval ])])
		eigvals = np.concatenate(eigvals)
		return eigvals
		# eigvals = torch.cat([ torch.eig(jacobian[i,:,i,:])[0].detach() for i in range(batch_dim) ])
		# return eigvals.numpy()




class ode_block_base(torch.nn.Module):
	def __init__(self, args):
		super().__init__()
		self.args = args

	def forward(self, y, evolution=False):
		self.ode_out = self.ode(y)
		return self.ode_out if evolution else self.ode_out[-1]

	@property
	def regularizer(self):
		reg  = {}
		rhs  = self.ode.rhs
		name = self.ode.name+'_'
		args = self.args

		p = 0

		# trapezoidal rule
		y0, yT = self.ode_out[0].detach(), self.ode_out[-1].detach()
		if args.adiv>0 or args.ajac>0:
			div0, jac0 = rhs.divjac(0,y0)
			divT, jacT = rhs.divjac(args.T,yT)
			if args.adiv>0:  reg[name+'div'] = 0.5 * ( div0/(args.T+1)**p + divT)
			if args.ajac>0:  reg[name+'jac'] = 0.5 * ( jac0 + jacT )
		if args.ajdiag>0: reg[name+'jdiag'] = 0.5 * ( rhs.jacdiag(0,y0) + rhs.jacdiag(args.T,yT) )
		if args.af>0:     reg[name+'f']     = 0.5 * ( rhs(0,y0).pow(2).sum() + rhs(args.T,yT).pow(2).sum() )
		if args.atan!=0:  reg[name+'tan']   = 0.5 * ( rhs.Ftan(0,y0) + rhs.Ftan(args.T,yT) )
		for t in range(1,args.T):
			y = self.ode_out[t].detach()
			if args.adiv>0 or args.ajac>0:
				divt, jact = rhs.divjac(t,y)
				if args.adiv>0:  reg[name+'div'] = reg[name+'div'] + ((t+1)/(args.T+1))**p * divt #- ((args.atan * rhs.Ftan(t,y)) if args.atan>0 else 0)
				if args.ajac>0:  reg[name+'jac'] = reg[name+'jac'] + jact
			if args.ajdiag>0: reg[name+'jdiag'] = reg[name+'jdiag'] + rhs.jacdiag(t,y)
			if args.af>0:    reg[name+'f']   = reg[name+'f']   + rhs(t,y).pow(2).sum()
			if args.atan!=0: reg[name+'tan'] = reg[name+'tan'] + rhs.Ftan(t,y)

		dim = y0.numel() / y0.size(0)
		if args.af>0:     reg[name+'f']     = (args.af     / args.T / dim)    * reg[name+'f']
		if args.ajac>0:   reg[name+'jac']   = (args.ajac   / args.T / dim**2) * reg[name+'jac']
		if args.adiv>0:   reg[name+'div']   = (args.adiv   / args.T / dim)    * reg[name+'div'] #( reg[name+'div'] + (args.T * args.max_rho if args.power_iters>0 else 0) )
		if args.ajdiag>0: reg[name+'jdiag'] = (args.ajdiag / args.T / dim)    * reg[name+'jdiag']
		if args.atan!=0:  reg[name+'tan']   = (args.atan   / args.T / dim)    * reg[name+'tan']

		if args.aresid>0:
			for step in range(args.T):
				x, y = self.ode_out[step].detach(), self.ode_out[step+1].detach()
				reg[name+'residual'] = reg.get(name+'residual',0) + self.ode.residual(step, [x,y])
			reg[name+'residual'] = (args.aresid / args.T) * reg[name+'residual']

		return reg

	@property
	def statistics(self):
		stat = {}
		rhs  = self.ode.rhs
		name = 'rhs/'+self.ode.name+'_'
		args = self.args

		p = 0

		# trapezoidal rule
		y0, yT = self.ode_out[0].detach(), self.ode_out[-1].detach()
		div0, jac0 = rhs.divjac(0,y0)
		divT, jacT = rhs.divjac(args.T,yT)
		stat[name+'div']   = 0.5 * ( div0/(args.T+1)**p + divT)
		stat[name+'jdiag'] = 0.5 * ( rhs.jacdiag(0,y0) + rhs.jacdiag(args.T,yT) )
		stat[name+'jac']   = 0.5 * ( jac0 + jacT )
		stat[name+'f']     = 0.5 * ( rhs(0,y0).pow(2).sum() + rhs(args.T,yT).pow(2).sum() )
		for t in range(1,args.T):
			y = self.ode_out[t].detach()
			divt, jact = rhs.divjac(t,y)
			stat[name+'div']   = stat[name+'div'] + ((t+1)/(args.T+1))**p * divt
			stat[name+'jdiag'] = stat[name+'jdiag'] + rhs.jacdiag(t,y)
			stat[name+'jac']   = stat[name+'jac'] + jact
			stat[name+'f']     = stat[name+'f']   + rhs(t,y).pow(2).sum()

		dim = y0.numel() / y0.size(0)
		stat[name+'f']     = stat[name+'f'].detach()     / args.T / dim
		stat[name+'jdiag'] = stat[name+'jdiag'].detach() / args.T / dim
		stat[name+'jac']   = stat[name+'jac'].detach()   / args.T / dim**2
		stat[name+'div']   = stat[name+'div'].detach()   / args.T / dim

		return stat












