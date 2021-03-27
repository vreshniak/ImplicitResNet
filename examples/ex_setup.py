import os

import argparse
import yaml
from random import randint

from pathlib import Path
import pickle


import torch




###############################################################################
###############################################################################


_gpu   = torch.device('cuda')
_cpu   = torch.device('cpu')
_dtype = torch.float

_collect_stat = True


###############################################################################
###############################################################################



def parse_args():

	# possible options and their data types
	with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'options.yml'), 'r') as f:
		option_types = yaml.load(f, Loader=yaml.Loader)

	parser = argparse.ArgumentParser()
	for key, val in option_types.items():
		if key in ['eiglim', 'stablim']:
			parser.add_argument('--%s'%(key), type=__builtins__[val], nargs='+' )
		elif val=='bool':
			parser.add_argument('--%s'%(key), action='store_true')
		else:
			parser.add_argument('--%s'%(key), type=__builtins__[val] )

	# create argparse object
	args = parser.parse_args()

	# read options file
	with open('options.yml', 'r') as f:
		opts = yaml.load(f, Loader=yaml.Loader)
		for key, value in opts.items():
			args.__dict__[key] = value

	# override with cmd options
	args1 = parser.parse_args()
	for key, val in args1.__dict__.items():
		if val is not None:
			setattr(args, key, val)

	if args.seed is None:
		args.seed = randint(0,10000)

	if args.T is None:
		args.T = 1
	if args.steps is None:
		assert int(args.T)==args.T
		args.steps = int(args.T)

	# regularizers
	args.alpha = {}
	args.alpha['div']   = args.adiv   if args.adiv   is not None else 0.0
	args.alpha['jac']   = args.ajac   if args.ajac   is not None else 0.0
	args.alpha['f']     = args.af     if args.af     is not None else 0.0
	args.alpha['resid'] = args.aresid if args.aresid is not None else 0.0
	args.alpha['TV']    = args.aTV    if args.aTV    is not None else 0.0
	args.alpha['wdecay']= args.wdecay if args.wdecay is not None else 0.0
	args.alpha['daugm'] = args.adaugm if args.adaugm is not None else 0.0
	args.alpha['perturb'] = args.aperturb if args.aperturb is not None else 0.0
	del args.adiv, args.ajac, args.af, args.aresid, args.aTV, args.wdecay, args.adaugm, args.aperturb


	# save args of each run
	create_paths(args)
	if args.mode == 'train':
		file_name = make_name(args, verbose=False)
		# write options to file
		if os.path.exists(Path('output','name2args')):
			with open(Path('output','name2args'),'rb') as f:
				name2args = pickle.load(f)
			name2args[file_name] = args
		else:
			name2args = {file_name:args}
		if os.path.exists(Path('output','runs2args')):
			with open(Path('output','runs2args'),'rb') as f:
				runs2args = pickle.load(f)
			runs2args.append(args)
		else:
			runs2args = [args]
		with open(Path('output','name2args'),'wb') as f:
			pickle.dump(name2args, f)
		with open(Path('output','runs2args'),'wb') as f:
			pickle.dump(runs2args, f)
	return args


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
	# ignore = ['prefix', 'sigma', 'mode', 'steps', 'eiglim', 'tol', 'ajdiag', 'diaval']
	for arg, value in vars(args).items():
		if value is not None:
			if verbose: print("{0:>{length}}: {1}".format(arg,str(value),length=max_len))
			file_name += arg+opsep+str(value)+sep
	if args.prefix is not None:
		file_name = str(args.prefix) + file_name
	if verbose: print("-------------------------------------------------------------------")
	return args.name if args.name is not None else file_name[:-len(sep)]


# def get_options_from_name(name):
# 	sep   = '|'
# 	opsep = '_'

# 	options = {}

# 	opts = name.split(sep)
# 	for opt in opts:
# 		opts = opt.split(opsep)
# 		if len(opts)==2:
# 			opt_name, opt_val = opts
# 			opt_val = option_type(opt_name)(opt_val)
# 		else:
# 			opt_name = opts[0]
# 			opt_val  = [ option_type(opt_name)(op) for op in opts[1:] ]
# 		options[opt_name] = opt_val
# 		# print(opt_name)
# 		# opt_name, opt_val = opt.split(opsep)
# 		# options[opt_name] = option_type(opt_name)(opt_val)
# 	return options




###############################################################################
###############################################################################



def create_paths(args):
	# subdir = "mlp" if args.prefix is None else args.prefix
	# if args.mode=="init": subdir += "/init"

	Path("checkpoints","initial").mkdir(parents=True, exist_ok=True)
	Path("checkpoints","final").mkdir(parents=True, exist_ok=True)
	Path("output","images").mkdir(parents=True, exist_ok=True)
	Path("output","data").mkdir(parents=True, exist_ok=True)

	paths = {
		'checkpoints': Path('checkpoints'),
		'chkp_init':   Path('checkpoints','initial'),
		'chkp_final':  Path('checkpoints','final'),
		'output':      Path('output'),
		'out_images':  Path("output","images"),
		'out_data':    Path("output","data")
	}
	return paths


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
	print('Mode: ', args.mode)
	print('Load model from: ',load_dir)
	print('\tmissing_keys:    ', missing_keys)
	print('\tunexpected_keys: ', unexpected_keys)
	return mod



###############################################################################
###############################################################################



def theta_stab(z, theta):
	return abs((1+(1-theta)*z)/(1-theta*z))


def plot_stab(theta, xlim=(-5,5), ylim=(-5,5), fname=None):
	import numpy as np
	import matplotlib.pyplot as plt
	# omit zero decimals
	class nf(float):
		def __repr__(self):
			s = f'{self:.2f}'
			if s[-1]+s[-2] == '00':
				return f'{self:.0f}'
			elif s[-1] == '0':
				return f'{self:.1f}'
			else:
				return s
	no_zero = lambda x: 1.e-6 if x==0 else x

	if theta==0.0:
		levels = [0.5, 1, 2, 3, 4, 5, 6, 7]
	elif theta==0.25:
		levels = [0.5, 0.8, 1.0, 1.5, 2, 3, 4, 5, 7, 10, 20]
	elif theta==0.50:
		levels = [0.14, 0.33, 0.5, 0.71, 0.83, 1.0, 1.2, 1.4, 2, 3, 7]
	elif theta==0.75:
		levels = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.3, 2, 5]
	elif theta==1.0:
		levels = [0.15, 0.2, 0.3, 0.5, 1.0, 2, 5]
	else:
		levels = [0.5, 0.8, 1.0, 1.5, 2, 3, 4, 5, 7, 10, 20]

	# make mesh
	xmesh = np.linspace(xlim[0], xlim[1], 200)
	ymesh = np.linspace(ylim[0], ylim[1], 200)
	X,Y   = np.meshgrid(xmesh, ymesh)

	# evaluate stability function on the mesh
	Z = theta_stab(X+1j*Y, theta)

	# plt.axhline(y=0, color='black')
	# plt.axvline(x=0, color='black')
	ax = plt.gca()
	ax.set_aspect('equal', adjustable='box')
	plt.contourf(X,Y,Z, levels=[0,1], colors='0.6')

	cs = plt.contour(X,Y,Z, levels=levels, colors='black')
	cs.levels = [nf(val) for val in cs.levels]
	ax.clabel(cs, cs.levels, inline=True, fmt=r'%r', fontsize=15)

	plt.gca().axes.xaxis.set_visible(False)
	plt.gca().axes.yaxis.set_visible(False)
	plt.gca().axis('off')

	if fname is not None:
		plt.savefig(fname, bbox_inches='tight')



def savefig(name, format='pdf', aspect=None):
	import matplotlib.pyplot as plt
	plt.gca().axes.xaxis.set_visible(False)
	plt.gca().axes.yaxis.set_visible(False)
	plt.gca().axis('off')
	if aspect is not None:
		plt.gca().set_aspect('equal')
	plt.savefig(name+'.%s'%(format), pad_inches=0.0, bbox_inches='tight')



def topk_acc(input, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = input.topk(k=maxk, dim=1, largest=True, sorted=True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].flatten().float().sum(0)
		res.append(float("{:.1f}".format(correct_k.mul_(100.0 / batch_size).detach().numpy())))
	return res