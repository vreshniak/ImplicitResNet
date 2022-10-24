import os

import argparse
import yaml
from random import randint

from pathlib import Path
import pickle

import torch

import numpy as np
import matplotlib.pyplot as plt


from tensorboard.backend.event_processing import event_accumulator


###############################################################################
###############################################################################


_gpu   = torch.device('cuda')
_cpu   = torch.device('cpu')
_dtype = torch.float

_collect_stat = True


###############################################################################
###############################################################################



def parse_args(config='options.yml'):

	# possible options and their data types
	with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'option_types.yml'), 'r') as f:
		option_types = yaml.load(f, Loader=yaml.Loader)

	parser = argparse.ArgumentParser()
	for key, val in option_types.items():
		if val=='bool':
			parser.add_argument(f'--{key}', action='store_true')
		else:
			parser.add_argument(f'--{key}', type=__builtins__[val], nargs='+' )
		# # if 'limits' in key:
		# if key in ['center','radius','stabmin','stabmax']: #'stab' in key or 'eig' in key:
		# 	parser.add_argument('--%s'%(key), type=__builtins__[val], nargs='+' )
		# elif val=='bool':
		# 	parser.add_argument('--%s'%(key), action='store_true')
		# else:
		# 	parser.add_argument('--%s'%(key), type=__builtins__[val] )

	# create argparse object
	args = parser.parse_args()

	# read default options from file and use them only if missing from cmd input
	with open(config, 'r') as f:
		opts = yaml.load(f, Loader=yaml.Loader)
		for key, value in opts.items():
			if args.__dict__[key] is None:
				args.__dict__[key] = value

	# replace single element list options with that element
	for key, val in args.__dict__.items():
		if val is not None and isinstance(val,list) and len(val)==1:
			args.__dict__[key] = val[0]

	if args.seed is None:
		args.seed = randint(0,10000)

	if args.T is None:
		args.T = 1
	if args.steps is None:
		assert int(args.T)==args.T
		args.steps = int(args.T)

	# extra stability options
	# if args.dissipative: args.stablim = [-1.0, 1.0, 1.1]
	# if args.stabzero:    args.stabval = 0.0

	# regularizers
	args.alpha = {}
	reg_keys = [key.split('_')[1] for key in vars(args).keys() if 'a_' in key]
	for key in reg_keys:
		val = vars(args)[f'a_{key}']
		if val is not None:
			args.alpha[key] = val
		elif '^' not in key:
			args.alpha[key] = 0
			# args.alpha[key] = val if val is not None else 0.0
		del vars(args)[f'a_{key}']

	# # regularizers
	# args.alpha = {}
	# args.alpha['div']   = args.adiv   if args.adiv   is not None else 0.0
	# args.alpha['jac']   = args.ajac   if args.ajac   is not None else 0.0
	# args.alpha['f']     = args.af     if args.af     is not None else 0.0
	# args.alpha['rad']   = args.arad   if args.arad   is not None else 0.0
	# args.alpha['cnt']   = args.acnt   if args.acnt   is not None else 0.0
	# args.alpha['tv']    = args.atv    if args.atv    is not None else 0.0
	# args.alpha['wdecay']= args.wdecay if args.wdecay is not None else 0.0
	# # args.alpha['daugm'] = args.adaugm if args.adaugm is not None else 0.0
	# # args.alpha['perturb'] = args.aperturb if args.aperturb is not None else 0.0
	# del args.adiv, args.ajac, args.af, args.arad, args.acnt, args.atv, args.wdecay #, args.adaugm, args.aperturb

	# save args of each run
	paths = create_paths(args.results_path)
	if args.mode == 'train':
		runname = make_name(args, verbose=False)
		# read existing dict with options or create a new one
		if os.path.exists(Path(paths['output'],'name2args')):
			with open(Path(paths['output'],'name2args'),'rb') as f:
				name2args = pickle.load(f)
				name2args[runname] = args
		else:
			name2args = {runname:args}
		# read existing list with options or create a new one
		if os.path.exists(Path(paths['output'],'runs2args')):
			with open(Path(paths['output'],'runs2args'),'rb') as f:
				runs2args = pickle.load(f)
				runs2args.append(args)
		else:
			runs2args = [args]
		# append options to files
		with open(Path(paths['output'],'name2args'),'wb') as f: pickle.dump(name2args, f)
		with open(Path(paths['output'],'runs2args'),'wb') as f: pickle.dump(runs2args, f)
	return args


def make_name(args, verbose=True):
	sep   = '|'
	opsep = '_'
	if verbose: print("\n-------------------------------------------------------------------")
	# file_name = str(args.prefix)+sep if args.prefix is not None and args.prefix!=opsep else ''
	runname = ''
	max_len = 0
	for arg in vars(args):
		length  = len(arg)
		max_len = length if length>max_len else max_len
	max_len += 1
	# ignore = ['prefix', 'sigma', 'mode', 'steps', 'eiglim', 'tol', 'ajdiag', 'diaval']
	for arg, value in vars(args).items():
		if arg=='alpha':
			value = {f'{k}': v for k,v in value.items() if v>0}
			svalue = ','.join([f'{k}_{v}' for k,v in value.items() if v>0])
		elif isinstance(value,list):
			svalue = ','.join([f'{v}' for v in value])
		else:
			svalue = str(value)
		if value is not None:
			if verbose: print("{0:>{length}}: {1}".format(arg,str(value),length=max_len))
			runname += arg+opsep+svalue+sep
	if args.prefix is not None:
		runname = str(args.prefix) + runname
	if verbose: print("-------------------------------------------------------------------")
	return args.runname if args.runname is not None else runname[:-len(sep)]


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



def create_paths(root=None):
	# subdir = "mlp" if args.prefix is None else args.prefix
	# if args.mode=="init": subdir += "/init"

	# if args is not None:
	# 	# root = "." if args.dataset is None else args.dataset+"_results"
	# 	if args.dataset is not None: root   = args.dataset+"_results"
	# 	if args.logdir is not None:  logdir = Path(root,"logs",args.logdir).mkdir(parents=True, exist_ok=True)
	# else:
	# 	logdir = Path(root,"logs").mkdir(parents=True, exist_ok=True)

	# if args.dataset is None:
	# 	if args.logdir is not None:
	# 		Path("logs", args.logdir).mkdir(parents=True, exist_ok=True)
	# 	else:
	# 		Path("logs").mkdir(parents=True, exist_ok=True)
	# 	Path("checkpoints","initial").mkdir(parents=True, exist_ok=True)
	# 	Path("checkpoints","final").mkdir(parents=True, exist_ok=True)
	# 	Path("output","images").mkdir(parents=True, exist_ok=True)
	# 	Path("output","data").mkdir(parents=True, exist_ok=True)

	# 	paths = {
	# 		'logs':        Path('logs') if args.logdir is None else Path('logs', args.logdir),
	# 		'checkpoints': Path('checkpoints'),
	# 		'chkp_init':   Path('checkpoints','initial'),
	# 		'chkp_final':  Path('checkpoints','final'),
	# 		'output':      Path('output'),
	# 		'out_images':  Path("output","images"),
	# 		'out_data':    Path("output","data")
	# 	}
	# else:
	# 	root = args.dataset+"_results"

	# if args.logdir is not None:
	# 	Path(root,"logs", args.logdir).mkdir(parents=True, exist_ok=True)
	# else:
	# 	Path(root,"logs").mkdir(parents=True, exist_ok=True)

	if root is None: root = 'results'
	Path(root,"checkpoints","initial").mkdir(parents=True, exist_ok=True)
	Path(root,"checkpoints","final").mkdir(parents=True, exist_ok=True)
	Path(root,"output","images").mkdir(parents=True, exist_ok=True)
	Path(root,"output","data").mkdir(parents=True, exist_ok=True)
	Path(root,"logs").mkdir(parents=True, exist_ok=True)

	paths = {
		# 'logs':        Path(root,'logs') if args.logdir is None else Path(root,'logs',args.logdir),
		'root':        Path(root),
		# 'logs':        logdir,
		'logs':        Path(root,'logs'),
		'checkpoints': Path(root,'checkpoints'),
		'chkp_init':   Path(root,'checkpoints','initial'),
		'chkp_final':  Path(root,'checkpoints','final'),
		'output':      Path(root,'output'),
		'out_images':  Path(root,'output','images'),
		'out_data':    Path(root,'output','data')
	}
	return paths


def load_model(model, args, device=_cpu, location=None):
	mod = model.to(device=device)

	if location is not None:
		load_dir = location
	else:
		paths   = create_paths(args.results_path)
		runname = make_name(args, verbose=False)

		# if args.mode=='init':
		# 	return mod
		if args.mode=='train':
			if args.init=='rnd':
				return mod
			# elif args.init=="init":
			# 	load_dir = Path(paths['initialization'],'%4.2f'%(args.theta))
				# import re
				# load_dir = Path( checkpoint_dir, re.sub('theta_\d*.\d*','theta_'+str(args.theta),file_name) )
			# elif args.init=="cont":
			# 	load_dir = Path(paths['chkp_final'], file_name)
		# if args.mode in ['plot','test','process','visualize']:
		else:
			load_dir = Path(paths['chkp_final'], runname)

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
		if args.mode!='train':
			mod.eval()
		print('Mode: ', args.mode)
		print('Load model from: ',load_dir)
		print('\tmissing_keys:    ', missing_keys)
		print('\tunexpected_keys: ', unexpected_keys)
		return mod



###############################################################################
###############################################################################



def stab_hist(model_spectrum, theta):
	histvals, histbins, patches = plt.hist(theta_stability_fun(m.theta.item(), model_rhs_spectrum[-1][:,0]), density=True, bins=20)
	# histvals, histbins, patches = plt.hist(ex_setup.theta_stab(rhs_spectrum[-1][:,0]+1j*rhs_spectrum[-1][:,1], m.theta.item()), density=True, bins=20)
	histogram = np.hstack( ((0.5*(histbins[:-1]+histbins[1:])).reshape((-1,1)), histvals.reshape((-1,1))) )
	np.savetxt( Path(paths['out_data'],'hist','hist_stab_%s_%s.txt'%(args.runname,m.name)), histogram, fmt='%.2e', delimiter=',')

	histvals, histbins, patches  = plt.hist( model_rhs_spectrum[-1][:,0], density=True, bins=20 )
	histogram = np.hstack( ((0.5*(histbins[:-1]+histbins[1:])).reshape((-1,1)), histvals.reshape((-1,1))) )
	np.savetxt( Path(paths['out_data'],'hist','hist_eig_%s_%s.txt'%(args.runname,m.name)), histogram, fmt='%.2e', delimiter=',')


def scatter(data, **kwarg):
	from matplotlib.colors import to_rgb, to_rgba
	from scipy.stats import gaussian_kde

	values = np.array(data).T.squeeze()
	# avoid singular matrix
	values += 1.e-12 * np.random.randn(*values.shape)
	#if values[0].var()<=1.e-6 and values[1].var()<=1.e-6: values = values[:,0]
	pdf = gaussian_kde(values)(values)
	pdf /= np.amax(pdf)

	r1, g1, b1 = to_rgb('blue')
	r2, g2, b2 = to_rgb('red')
	ind = np.argsort(pdf)
	pdf = pdf[ind]
	r = (1-pdf)*r1 + pdf*r2
	g = (1-pdf)*g1 + pdf*g2
	b = (1-pdf)*b1 + pdf*b2
	# color = [(rr,gg,bb,alpha if alpha>0.0 else 0) for rr,gg,bb,alpha in zip(r,g,b,pdf)]
	color = [(rr,gg,bb,1) for rr,gg,bb,alpha in zip(r,g,b,pdf)]
	plt.scatter(data[:,0][ind], data[:,1][ind], c=color, **kwarg)


def eval_model_spectrum(model, data, batch=1, approximate=True, eigs_per_batch=None, tol=1.e-5):
	from implicitresnet.utils.spectral import eigenvalues, spectralnorm
	from implicitresnet import theta_solver
	from shapely.geometry import Point, MultiPolygon, Polygon
	from shapely.ops import unary_union
	from descartes import PolygonPatch

	if not isinstance(model,torch.nn.Sequential) and not isinstance(model,list):
		model = [model]

	batch_dim = data.size(0)

	# all_rhs_spectrum   = []
	# model_spectralnrm  = []
	model_rhs_spectrum      = {}
	model_rhs_gershcircles  = {}

	x = data
	for i, m in enumerate(model):
		if not isinstance(m, theta_solver) or m(x).shape!=x.shape:
			print(f"Can't evaluate spectrum for layer {m.name}, skipping")
			x = m(x)
			continue
		print(f'Evaluating spectrum for layer {m.name}')
		if isinstance(m, theta_solver):
			theta  = m.theta.item()
			odesol = m.trajectory(x)
			m_rhs_spectrum     = []
			m_rhs_gershcircles = []
			for t in range(m.num_steps):
				y_theta_t = (1-theta)*odesol[:,t,...] + theta*odesol[:,t+1,...]
				for b in np.arange(0,batch_dim,batch):
					y_theta_t_batch = y_theta_t[b:min(b+batch,batch_dim)]
					# evaluate eigenvalues
					eigs, jac = eigenvalues( lambda x: m.rhs(t,x), y_theta_t_batch, return_jacobian=True, approximate=approximate, eigs_per_batch=eigs_per_batch, tol=tol )
					m_rhs_spectrum.append(eigs)
					# find Gershgorin circles
					for ii in range(jac.shape[0]):
						centers  = np.diag(jac[ii,:,ii,:]).ravel()
						radiuses = np.sum(np.abs(jac[ii,:,ii,:]), axis=1) - np.abs(centers)
						m_rhs_gershcircles.append([[c,r] for c,r in zip(centers,radiuses)])
				# model_spectralnrm.append(np.amax(theta_stab(m_rhs_spectrum[-1][:,0]+1j*m_rhs_spectrum[-1][:,1], theta)))
		else:
			m_rhs_spectrum     = []
			m_rhs_gershcircles = []

			for b in np.arange(0,batch_dim,batch):
				x_batch = x[b:min(b+batch,batch_dim),...]
				# evaluate eigenvalues
				eigs, jac = eigenvalues( lambda x: m(x), x_batch, return_jacobian=True, eigs_per_batch=eigs_per_batch, tol=tol )
				m_rhs_spectrum.append(eigs)
				# find Gershgorin circles
				for ii in range(jac.shape[0]):
					centers  = np.diag(jac[ii,:,ii,:]).ravel()
					radiuses = np.sum(np.abs(jac[ii,:,ii,:]), axis=1) - np.abs(centers)
					m_rhs_gershcircles.append([[c,r] for c,r in zip(centers,radiuses)])
			# model_spectralnrm.append(np.amax(theta_stab(m_rhs_spectrum[-1][:,0]+1j*m_rhs_spectrum[-1][:,1], theta)))

		# find union of Gershgorin circles
		m_rhs_gershcircles = np.concatenate(np.array(m_rhs_gershcircles))
		gershgorin_union = unary_union([Point(circle[0], 0).buffer(circle[1]) for circle in m_rhs_gershcircles])
		gershgorin_patches = []
		if isinstance(gershgorin_union, MultiPolygon):
			for polygon in gershgorin_union.geoms:
				gershgorin_patches.append( PolygonPatch(polygon, fc='none', ec='none', alpha=1.0, zorder=2) )
		elif isinstance(gershgorin_union, Polygon):
			gershgorin_patches.append( PolygonPatch(gershgorin_union, fc='none', ec='none', alpha=1.0, zorder=2) )

		# collect all spectral properties in a single dict
		#a, b = m.rhs.get_spectral_spread()
		c, r = m.rhs[0].get_spectral_circle()
		model_rhs_spectrum[m.name] = {
			'eigenvalues': np.concatenate(np.array(m_rhs_spectrum)),
			'gershgorin_circles': gershgorin_patches,
			#'gershgorin_circles': m_rhs_gershcircles,
			#'spectral_limits': [a.item(), b.item()],
			'spectral_circle': [c.item(), r.item()],
			}
		# model_rhs_spectrum[m.name]     = {'spectrum': np.concatenate(np.array(m_rhs_spectrum))}
		# model_rhs_gershcircles[m.name] = gershgorin_patches
		# all_rhs_spectrum.append(model_rhs_spectrum[m.name])
		# model_rhs_gershcircles[m.name] = np.concatenate(np.array(m_rhs_gershcircles))
		# else:
			#model_spectralnrm.append(spectralnorm(m,x).mean().item())
		x = m(x)
	# spectrum of all layers in a model as a single array
	# model_rhs_spectrum['all'] = np.concatenate(all_rhs_spectrum)
	return model_rhs_spectrum #, model_rhs_gershcircles #, model_spectralnrm


#######################################
# complex stability function
def theta_stab(z, theta):
	return abs((1+(1-theta)*z)/(1-theta*z))

# signed real stability function and its inverse
def theta_stability_fun(theta, x):
	return (1+(1-theta)*x) / (1-theta*x)

def theta_inv_stability_fun(theta, y):
	y = max(y, 1 - 1.0/(theta+1.e-12) + 1.e-6)
	return (1-y) / ((1-y)*theta-1)
#######################################


def plot_stab(theta, levels=None, xlim=(-5,5), ylim=(-5,5), fname=None):
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

	# levels = [0.0,0.05,0.1,0.25,0.5,0.75,1.0]
	if levels is None:
		levels = [0.2*i for i in range(6)]

	# if theta==0.0:
	# 	levels = [0.5, 1, 2, 3, 4, 5, 6, 7]
	# elif abs(theta-0.10)<=1.e-5:
	# 	levels = [0.5, 1.0, 2, 3, 4, 5, 6, 7]
	# elif abs(theta-0.20)<=1.e-5:
	# 	levels = [0.4, 0.75, 1.0, 1.5, 2, 3, 4, 5, 7, 10, 20]
	# elif abs(theta-0.25)<=1.e-5:
	# 	levels = [0.5, 0.8, 1.0, 1.25, 1.5, 2, 3, 4, 5, 7, 10, 20]
	# elif abs(theta-0.30)<=1.e-5:
	# 	levels = [0.5, 0.8, 1.0, 1.5, 2, 3, 4, 5, 7, 10, 20]
	# elif abs(theta-0.40)<=1.e-5:
	# 	levels = [0.3, 0.5, 0.8, 1.0, 1.5, 2, 3, 5, 10]
	# elif abs(theta-0.50)<=1.e-5:
	# 	levels = [0.14, 0.33, 0.5, 0.71, 0.83, 1.0, 1.2, 1.4, 2, 3, 7]
	# elif theta==0.75:
	# 	levels = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.3, 2, 5]
	# elif theta==1.0:
	# 	levels = [0.15, 0.2, 0.3, 0.5, 1.0, 2, 5]
	# else:
	# 	levels = [0.5, 0.8, 1.0, 1.5, 2, 3, 4, 5, 7, 10, 20]

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
	plt.contourf(X,Y,Z, levels=[0,1], colors='0.7')

	cs = plt.contour(X,Y,Z, levels=levels, colors='black')
	cs.levels = [nf(val) for val in cs.levels]
	ax.clabel(cs, cs.levels, inline=True, fmt=r'%r', fontsize=15)

	plt.gca().axes.xaxis.set_visible(False)
	plt.gca().axes.yaxis.set_visible(False)
	plt.gca().axis('off')

	if fname is not None:
		plt.savefig(fname, bbox_inches='tight')



def plot_spectrum(spectrum, theta, save_path=None, xlim=(-5,5), ylim=(-5,5), density=True, plot_stability=True, plot_circle=True, stab_levels=None): #, only_single_plot=True):
	# all_rhs_spectrum = []
	# for key, value in model_spectrum.items():
	# 	all_rhs_spectrum.append(value)
	# 	if not only_single_plot:
	# 		plt.gca().clear()
	# 		# plot_stab(theta, xlim=xlim, ylim=ylim)
	# 		if density:
	# 			# scatter(model_rhs_spectrum[-1][:,0], model_rhs_spectrum[-1][:,1], kernel, marker='o', s=30) #, markerfacecolor='none')
	# 			scatter(value, marker='o', s=30)
	# 		else:
	# 			plt.plot(value[:,0], value[:,1], 'o', markersize=2) #, markerfacecolor='none')
	# 		plt.xlim(xlim)
	# 		plt.ylim(ylim)
	# 		if save_path is not None:
	# 			plt.savefig(save_path+"_%s_spectrum.jpg"%(key), bbox_inches='tight', pad_inches=0.0, dpi=100)

	eigenvalues = spectrum['eigenvalues']
	circle      = spectrum['spectral_circle']

	# all eigenvalues in one plot
	# all_rhs_spectrum = np.concatenate(all_rhs_spectrum)
	# plt.gca().clear()

	# plot stability region
	if plot_stability:
		plot_stab(theta, xlim=xlim, ylim=ylim, levels=stab_levels)

	# plot eigenvalues
	if density:
		# scatter(all_rhs_spectrum[:,0], all_rhs_spectrum[:,1], kernel, marker='o', s=30) #, markerfacecolor='none')
		#scatter(spectrum, marker='o', s=30)
		scatter(np.hstack((np.real(eigenvalues).reshape(-1,1),np.imag(eigenvalues).reshape(-1,1))), marker='o', s=30)
	else:
		#plt.plot(spectrum[:,0], spectrum[:,1], 'o', markersize=3) #, markerfacecolor='none')
		plt.plot(np.real(eigenvalues), np.imag(eigenvalues), 'o', markersize=3) #, markerfacecolor='none')

	# plot spectral circle
	if plot_circle:
		patch = plt.Circle((circle[0], 0), circle[1], fill=False)
		patch.set_ec('b')
		patch.set_lw(1.5)
		# patch.set_ls(line_styles[i])
		plt.gca().add_patch(patch)

	# save
	plt.gca().set_aspect('equal', adjustable='box')
	plt.gca().axes.xaxis.set_visible(False)
	plt.gca().axes.yaxis.set_visible(False)
	plt.gca().axis('off')
	plt.xlim(xlim)
	plt.ylim(ylim)
	if save_path is not None:
		plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=100)
		# plt.savefig(save_path+"_spectrum.jpg", bbox_inches='tight', pad_inches=0.0, dpi=100)

	# nonzero = 100 * np.count_nonzero(all_rhs_spectrum[:,0]+all_rhs_spectrum[:,1]) / all_rhs_spectrum.shape[0]
	# large   = 100 * np.count_nonzero( np.sqrt(all_rhs_spectrum[:,0]**2+all_rhs_spectrum[:,1]**2) * (np.sqrt(all_rhs_spectrum[:,0]**2+all_rhs_spectrum[:,1]**2)>1.e-3) ) / all_rhs_spectrum.shape[0]
	#print(args.theta, nonzero, large)

	# # save spectral norm of layers
	# if spectral_norm:
	# 	return spectralnrm
	# 	fname  = Path(paths['out_data'],('spectral_norm.txt'))
	# 	header = 'layer-1,'+','.join(['layer-%d'%(i) for i in range(2,len(spectralnrm))]) if not os.path.exists(fname) else ''
	# 	with open(fname, 'a') as f:
	# 		np.savetxt( f, np.array(spectralnrm).reshape((1,-1)), fmt='%.2e', delimiter=',', header=header)


###############################################################################
###############################################################################


# https://stackoverflow.com/questions/52756152/tensorboard-extract-scalar-by-a-script
def load_scalar_logs(path):
	path = str(path)
	event_acc = event_accumulator.EventAccumulator(path)
	event_acc.Reload()
	data = {}

	for tag in sorted(event_acc.Tags()["scalars"]):
		# print(tag)
		x, y = [], []

		for scalar_event in event_acc.Scalars(tag):
			x.append(scalar_event.step)
			y.append(scalar_event.value)

		data[tag] = np.hstack( (np.asarray(x).reshape(-1,1), np.asarray(y).reshape(-1,1)) )
	return data



###############################################################################
###############################################################################



def pickle_data(name, data, path):
	if os.path.exists(path):
		with open(path,'rb') as f:
			name2data = pickle.load(f)
		name2data[name] = data
	else:
		name2data = {name: data}
	with open(path,'wb') as f:
		pickle.dump(name2data, f)


def save_data_as_txt(name, data, path):
	if os.path.exists(path):
		with open(path,'rb') as f:
			header = f.readline()
			name2data = np.loadtxt(f, skiprows=1)
		header = "%s %s"%(header,name)
		name2data = np.c_[name2data, data]
	else:
		header = name
		name2data = np.array(data)
		name2data = np.reshape(name2data, (name2data.size(),1))
	np.savetxt( path, name2data, delimiter=' ', fmt='%0.2f', header=header )


def savefig(path, ax=None, format=None, aspect=None, dpi=None):
	import matplotlib.pyplot as plt
	if ax is None:
		ax = plt.gca()
	ax.axes.xaxis.set_visible(False)
	ax.axes.yaxis.set_visible(False)
	ax.axis('off')
	if aspect is not None:
		ax.set_aspect(aspect)
	if format is not None:
		plt.savefig(str(path)+'.%s'%(format), pad_inches=0.0, bbox_inches='tight', dpi=dpi)
	else:
		plt.savefig(path, pad_inches=0.0, bbox_inches='tight', dpi=dpi)



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
		res.append(float("{:.2f}".format(correct_k.mul_(100.0 / batch_size).detach().numpy())))
	return res