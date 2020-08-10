import sys
import os
import time
import argparse
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from jacobian import JacobianReg

import custom_layers as layers
import custom_utils  as utils



if __name__ == '__main__':
	#########################################################################################
	#########################################################################################


	parser = argparse.ArgumentParser()

	####################
	parser.add_argument("--prefix",      default=None )
	parser.add_argument("--mode",        type=str,   default="train", choices=["initialize", "train", "plot", "test"] )
	parser.add_argument("--seed",        type=int,   default=np.random.randint(0,10000))
	####################
	parser.add_argument("--theta",       type=float, default=0.0 )
	parser.add_argument("--T",           type=float, default=1   )
	parser.add_argument("--steps",       type=int,   default=10  )
	parser.add_argument("--nodes",       type=int,   default=2   )
	parser.add_argument("--codim",       type=int,   default=1   )
	####################
	parser.add_argument("--init",        type=str,   default="warm_up", choices=["warmup", "chkp", "rnd"])
	parser.add_argument("--epochs",      type=int,   default=1000)
	parser.add_argument("--lr",          type=float, default=0.01)
	parser.add_argument("--datasize",    type=int,   default=500 )
	parser.add_argument("--batch",       type=int,   default=-1  )
	####################
	parser.add_argument("--power_iters", type=int,   default=0   )
	parser.add_argument("--w_decay",     type=float, default=0   )
	####################
	parser.add_argument("--alpha_TV",    type=float, default=0   )
	parser.add_argument("--alpha_fpdiv", type=float, default=0   )
	parser.add_argument("--alpha_model", type=float, default=0   )
	parser.add_argument("--alpha_rhsjac",type=float, default=0   )
	parser.add_argument("--alpha_rhsdiv",type=float, default=0   )
	args = parser.parse_args()

	print("\n-------------------------------------------------------------------")
	file_name = str(args.prefix)+"__" if args.prefix is not None and args.prefix!="_" else ''
	max_len = 0
	for arg in vars(args):
		length  = len(arg)
		max_len = length if length>max_len else max_len
	max_len += 1
	noise_args = ""
	for arg,value in vars(args).items():
		if value is not None:
			print("{0:>{length}}: {1}".format(arg,str(value),length=max_len))
			if arg!='prefix' and arg!='mode' and arg!='robust' and arg!='init' and "noise" not in arg:
				file_name += arg+"_"+str(value)+"__"
	print("-------------------------------------------------------------------")

	file_name  = file_name[:-2]
	# if args.method!='theta':
	# 	file_name = args.method+'__T_'+str(args.T)+'__steps_'+str(args.steps)
	script_name = sys.argv[0][:-3]



	#########################################################################################
	#########################################################################################


	gpu = torch.device('cuda')
	cpu = torch.device('cpu')
	_device = cpu #if torch.cuda.is_available() else cpu
	_dtype  = torch.float


	#########################################################################################
	#########################################################################################
	# Data
	np.random.seed(args.seed)

	fun  = lambda x: np.sin(x) + 0.1*np.cos(19.0*x)
	# phi  = lambda x: torch.cat( [x for _ in range(args.codim)], 1 )
	phi  = lambda x: torch.zeros((x.size()[0],args.codim))


	# training data
	ntrain = args.datasize
	xtrain = np.linspace(-5, 5, ntrain).reshape((ntrain,1))
	# xtrain = np.vstack( [np.linspace(-5, -1, ntrain//2).reshape((-1,1)), np.linspace(1, 5, ntrain//2).reshape((-1,1))] )
	ytrain = fun(xtrain)
	# xtrain = np.vstack([xtrain + 0.1*(2*np.random.rand(xtrain.shape[0],1)-1) for _ in range(5)])
	# ytrain = np.vstack([ytrain for _ in range(5)])


	# validation data
	nval = 2*ntrain
	xval = np.linspace(-5, 5, nval).reshape((nval,1))
	yval = fun(xval)

	# training and validation datasets
	dataset     = torch.utils.data.TensorDataset( torch.from_numpy(xtrain).to(_dtype), torch.from_numpy(ytrain).to(_dtype) )
	val_dataset = torch.utils.data.TensorDataset( torch.from_numpy(xval).to(_dtype),   torch.from_numpy(yval).to(_dtype)   )


	#########################################################################################
	#########################################################################################
	# NN parameters

	loss_fn         = torch.nn.MSELoss(reduction='mean')
	optimizer_Adam  = lambda model, lr: torch.optim.Adam(model.parameters(),    lr=lr, weight_decay=args.w_decay)
	optimizer_RMS   = lambda model, lr: torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=args.w_decay)
	optimizer_SGD   = lambda model, lr: torch.optim.SGD(model.parameters(),     lr=lr, weight_decay=args.w_decay, momentum=0.5)
	optimizer_LBFGS = lambda model, lr: torch.optim.LBFGS(model.parameters(),   lr=1., max_iter=100, max_eval=None, tolerance_grad=1.e-9, tolerance_change=1.e-9, history_size=100, line_search_fn='strong_wolfe')
	scheduler_fn    = lambda optimizer: torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=100, verbose=True, threshold=1.e-6, threshold_mode='rel', cooldown=200, min_lr=1.e-6, eps=1.e-8)
	# scheduler_fn    = lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.7, last_epoch=-1)


	#########################################################################################
	#########################################################################################
	# NN model


	########################################################
	# augment original data with additional dimension
	class augment(torch.nn.Module):
		def __init__(self):
			super(augment, self).__init__()
			# self.linear = torch.nn.Linear(1, args.codim+1, bias=False)
			# if args.power_iters>0:
			# 	self.linear = torch.nn.utils.spectral_norm( self.linear, name='weight', n_power_iterations=args.power_iters, eps=1e-12, dim=None)

		def forward(self, x):
			# return self.linear(x)
			return torch.cat( (x, phi(x)), 1 )
	########################################################



	########################################################
	# last component as the function value
	class output(torch.nn.Module):
		def __init__(self):
			super(output, self).__init__()
			# self.linear = torch.nn.Linear(args.codim+1, 1, bias=False)
			# if args.power_iters>0:
			# 	self.linear = torch.nn.utils.spectral_norm( self.linear, name='weight', n_power_iterations=args.power_iters, eps=1e-12, dim=None)

		def forward(self, x):
			# return self.linear(x)
			return x[:,args.codim:]
	########################################################



	########################################################
	class rhs(torch.nn.Module):
		def __init__(self):
			super(rhs, self).__init__()

			activation = torch.nn.ReLU()
			# activation = torch.nn.GELU()
			# activation = torch.nn.Tanhshrink()
			# activation = torch.nn.CELU()
			# activation = torch.nn.Tanh()
			# activation = torch.nn.Softsign()


			# learnable scales of each dimension
			self.scales = torch.nn.parameter.Parameter(torch.tensor(np.ones((1,args.codim+1)), dtype=_dtype), requires_grad=True)
			# self.scales = torch.tensor([[1,1]], dtype=_dtype)

			self.F = []
			for _ in range(args.steps+1):
				# linear layers
				linear_in  =   torch.nn.Linear(args.codim+1, args.nodes,   bias=False)
				linear_mlp = [ torch.nn.Linear(args.nodes,   args.nodes,   bias=True) for _ in range(4) ]
				linear_out =   torch.nn.Linear(args.nodes,   args.codim+1, bias=False)

				# spectral normalization
				if args.power_iters>0:
					linear_in  =   torch.nn.utils.spectral_norm( linear_in,  name='weight', n_power_iterations=args.power_iters, eps=1e-12, dim=None)
					linear_mlp = [ torch.nn.utils.spectral_norm( linear,     name='weight', n_power_iterations=args.power_iters, eps=1e-12, dim=None) for linear in linear_mlp ]
					linear_out =   torch.nn.utils.spectral_norm( linear_out, name='weight', n_power_iterations=args.power_iters, eps=1e-12, dim=None)

				# Multilayer perceptron
				mlp = [val for pair in zip(linear_mlp, [activation]*len(linear_mlp)) for val in pair]
				# mlp[-1][1] = torch.nn.Softsign()

				# rhs
				self.F.append( torch.nn.Sequential(
					linear_in,
					activation,
					*mlp,
					linear_out
					)
				)
			self.F = torch.nn.ModuleList(self.F)

			# intialization
			for name, weight in self.F.named_parameters():
				if 'weight' in name:
					# torch.nn.init.xavier_normal_(weight, gain=1.e-3)
					# torch.nn.init.xavier_normal_(weight, gain=1.e-2)
					# torch.nn.init.xavier_normal_(weight, gain=1.e-1)
					# torch.nn.init.xavier_normal_(weight)
					torch.nn.init.uniform_(weight, -1, 1)
				else:
					torch.nn.init.zeros_(weight)

		def forward(self, t, y, p=None):
			return self.scales * self.F[t](y)

		@property
		def regularizer(self):
			regularizer = {}
			if args.alpha_TV==0 or args.steps==1:
				return regularizer
			for t in range(len(self.F)-1):
				for w2, w1 in zip(self.F[t+1].parameters(), self.F[t].parameters()):
					regularizer['TV'] = regularizer.get('TV',0) + args.alpha_TV * ( w2 - w1 ).pow(2).sum()
			return regularizer

		def spectrum(self, t, data):
			data = torch.tensor(data).requires_grad_(True)
			jac  = utils.jacobian( self.forward(t,data), data, True ).reshape( data.shape[0], data.numel()//data.shape[0], data.shape[0], data.numel()//data.shape[0] )
			eigvals = torch.cat([ torch.eig(jac[i,:,i,:])[0].detach() for i in range(data.shape[0]) ])
			return eigvals.numpy()

	########################################################



	########################################################
	class ode_block(torch.nn.Module):
		def __init__(self):
			super(ode_block, self).__init__()
			self.ode = layers.ode_solver( rhs(), args.T, args.steps, args.theta, solver='cg', alpha_rhsjac=args.alpha_rhsjac, alpha_rhsdiv=args.alpha_rhsdiv, alpha_fpdiv=args.alpha_fpdiv )

		def forward(self, y):
			# return self.ode(y)
			return self.ode(y)[-1]

		def evolution(self, y):
			return self.ode(y)
	########################################################



	########################################################
	class model(torch.nn.Module):
		def __init__(self):
			super(model, self).__init__()

			self.regularizers = {}
			self.model = torch.nn.Sequential( augment(), ode_block(), output() )
			self.reg   = JacobianReg()

		def forward(self, x):
			x.requires_grad = True
			out = self.model(x)
			if self.training and args.alpha_model>0:
				# jacobian = torch.autograd.functional.jacobian(lambda z: self.model(z), x, create_graph=True, strict=True)
				# jacobian_loss = jacobian.pow(2).sum()
				# jacobian_loss = torch.trace((jacobian.reshape((jacobian.size()[0],jacobian.size()[0]))+0.0).pow(2))
				# self.regularizers['model_Jac'] = args.alpha_model / args.datasize * jacobian_loss
				self.regularizers['model_Jac'] = args.alpha_model * self.reg(x, out)
				# self.regularizers['model_Jac'] = args.alpha_model / args.datasize *  utils.jacobian(out,x,create_graph=True).pow(2).sum()
			return out

		@property
		def regularizer(self):
			return {} if args.alpha_model==0 else self.regularizers
	########################################################


	def get_model(seed=None):
		if seed is not None: torch.manual_seed(seed)
		return model().to(device=_device)


	#########################################################################################
	torch.autograd.set_detect_anomaly(True)

	if args.mode=="initialize":
		# create directories for the checkpoints and logs
		chkp_dir = "initialization"
		Path(chkp_dir).mkdir(parents=True, exist_ok=True)
		writer = SummaryWriter(chkp_dir)

		model       = get_model(args.seed)
		optimizer   = optimizer_SGD(model, args.lr)
		scheduler   = None
		regularizer = None

		train_obj = utils.training_loop(model, dataset, val_dataset, args.batch, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn, accuracy_fn=accuracy_fn, regularizer=regularizer,
			writer=writer, write_hist=True, history=False, checkpoint=None)

		# initialize with analytic continuation
		for theta in np.linspace(0,1,101):
			model.apply(lambda m: setattr(m,'theta',theta))
			train_obj(args.epochs)
			torch.save( model.state_dict(), Path(chkp_dir,"%4.2f"%(theta)) )

		writer.close()

	elif args.mode=="train":
		logdir = Path("logs",file_name)
		writer = SummaryWriter(logdir)

		losses = []
		for sim in range(1):
			try:
				model       = get_model(args.seed+sim)
				optimizer   = optimizer_Adam(model, args.lr)
				# scheduler   = None
				scheduler   = scheduler_fn(optimizer)
				# regularizer = regularizer_fn(model, rhs_obj)
				regularizer = None

				# lr_schedule = np.linspace(args.lr, args.lr/100, args.epochs)
				# checkpoint={'epochs':1000, 'name':"models/"+script_name+"/sim_"+str(sim)+'_'+file_name[:]}
				train_obj = utils.training_loop(model, dataset, val_dataset, args.batch, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn, regularizer=regularizer,
					writer=writer, write_hist=True, history=False, checkpoint=None)

				losses.append(train_obj(args.epochs))
			except:
				raise
			finally:
				Path("checkpoints/").mkdir(parents=True, exist_ok=True)
				torch.save( model.state_dict(), Path("checkpoints",file_name[:]) )

		writer.close()

	elif args.mode=="plot":
		# from skimage.util import montage
		fig_no = 0


		###############################################
		# prepare data
		model = get_model(args.seed)
		missing_keys, unexpected_keys = model.load_state_dict(torch.load(Path("checkpoints",file_name[:]), map_location=cpu))
		model.eval()

		# plot data
		nplot = 1000
		xplot = np.linspace(-5, 5, nplot).reshape((nplot,1))
		ytrue = fun(xplot)

		# propagation of input through layers
		ypred0 = model.model[0](torch.from_numpy(xplot).float()).detach().numpy()
		ypred1 = model.model[1].evolution(torch.from_numpy(ypred0).float()).detach().numpy()
		ypred2 = model.model[2](torch.from_numpy(ypred1[-1]).float()).detach().numpy()
		# ypred2 = model.model[2](torch.from_numpy(ypred1).float()).detach().numpy()

		ytrain0 = model.model[0](torch.from_numpy(xtrain).float()).detach().numpy()
		ytrain1 = model.model[1].evolution(torch.from_numpy(ytrain0).float()).detach().numpy()
		ytrain2 = model.model[2](torch.from_numpy(ytrain1[-1]).float()).detach().numpy()


		###############################################
		# plot function
		fig = plt.figure(fig_no); fig_no += 1

		# plt.plot(xplot,ytrue)
		plt.plot(xplot,ypred2)
		plt.plot(xtrain,ytrain,'o')
		# plt.show()


		###############################################
		# plot spectrum
		fig = plt.figure(fig_no); fig_no += 1

		def plot_stab(theta, fig_no=1, levels=20, xlim=(-5,5), ylim=(-5,5), fname=None):
			theta_stab = lambda z, theta: (1+(1-theta)*z)/(1-theta*z)
			class nf(float):
			    def __repr__(self):
			        s = f'{self:.2f}'
			        return f'{self:.0f}' if s[-1]+s[-2] == '0' else s
			def no_zero(x):
				if x==0:
					return 1.e-6
				else:
					return x
			X,Y = np.meshgrid(np.linspace(xlim[0],xlim[1],200),np.linspace(ylim[0],ylim[1],200))
			Z = abs(theta_stab(X+1j*Y, theta))
			z_levels = np.unique(np.hstack((np.linspace(xlim[0],0,5)+1j*0,np.linspace(0,xlim[1],5)+1j*0)))
			zz = abs(theta_stab(z_levels, theta))
			levels = np.logspace(np.log10(no_zero(np.amin(zz))), np.log10(no_zero(np.amax(zz))), levels)
			levels = np.unique(np.sort(levels))

			plt.figure(fig_no); fig_no += 1
			# plt.axhline(y=0, color='black')
			# plt.axvline(x=0, color='black')
			ax = plt.gca()
			ax.set_aspect('equal', adjustable='box')
			plt.contourf(X,Y,Z, levels=[0,1], colors='0.6',   linewidths=1)
			# plt.contour(X,Y,Z,  levels=[1],   colors='black', linewidths=1)
			cs = plt.contour(X,Y,Z, levels=levels, colors='black')
			cs.levels = [nf(val) for val in cs.levels]
			ax.clabel(cs, cs.levels, inline=True, fmt=r'%r', fontsize=10)
			# plt.hlines(0,xlim[0],xlim[1])
			# plt.vlines(0,ylim[0],ylim[1])
			# plt.show()
			if fname is not None:
				plt.savefig(fname, bbox_inches='tight')

		rhs_obj = model.model[1].ode.rhs
		for t in range(0 if args.theta<1 else 1, args.steps if args.theta==0 else args.steps+1):
			spectrum = args.T/args.steps * rhs_obj.spectrum(t,ytrain1[t])
			plt.subplot(1, args.steps+1, t+1)
			plot_stab(args.theta, fig_no-1, 0)
			plt.plot(spectrum[:,0],spectrum[:,1], '.')
			plt.gca().title.set_text('Layer %d'%t)


		###############################################
		# plot vector field
		from matplotlib.widgets import Slider

		rhs_obj = model.model[1].ode.rhs

		fig = plt.figure(fig_no); fig_no += 1
		plot_ax   = plt.axes([0.1, 0.2, 0.8, 0.65])
		slider_ax = plt.axes([0.1, 0.05, 0.8, 0.05])

		# vector fields
		X = np.linspace(np.amin(ypred1[...,0]),     np.amax(ypred1[...,0]),     25)
		Y = np.linspace(np.amin(ypred1[...,1]) - 1, np.amax(ypred1[...,1]) + 1, 25)
		X,Y = np.meshgrid(X,Y)
		XY  = np.zeros((X.size,2))
		XY[:,0] = X.ravel()
		XY[:,1] = Y.ravel()
		UV = []
		for step in range(args.steps+1):
			UV.append(rhs_obj(step, torch.from_numpy(XY).float()).detach().numpy())
		UV = np.array(UV)


		# true function
		xtrue = np.linspace(-5, 5, 200)
		ytrue = fun(xtrue)

		# initial plot
		plt.axes(plot_ax)
		############
		# if args.theta<1:
		# 	plot_ax.quiver(X, Y, (1-args.theta)*UV[0][:,0], (1-args.theta)*UV[0][:,1], color='r')
		# if args.theta>0:
		# 	plot_ax.quiver(X, Y,     args.theta*UV[1][:,0],     args.theta*UV[1][:,1], color='b')
		plot_ax.quiver(X, Y, (1-args.theta)*UV[0][:,0] + args.theta*UV[1][:,0], (1-args.theta)*UV[0][:,1] + args.theta*UV[1][:,1])
		############
		for point in range(ytrain1.shape[1]):
			xx = [ ytrain1[0,point,0], ytrain1[1,point,0] ]
			yy = [ ytrain1[0,point,1], ytrain1[1,point,1] ]
			plot_ax.plot(xx, yy, '-b')
		############
		plot_ax.plot(ypred1[0,:,0], ypred1[0,:,1], '-r')
		plot_ax.plot(ypred1[1,:,0], ypred1[1,:,1], '-b')
		plot_ax.plot(ytrain1[0,:,0], ytrain1[0,:,1], '.r')
		plot_ax.plot(ytrain1[1,:,0], ytrain1[1,:,1], '.b')
		############
		plot_ax.set_xlim(np.amin(X),np.amax(X))
		plot_ax.set_ylim(np.amin(Y),np.amax(Y))

		# create the slider
		a_slider = Slider( slider_ax, label='step', valmin=0, valmax=args.steps, valinit=0, valstep=1 )
		def update(step):
			t = int(step)
			plot_ax.clear()
			if step<args.steps:
				############
				# if args.theta<1:
				# 	plot_ax.quiver(X, Y, (1-args.theta)*UV[t][:,0], (1-args.theta)*UV[t][:,1], color='r')
				# if args.theta>0:
				# 	plot_ax.quiver(X, Y, args.theta*UV[t+1][:,0], args.theta*UV[t+1][:,1], color='b')
				plot_ax.quiver(X, Y, (1-args.theta)*UV[t][:,0] + args.theta*UV[t+1][:,0], (1-args.theta)*UV[t][:,1] + args.theta*UV[t+1][:,1])
				############
				for point in range(ytrain1.shape[1]):
					xx = [ ytrain1[t,point,0], ytrain1[t+1,point,0] ]
					yy = [ ytrain1[t,point,1], ytrain1[t+1,point,1] ]
					plot_ax.plot(xx, yy, '-b')
				############
				plot_ax.plot(ypred1[t,:,0],    ypred1[t,:,1],    '-r')
				plot_ax.plot(ypred1[t+1,:,0],  ypred1[t+1,:,1],  '-b')
				plot_ax.plot(ytrain1[t,:,0],   ytrain1[t,:,1],   '.r')
				plot_ax.plot(ytrain1[t+1,:,0], ytrain1[t+1,:,1], '.b')
				############
				plot_ax.set_xlim(np.amin(X),np.amax(X))
				plot_ax.set_ylim(np.amin(Y),np.amax(Y))
			elif step==args.steps:
				plot_ax.quiver(X, Y, (1-args.theta)*UV[t-1][:,0] + args.theta*UV[t][:,0], (1-args.theta)*UV[t-1][:,1] + args.theta*UV[t][:,1])
				plot_ax.plot(ypred1[t,:,0], ypred1[t,:,1], '-b')
				plot_ax.set_xlim(np.amin(X),np.amax(X))
				plot_ax.set_ylim(np.amin(Y),np.amax(Y))
			fig.canvas.draw_idle()

		a_slider.on_changed(update)


		###############################################
		# trajectories

		# ntest = 25
		# xtest = np.zeros((ntest,2))
		# xtest[:,0] = np.linspace(-6, 6, ntest)
		# ytest = model(torch.from_numpy(xtest).float(), cache_hidden=True).detach().numpy()
		# fig = plt.figure(fig_no); fig_no += 1
		# for point in range(ntest):
		# 	plt.plot(ytest[:,point,0], ytest[:,point,1], '-o')

		plt.show()