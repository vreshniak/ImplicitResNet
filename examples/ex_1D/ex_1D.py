import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
import numpy as np
import pickle

import torch
from torch.utils.tensorboard import SummaryWriter

from implicitresnet import utils, theta_solver, regularized_ode_solver, rhs_mlp
import ex_setup



if __name__ == '__main__':
	#########################################################################################
	#########################################################################################


	args  = ex_setup.parse_args()
	paths = ex_setup.create_paths(args)
	if args.seed is not None: torch.manual_seed(args.seed)


	#########################################################################################
	#########################################################################################
	# compose file name


	file_name   = ex_setup.make_name(args)
	script_name = sys.argv[0][:-3]


	#########################################################################################
	#########################################################################################


	_device = ex_setup._cpu
	_dtype  = ex_setup._dtype


	#########################################################################################
	#########################################################################################
	# Data
	np.random.seed(args.seed)

	fun  = (lambda x: np.sin(x) + 0.0*np.cos(19.0*x))
	# initialization of codimensions
	phi  = lambda x: torch.zeros((x.size(0),args.codim))
	# phi  = lambda x: torch.cat( [x for _ in range(args.codim)], 1 )
	# phi  = lambda x: torch.from_numpy(fun(x.detach().numpy()))


	# training data
	ntrain = args.datasize
	xtrain = np.linspace(-5, 5, ntrain).reshape((ntrain,1)) if ntrain>1 else np.array([[1]])
	ytrain = fun(xtrain)
	# xtrain = np.vstack( [np.linspace(-5, -1, ntrain//2).reshape((-1,1)), np.linspace(1, 5, ntrain//2).reshape((-1,1))] )
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
	# Loss


	loss_fn = torch.nn.MSELoss(reduction='mean')


	#########################################################################################
	#########################################################################################
	# NN model


	########################################################
	# augment original data with additional dimensions
	class augment(torch.nn.Module):
		def __init__(self):
			super().__init__()

		def forward(self, x, val=0):
			if val==0:
				return torch.cat( (x, phi(x)), 1 )
			else:
				return torch.cat( (x, val*torch.ones((x.size(0),args.codim))), 1 )
	########################################################

	########################################################
	# last component as the function value
	class output(torch.nn.Module):
		def __init__(self):
			super().__init__()

		def forward(self, x):
			return x[:,-1:]
	########################################################

	########################################################
	def ode_block():
		dim = 1 + args.codim
		rhs_steps = args.steps if args.alpha['TV']>=0 else 1
		rhs    = rhs_mlp(dim, args.width, args.depth, T=args.T, num_steps=rhs_steps, activation=args.sigma, learn_scales=args.learn_scales, learn_shift=args.learn_shift)
		solver = theta_solver( rhs, args.T, args.steps, args.theta, tol=args.tol )
		return regularized_ode_solver( solver, args.alpha, mciters=args.mciters )
	########################################################

	model = torch.nn.Sequential( augment(), ode_block(), output() )


	#########################################################################################
	#########################################################################################
	# init/train/test model

	# uncommenting this line will enable debug mode and lead to increased cost and memory leaking
	# torch.autograd.set_detect_anomaly(True)


	model = ex_setup.load_model(model, args, _device)


	if args.mode=="train":
		optimizer   = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.alpha['wdecay'])
		scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=50, verbose=True, threshold=1.e-4, threshold_mode='rel', cooldown=50, min_lr=1.e-6, eps=1.e-8)
		train_model = utils.TrainingLoop(model, loss_fn, dataset, args.batch, optimizer, val_dataset=val_dataset, scheduler=scheduler, val_freq=50, stat_freq=10)

		# write options to file
		if os.path.exists(Path('output','name2args')):
			with open(Path('output','name2args'),'rb') as f:
				name2args = pickle.load(f)
			name2args[file_name] = args
		else:
			name2args = {file_name:args}
		with open(Path('output','name2args'),'wb') as f:
			pickle.dump(name2args, f)

		writer = SummaryWriter(Path("logs",file_name))

		# save initial model and train
		torch.save( model.state_dict(), Path(paths['checkpoints_0'],file_name) )
		try:
			train_model(args.epochs, writer=writer)
		except:
			raise
		finally:
			torch.save( model.state_dict(), Path(paths['checkpoints'],file_name) )

		writer.close()

	elif args.mode=="test":
		import matplotlib.pyplot as plt
		from matplotlib.widgets import Slider
		from implicitresnet.utils.spectral import eigenvalues, spectralnorm
		#########################################################################################
		fig_no = 0

		# images_output = ("%s/th%4.2f_T%d_data%d_adiv%4.2f"%(Path(paths['output'],'images'), args.theta, args.T, args.datasize, args.adiv)).replace('.','')
		images_output = "%s/%s"%(Path(paths['output'],'images'), args.name)

		model.eval()

		rhs_obj = model[1].rhs

		#########################################################################################
		# prepare data

		# test data
		ntest = 200
		xtest = np.linspace(-6, 6, ntest).reshape((ntest,1))

		model_output_train = model(torch.from_numpy(xtrain).float()).detach().numpy()
		model_output_test  = model(torch.from_numpy(xtest).float()).detach().numpy()

		ode_output_train = [ y.detach().numpy() for y in model[1].trajectory( model[0](torch.from_numpy(xtrain).float()))[1] ]
		ode_output_test  = [ y.detach().numpy() for y in model[1].trajectory( model[0](torch.from_numpy(xtest).float()) )[1] ]

		std = 0.2
		ode_output_up   = [ y.detach().numpy() for y in model[1].trajectory( model[0](torch.from_numpy(xtest).float(), std))[1] ]
		ode_output_down = [ y.detach().numpy() for y in model[1].trajectory( model[0](torch.from_numpy(xtest).float(),-std))[1] ]

		###############################################
		# # plot function
		# fig = plt.figure(fig_no); fig_no += 1

		# plt.plot(xtest,  model_output_test)
		# plt.plot(xtrain, ytrain,'o')
		# plt.show()


		###############################################
		# evaluate spectrum

		spectrum_train = []
		spectrum_test  = []
		xmax = ymax = 4
		for t in range(len(ode_output_train)-1):
			y = (1-args.theta) * ode_output_train[t] + args.theta * ode_output_train[t+1]
			spectrum_train.append( eigenvalues( lambda x: rhs_obj(t,x), torch.from_numpy(y) ) )
			xmax = max(np.amax(np.abs(spectrum_train[-1][:,0])),xmax)
			ymax = max(np.amax(np.abs(spectrum_train[-1][:,1])),ymax)

			y = (1-args.theta) * ode_output_test[t] + args.theta * ode_output_test[t+1]
			spectrum_test.append( eigenvalues( lambda x: rhs_obj(t,x), torch.from_numpy(y) ) )
			xmax = max(np.amax(np.abs(spectrum_test[-1][:,0])),xmax)
			ymax = max(np.amax(np.abs(spectrum_test[-1][:,1])),ymax)
		spectrum_train = np.concatenate(spectrum_train)
		spectrum_test  = np.concatenate(spectrum_test)


		###############################################
		# plot spectrum
		fig = plt.figure(fig_no); fig_no += 1

		ex_setup.plot_stab(args.theta, xlim=(-xmax,xmax), ylim=(-ymax,ymax))

		plt.plot(spectrum_test[:,0],  spectrum_test[:,1], 'bo', markersize=4) #, markerfacecolor='none')
		plt.plot(spectrum_train[:,0], spectrum_train[:,1],'ro', markersize=4)

		plt.savefig(images_output+"_spectrum.pdf", bbox_inches='tight', pad_inches=0.0)


		###############################################
		# evaluate vector field
		fig = plt.figure(fig_no); fig_no += 1

		fig = plt.figure(fig_no); fig_no += 1
		plot_ax   = plt.axes([0.1, 0.2, 0.8, 0.65])
		slider_ax = plt.axes([0.1, 0.05, 0.8, 0.05])

		# vector fields
		X = np.linspace(-7, 7, 25)
		Y = np.linspace(-2, 2, 25)
		X,Y = np.meshgrid(X,Y)
		XY  = np.hstack((X.reshape(-1,1), Y.reshape(-1,1)))
		UV  = []
		for t in range(len(ode_output_train)):
			UV.append(rhs_obj(t, torch.from_numpy(XY).float()).detach().numpy())
		UV = np.array(UV)


		# ###############################################
		# # plot vector field and trajectories
		# fig = plt.figure(fig_no); fig_no += 1

		# def plot_step(t, ax=plot_ax):
		# 	ax.clear()
		# 	if t<args.T:
		# 		ax.quiver(X, Y, UV[t][:,0], UV[t][:,1], angles='xy')
		# 		# ax.quiver(X, Y, (1-args.theta)*UV[t][:,0] + args.theta*UV[t+1][:,0], (1-args.theta)*UV[t][:,1] + args.theta*UV[t+1][:,1])
		# 		############
		# 		# connecting lines
		# 		for point in range(len(ode_output_train)):
		# 			xx = [ ode_output_train[t][point,0], ode_output_train[t+1][point,0] ]
		# 			yy = [ ode_output_train[t][point,1], ode_output_train[t+1][point,1] ]
		# 			ax.annotate("", xy=(xx[1], yy[1]), xytext=(xx[0],yy[0]), arrowprops=dict(arrowstyle="->"))
		# 			# plot_ax.plot(xx, yy, '-b')
		# 		############
		# 		ax.plot(ode_output_test[t][:,0],    ode_output_test[t][:,1],    '-r')
		# 		ax.plot(ode_output_test[t+1][:,0],  ode_output_test[t+1][:,1],  '-b')
		# 		ax.plot(ode_output_train[t][:,0],   ode_output_train[t][:,1],   '.r')
		# 		ax.plot(ode_output_train[t+1][:,0], ode_output_train[t+1][:,1], '.b')
		# 		############
		# 		ax.set_xlim(np.amin(X),np.amax(X))
		# 		ax.set_ylim(np.amin(Y),np.amax(Y))
		# 	elif t==args.T:
		# 		ax.quiver(X, Y, UV[t-1][:,0], UV[t-1][:,1], angles='xy')
		# 		ax.plot(ode_output_test[t][:,0], ode_output_test[t][:,1], '-b')
		# 		ax.set_xlim(np.amin(X),np.amax(X))
		# 		ax.set_ylim(np.amin(Y),np.amax(Y))
		# 	fig.canvas.draw_idle()

		# # initial plot
		# plot_step(0)
		# # create the slider
		# a_slider = Slider( slider_ax, label='step', valmin=0, valmax=len(ode_output_train)-1, valinit=0, valstep=1 )
		# def update(step):
		# 	plot_step(int(step))
		# a_slider.on_changed(update)
		# plt.show()


		###############################################
		# plot spread of the solution
		fig = plt.figure(fig_no); fig_no += 1

		plt.quiver(X, Y, UV[0][:,0], UV[0][:,1], angles='xy')
		for t in range(len(ode_output_train)-1):
			for point in range(ode_output_train[t].shape[0]):
				xx = [ ode_output_train[t][point,0], ode_output_train[t+1][point,0] ]
				yy = [ ode_output_train[t][point,1], ode_output_train[t+1][point,1] ]
				plt.plot(xx, yy, '-k', linewidth=1.5)
				plt.gca().annotate("", xy=(xx[1], yy[1]), xytext=(xx[0],yy[0]), arrowprops=dict(arrowstyle="->"))
		plt.fill( np.concatenate((ode_output_down[0][:,0], ode_output_up[0][-1::-1,0])),  np.concatenate((ode_output_down[0][:,1], ode_output_up[0][-1::-1,1])),  'b', alpha=0.4 )
		plt.fill( np.concatenate((ode_output_down[-1][:,0],ode_output_up[-1][-1::-1,0])), np.concatenate((ode_output_down[-1][:,1],ode_output_up[-1][-1::-1,1])), 'r', alpha=0.4 )
		plt.plot(ode_output_test[0][:,0],   ode_output_test[0][:,1],   '-b', linewidth=2.5)
		plt.plot(ode_output_test[-1][:,0],  ode_output_test[-1][:,1],  '-r', linewidth=2.5)
		plt.plot(ode_output_train[0][:,0],  ode_output_train[0][:,1],  '.b', markersize=8)
		plt.plot(ode_output_train[-1][:,0], ode_output_train[-1][:,1], '.r', markersize=8)
		plt.xlim(-7,7)
		plt.ylim(-2,2)
		# plt.plot(ytrain11[-1,:,0], ytrain11[-1,:,1], '.g', markersize=4)
		plt.gca().axes.xaxis.set_visible(False)
		plt.gca().axes.yaxis.set_visible(False)
		plt.gca().axis('off')
		plt.savefig(images_output+"_traj.pdf", pad_inches=0.0, bbox_inches='tight')
