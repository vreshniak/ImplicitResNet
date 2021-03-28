import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
import numpy as np

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


	correct_rhs   = lambda t,y: -20*(y-np.cos(t))
	piecewise_rhs = lambda t,y: -20*(y-np.cos(int(t*args.steps/args.T)*args.T/args.steps))


	def get_data(samples=args.datasize, T=args.T, steps=args.steps, t0=None, y0=None, rhs=correct_rhs):
		from scipy.integrate import solve_ivp
		# data points
		if t0 is not None and y0 is not None:
			samples = len(t0)
			t0 = np.array(t0).ravel()
			y0 = np.array(y0).ravel()
		else:
			t0 = np.random.rand(samples) * T
			y0 = 2*np.random.rand(*t0.shape) - 0.5

			t0.ravel()[0] = 0
			y0.ravel()[0] = 0

		t = []
		y = []
		for tt0, yy0 in zip(t0,y0):
			sol = solve_ivp(rhs, t_span=[tt0, 2*T], y0=[yy0], t_eval=np.linspace(tt0,tt0+T,steps+1))
			t.append( sol.t )
			y.append( sol.y )
		t = np.vstack(t)
		y = np.vstack(y)

		t  = torch.from_numpy(t).float()
		y  = torch.from_numpy(y).reshape((samples,steps+1,1)).float()
		t0 = torch.from_numpy(t0).float()
		y0 = torch.from_numpy(y0).reshape((samples,1)).float()

		return t0, y0, t, y


	# # plot exact solution
	# import matplotlib.pyplot as plt
	# t0 = np.linspace(0,args.T,40)
	# y0 = 1.5*np.ones_like(t0)
	# t0 = np.hstack((t0,t0))
	# y0 = np.hstack((y0,np.zeros_like(y0)))
	# t0_train, y0_train, t_train, y_train = get_data(t0=t0, y0=y0)
	# t_train = t_train.reshape((-1,args.steps+1))
	# y_train = y_train.reshape((-1,args.steps+1))
	# for i in range(len(t_train)):
	# 	plt.plot(t_train[i], y_train[i], '-k')
	# plt.show()
	# exit()


	t0_train, y0_train, t_train, y_train = get_data(args.datasize)
	t0_valid, y0_valid, t_valid, y_valid = get_data(args.steps)

	dataset     = torch.utils.data.TensorDataset( y0_train, t0_train, y_train)
	val_dataset = torch.utils.data.TensorDataset( y0_valid, t0_valid, y_valid)


	#########################################################################################
	#########################################################################################
	# NN parameters


	# loss_fn = lambda input, target: (input[:,::args.steps//args.datasteps,:]-target[:,::args.steps//args.datasteps,:]).pow(2).flatten().mean()
	loss_fn = lambda input, target: (input-target[:,::args.steps//args.datasteps,:]).pow(2).flatten().mean()


	#########################################################################################
	#########################################################################################
	# NN model


	rhs   = rhs_mlp(1, args.width, args.depth, T=2*args.T, num_steps=2*args.steps, activation=args.sigma, power_iters=args.piters, spectral_limits=args.eiglims, learn_scales=args.learn_scales, learn_shift=args.learn_shift)
	model = regularized_ode_solver(theta_solver(rhs, args.T, args.steps, args.theta, ind_out=torch.arange(0,args.steps+1,args.steps//args.datasteps), tol=args.tol), alpha=args.alpha, mciters=1, p=0)



	#########################################################################################
	#########################################################################################
	# init/train/test model

	# uncommenting this line will enable debug mode and lead to increased cost and memory leaking
	# torch.autograd.set_detect_anomaly(True)


	model = ex_setup.load_model(model, args, _device)


	if args.mode=="train":
		optimizer   = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.alpha['wdecay'])
		# scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=10, verbose=True, threshold=1.e-4, threshold_mode='rel', cooldown=10, min_lr=1.e-6, eps=1.e-8)
		scheduler   = utils.optim.EvenReductionLR(optimizer, lr_reduction=0.2, gamma=0.8, epochs=args.epochs, last_epoch=-1)
		train_model = utils.TrainingLoop(model, loss_fn, dataset, args.batch, optimizer, val_dataset=val_dataset, scheduler=scheduler, val_freq=1, stat_freq=1)

		writer = SummaryWriter(Path("logs",file_name))

		# save initial model and train
		torch.save( model.state_dict(), Path(paths['chkp_init'],file_name) )
		try:
			train_model(args.epochs, writer=writer)
		except:
			raise
		finally:
			torch.save( model.state_dict(), Path(paths['chkp_final'],file_name) )

		writer.close()

	elif args.mode=="test":
		import matplotlib.pyplot as plt
		from implicitresnet.utils.spectral import eigenvalues, spectralnorm

		def savefig(*aargs, **kwargs):
			plt.gca().axis('off')
			plt.xlim(0, args.T)
			plt.ylim(-0.5, 1.5)
			plt.savefig(*aargs, **kwargs)

		#########################################################################################
		fig_no = 0


		images_output = "%s/%s"%(Path(paths['out_images']), args.name)
		data_output   = "%s/%s"%(Path(paths['out_data']),   args.name)

		model.eval()
		rhs_obj = model.rhs


		#########################################################################################
		# evaluate vector fields
		X = np.linspace(0, args.T, 2*args.steps).reshape(-1,1)
		Y = np.linspace(-0.5, 1.5, 15).reshape(-1,1)
		UV_ode = []
		UV     = []
		for i in range(len(X)):
			UV_ode.append(correct_rhs(X[i,0], Y).reshape((-1,1)))
			UV.append(rhs_obj(X[i,0], torch.from_numpy(Y).float()).detach().numpy().reshape((-1,1)))
		UV_ode = np.hstack(UV_ode)
		UV     = np.hstack(UV)
		X,Y = np.meshgrid(X,Y)
		XY  = np.hstack((X.reshape(-1,1), Y.reshape(-1,1)))


		#########################################################################################
		# prepare data


		# true solution on the discrete time grid
		_, _, t_true, y_true = get_data(samples=20)
		t_true = t_true.reshape((-1,args.steps+1)).detach().numpy()
		y_true = y_true.reshape((-1,args.steps+1)).detach().numpy()


		# true continuous solution
		t0, y0, t_ode, y_ode = get_data(samples=20, steps=1000)
		t_ode = t_ode.reshape((-1,1001)).detach().numpy()
		y_ode = y_ode.reshape((-1,1001)).detach().numpy()


		# learned solution on the discrete time grid
		model.ind_out = torch.arange(args.steps+1)
		t_learned, y_learned = model(y0, t0, return_t=True)
		t_learned = t_learned.reshape((-1,args.steps+1)).detach().numpy()
		y_learned = y_learned.reshape((-1,args.steps+1)).detach().numpy()


		# continuous solution with learned vector field
		model.num_steps = 1000
		model.theta     = 0.0
		model.ind_out   = torch.arange(1001)
		t_learned_ode, y_learned_ode = model(y0, t0, return_t=True)
		t_learned_ode = t_learned_ode.reshape((-1,1001)).detach().numpy()
		y_learned_ode = y_learned_ode.reshape((-1,1001)).detach().numpy()


		# write data as tables with time/solution coming in pairs (total samples pairs)
		np.savetxt( Path(paths['out_data'],'true_solution.csv'), np.moveaxis(np.stack((t_true,y_true)),0,1).reshape((-1,args.steps+1)).T,       delimiter=',')
		np.savetxt( Path(paths['out_data'],'ode_solution.csv'),  np.moveaxis(np.stack((t_ode,y_ode)),0,1).reshape((-1,1001)).T,                 delimiter=',')
		np.savetxt( Path(data_output+'_learned_solution.csv'),      np.moveaxis(np.stack((t_learned,y_learned)),0,1).reshape((-1,args.steps+1)).T, delimiter=',')
		np.savetxt( Path(data_output+'_learned_ode_solution.csv'),  np.moveaxis(np.stack((t_learned_ode,y_learned_ode)),0,1).reshape((-1,1001)).T, delimiter=',')


		###############################################
		fig = plt.figure(fig_no); fig_no += 1

		# evaluate spectrum
		spectrum_train = []
		for i in range(t_learned.shape[1]-1):
			y = (1-args.theta) * y_learned[:,i] + args.theta * y_learned[:,i+1]
			y = torch.from_numpy(y).unsqueeze(1)
			t = torch.from_numpy(t_learned[:,i])
			spectrum_train.append( eigenvalues( lambda x: rhs_obj(t,x), y ) )
		spectrum_train = np.concatenate(spectrum_train)

		# plot spectrum
		# ex_setup.plot_stab(args.theta, xlim=(-25,-15), ylim=(-5,5))
		plt.plot(spectrum_train[:,0], spectrum_train[:,1],'bo', markersize=4)
		plt.savefig(images_output+"_spectrum.pdf", bbox_inches='tight', pad_inches=0.0)



		###############################################
		# plot vector field


		# plot learned trajectories
		fig = plt.figure(fig_no); fig_no += 1
		plt.quiver(X, Y, np.ones_like(X), UV.reshape(X.shape), angles='xy')
		plt.plot(t_ode[0],np.cos(t_ode[0]),'-r', linewidth=2)
		for t, y in zip(t_learned,y_learned):
			plt.plot(t,y,'-b', linewidth=1)
		savefig(images_output+'_learned_trajectories.pdf', bbox_inches='tight', pad_inches=0.0)

		# plot learned continuous trajectories
		fig = plt.figure(fig_no); fig_no += 1
		plt.quiver(X, Y, np.ones_like(X), UV.reshape(X.shape), angles='xy')
		plt.plot(t_ode[0],np.cos(t_ode[0]),'-r', linewidth=2)
		for t, y in zip(t_learned_ode,y_learned_ode):
			plt.plot(t,y,'-b', linewidth=1)
		savefig(images_output+'_learned_ode_trajectories.pdf', bbox_inches='tight', pad_inches=0.0)


		# plot exact vector field
		fig = plt.figure(fig_no); fig_no += 1
		plt.quiver(X, Y, np.ones_like(X), UV_ode.reshape(X.shape), angles='xy')
		savefig(Path(paths['out_images'],'ode_vector_field.pdf'), bbox_inches='tight', pad_inches=0.0)

		# plot learned vector field
		fig = plt.figure(fig_no); fig_no += 1
		plt.quiver(X, Y, np.ones_like(X), UV.reshape(X.shape), angles='xy')
		savefig(Path(images_output+'_vector_field.pdf'), bbox_inches='tight', pad_inches=0.0)


		# plot training data
		fig = plt.figure(fig_no); fig_no += 1
		steps=1000
		t0_train = t0_train.detach().numpy().flatten()
		y0_train = y0_train.detach().numpy().flatten()
		_, _, t_init, y_init = get_data(steps=steps, t0=t0_train, y0=y0_train)
		t_init = t_init.detach().numpy().reshape((len(t0_train),steps+1))
		y_init = y_init.detach().numpy().reshape((len(t0_train),steps+1))
		plt.quiver(X, Y, np.ones_like(X), UV_ode.reshape(X.shape), angles='xy')
		for i in range(len(t_init)):
			plt.plot(t_init[i],  y_init[i],  '-b', linewidth=1.5)
			# plt.plot(t_train[i], y_train[i], 'ok', markersize=5.0)
		plt.plot(t0_train, y0_train, 'or', markersize=5.0)
		# plt.plot(tt, yy, 'or')
		savefig(Path(paths['out_images'],'training_data.pdf'), bbox_inches='tight', pad_inches=0.0)
