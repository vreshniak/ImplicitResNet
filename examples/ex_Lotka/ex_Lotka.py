import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from implicitresnet import utils, theta_solver, rhs_mlp
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


	def correct_rhs(t,x):
		alpha, beta, gamma, delta = 2./3., 4./3., 1., 1.
		z = np.ones_like(x) if isinstance(x,np.ndarray) else torch.ones_like(x)
		z[...,0] = alpha*x[...,0] - beta*x[...,0]*x[...,1]
		z[...,1] = delta*x[...,0]*x[...,1] - gamma*x[...,1]
		return z


	def get_data(train_valid=None, T=args.T, steps=args.steps, rhs=correct_rhs):
		from scipy.integrate import solve_ivp

		# initial data points
		num_traj = 14
		y0 = np.ones((num_traj,2))
		y0[:,1] = 0.5 + 0.1*np.arange(num_traj)
		if train_valid is not None:
			if train_valid=='training':
				y0 = y0[1::2,:]
			elif train_valid=='validation':
				y0 = y0[::2,:]

		t = np.linspace(0,T,steps+1)
		y = []
		for yy0 in y0:
			sol = solve_ivp(rhs, t_span=[0, T], y0=yy0, t_eval=t, rtol=1.e-6)
			y.append( sol.y.T[np.newaxis,...] )
		y = np.vstack(y)

		y0 = torch.from_numpy(y0).float()
		y  = torch.from_numpy(y).float()

		return t, y0, y

	# import matplotlib.pyplot as plt
	# # plot exact solution
	# t_train, y0_train, y_train = get_data()
	# for i in range(len(y0_train)):
	# 	plt.plot(t_train, y_train[i,:,0], '-r')
	# 	plt.plot(t_train, y_train[i,:,1], '-b')
	# plt.show()

	# # exact phase plot
	# for i in range(len(y0_train)):
	# 	plt.plot(y_train[i,:,0], y_train[i,:,1], '-')
	# plt.show()
	# exit()

	t_train, y0_train, y_train = get_data('training')
	t_valid, y0_valid, y_valid = get_data('validation')

	dataset     = torch.utils.data.TensorDataset( y0_train, y_train)
	val_dataset = torch.utils.data.TensorDataset( y0_valid, y_valid)


	#########################################################################################
	#########################################################################################
	# Loss


	# loss_fn = lambda input, target: (input-target[:,::args.steps//args.datasteps,:]).pow(2).flatten().mean()
	loss_fn = lambda input, target: (input-target).pow(2).flatten().mean()


	#########################################################################################
	#########################################################################################
	# NN model


	rhs   = rhs_mlp(2, args.width, args.depth, T=1, num_steps=1, activation=args.sigma, learn_scales=args.learn_scales, learn_shift=args.learn_shift)
	model = theta_solver(rhs, args.T, args.steps, args.theta, ind_out=torch.arange(0,args.steps+1), tol=args.tol)


	#########################################################################################
	#########################################################################################
	# init/train/plot model

	# uncommenting this line will enable debug mode and lead to increased cost and memory leaking
	# torch.autograd.set_detect_anomaly(True)


	model = ex_setup.load_model(model, args, _device)


	if args.mode=="train":
		optimizer1 = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.alpha['wdecay'])
		optimizer2 = torch.optim.LBFGS(model.parameters(), lr=args.lr, tolerance_grad=1.e-12, tolerance_change=1.e-12, history_size=100, max_iter=20)
		scheduler = utils.optim.EvenReductionLR(optimizer1, lr_reduction=0.1, gamma=0.8, epochs=args.epochs, last_epoch=-1)
		# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True, threshold=1.e-5, threshold_mode='rel', cooldown=50, min_lr=1.e-6, eps=1.e-8)
		# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.7, last_epoch=-1)
		train_model = utils.TrainingLoop(model, loss_fn, dataset, args.batch, optimizer1, val_dataset=val_dataset, scheduler=scheduler, val_freq=10, stat_freq=10)

		writer = SummaryWriter(Path("logs",file_name))

		# save initial model and train
		torch.save( model.state_dict(), Path(paths['chkp_init'],file_name) )
		try:
			train_model(args.epochs, writer=writer)
			train_model(200, writer=writer, optimizer=optimizer2, scheduler=None)
		except:
			raise
		finally:
			torch.save( model.state_dict(), Path(paths['chkp_final'],file_name) )

		writer.close()

	elif args.mode=="test":
		import matplotlib.pyplot as plt
		from implicitresnet.utils.spectral import eigenvalues, spectralnorm


		def savefig(*args, **kwargs):
			plt.gca().axis('off')
			plt.xlim(0, 3)
			plt.ylim(0, 2)
			plt.savefig(*args, **kwargs)

		#########################################################################################
		fig_no = 0


		images_output = "%s/%s"%(paths['out_images'], args.name)
		data_output   = "%s/%s"%(paths['out_data'],   args.name)

		model.eval()
		rhs_obj = model.rhs

		# extrapolation periods
		periods = 5


		#########################################################################################
		# evaluate vector fields

		X = np.linspace(0, 3, 25).reshape(-1,1)
		Y = np.linspace(0, 2, 25).reshape(-1,1)
		X,Y = np.meshgrid(X,Y)
		XY  = np.hstack((X.reshape(-1,1), Y.reshape(-1,1)))
		UV_ode = []
		UV     = []
		for i in range(len(X)):
			UV_ode.append(correct_rhs(0, XY).reshape((-1,2)))
			UV.append(rhs_obj(0.0, torch.from_numpy(XY).float()).detach().numpy().reshape((-1,2)))
		UV_ode = np.hstack(UV_ode)
		UV     = np.hstack(UV)


		#########################################################################################
		# prepare data

		t_data, y0_data, y_data = get_data()
		t_ode,  y0_ode,  y_ode  = get_data(steps=1000)
		y_data = y_data.detach().numpy()
		y_ode  = y_ode.detach().numpy()


		# learned discrete solution
		model.T          = periods*args.T
		model.num_steps  = periods*args.steps
		model.ind_out    = torch.arange(model.num_steps+1)
		t_learned, y_learned = model(y0_data, return_t=True)
		t_learned = t_learned.detach().numpy()
		y_learned = y_learned.detach().numpy()

		# continuous solution with learned vector field
		model.T         = periods*args.T
		model.num_steps = periods*1000
		model.theta     = 0.5
		model.ind_out   = torch.arange(model.num_steps+1)
		t_learned_ode, y_learned_ode = model(y0_data, return_t=True)
		t_learned_ode = t_learned_ode.detach().numpy()
		y_learned_ode = y_learned_ode.detach().numpy()


		###############################################
		# save data

		# save solutions as tables such that x,y solution components come in pairs (total num_trajectories pairs) along the second dimension
		np.savetxt( Path(paths['out_data'],'training_data.csv'),                    np.hstack( (t_data.reshape((-1,1)),                 np.concatenate(y_data,  axis=1))                 ),  delimiter=',')
		np.savetxt( Path(data_output+'_learned_solution_1period.csv'),                 np.hstack( (t_learned.reshape((-1,1))[:args.steps], np.concatenate(y_learned,axis=1)[:args.steps,...])), delimiter=',')
		np.savetxt( Path(data_output+'_learned_solution_%dperiods.csv'%(periods)),     np.hstack( (t_learned.reshape((-1,1)),              np.concatenate(y_learned,axis=1))                 ), delimiter=',')
		np.savetxt( Path(data_output+'_learned_ode_solution_%dperiods.csv'%(periods)), np.hstack( (t_learned_ode.reshape((-1,1))[::10,:],  np.concatenate(y_learned_ode,axis=1)[::10,...])   ), delimiter=',')


		###############################################
		# plot data

		# plot exact vector field
		fig = plt.figure(fig_no); fig_no += 1
		plt.quiver(X, Y, UV_ode[:,0], UV_ode[:,1], angles='xy')
		savefig(Path(paths['out_images'],'true_vector_field.pdf'), bbox_inches='tight', pad_inches=0.0)


		# plot training/validation data
		for i in range(0,len(y_ode),2):
			plt.plot(y_ode[i,:,0],   y_ode[i,:,1],   '-r')
			plt.plot(y_ode[i+1,:,0], y_ode[i+1,:,1], '-b')
			plt.plot(y_data[i,:,0],   y_data[i,:,1],   'or', markersize=4.0)
			plt.plot(y_data[i+1,:,0], y_data[i+1,:,1], 'ob', markersize=4.0)
		plt.legend(['validation', 'training'])
		savefig(Path(paths['out_images'],'training_validation_data.pdf'), bbox_inches='tight', pad_inches=0.0)


		# plot learned vector field
		fig = plt.figure(fig_no); fig_no += 1
		plt.quiver(X, Y, UV[:,0],     UV[:,1],     angles='xy')
		savefig(images_output+'_learned_vector_field.pdf', bbox_inches='tight', pad_inches=0.0)

		# plot learned vs true vector field
		fig = plt.figure(fig_no); fig_no += 1
		plt.quiver(X, Y, UV_ode[:,0], UV_ode[:,1], angles='xy', color='r')
		plt.quiver(X, Y, UV[:,0],     UV[:,1],     angles='xy')
		plt.legend(['exact', r'$\theta=%4.2f$'%(args.theta)])
		savefig(images_output+'_learned_vs_true_vector_field.pdf', bbox_inches='tight', pad_inches=0.0)


		# plot learned trajectories, 1 period
		fig = plt.figure(fig_no); fig_no += 1
		plt.quiver(X, Y, UV[:,0], UV[:,1], angles='xy')
		for i in range(0,len(y_ode),2):
			plt.plot(y_learned[i,:args.steps,0],   y_learned[i,:args.steps,1],   '-r')
			plt.plot(y_learned[i+1,:args.steps,0], y_learned[i+1,:args.steps,1], '-b')
		plt.legend(['validation', 'training'], loc='upper right')
		savefig(images_output+'_learned_trajectories_1period.pdf', bbox_inches='tight', pad_inches=0.0)

		# plot learned trajectories, multiple periods
		fig = plt.figure(fig_no); fig_no += 1
		plt.quiver(X, Y, UV[:,0], UV[:,1], angles='xy')
		for i in range(0,len(y_ode),2):
			plt.plot(y_learned[i,:,0],   y_learned[i,:,1],   '-r')
			plt.plot(y_learned[i+1,:,0], y_learned[i+1,:,1], '-b')
		plt.legend(['validation', 'training'], loc='upper right')
		savefig(images_output+'_learned_trajectories_%dperiods.pdf'%(periods), bbox_inches='tight', pad_inches=0.0)


		# plot learned continuous trajectories, 1 period
		fig = plt.figure(fig_no); fig_no += 1
		plt.quiver(X, Y, UV[:,0], UV[:,1], angles='xy')
		for i in range(0,len(y_ode),2):
			plt.plot(y_learned_ode[i,:1000,0],   y_learned_ode[i,:1000,1],  '-r')
			plt.plot(y_learned_ode[i+1,:1000,0], y_learned_ode[i+1,:1000,1], '-b')
		plt.legend(['validation', 'training'], loc='upper right')
		savefig(images_output+'_learned_ode_trajectories_1period.pdf', bbox_inches='tight', pad_inches=0.0)

		# plot learned continuous trajectories, multiple periods
		fig = plt.figure(fig_no); fig_no += 1
		plt.quiver(X, Y, UV[:,0], UV[:,1], angles='xy')
		for i in range(0,len(y_ode),2):
			plt.plot(y_learned_ode[i,:,0],   y_learned_ode[i,:,1],   '-r')
			plt.plot(y_learned_ode[i+1,:,0], y_learned_ode[i+1,:,1], '-b')
		plt.legend(['validation', 'training'], loc='upper right')
		savefig(images_output+'_learned_ode_trajectories_%dperiods.pdf'%(periods), bbox_inches='tight', pad_inches=0.0)


		###############################################
		fig = plt.figure(fig_no); fig_no += 1

		# evaluate correct spectrum along the true trajectories
		spectrum_exact = []
		for traj in y_ode[:,:1000,:]:
			spectrum_traj = []
			for i, t in enumerate(t_ode[:1000]):
				spectrum_traj.append( eigenvalues( lambda x: correct_rhs(t,x), torch.from_numpy(traj[i:i+1,:]) ) )
			spectrum_exact.append(np.concatenate(spectrum_traj))

		# plot correct spectrum
		# ex_setup.plot_stab(args.theta, xlim=(-2,2), ylim=(-2,2))
		for i in range(0,len(spectrum_exact),2):
			plt.plot(spectrum_exact[i][:,0],    spectrum_exact[i][:,1],   'r.', markersize=1)
			plt.plot(spectrum_exact[i+1][:,0],  spectrum_exact[i+1][:,1], 'b.', markersize=1)
		plt.legend(['validation', 'training'], loc='upper right')
		# plt.gca().axis('off')
		plt.xlim(-2, 2)
		plt.ylim(-2, 2)
		plt.savefig(Path(paths['out_images'],"true_spectrum.jpg"), bbox_inches='tight', pad_inches=0.0, dpi=300)



		###############################################
		fig = plt.figure(fig_no); fig_no += 1

		# evaluate learned spectrum along the learned continuous trajectories
		spectrum_learned = []
		for traj in y_learned_ode[:,:1000,:]:
			spectrum_traj = []
			for i, t in enumerate(t_learned_ode[:1000]):
				spectrum_traj.append( eigenvalues( lambda x: rhs_obj(t,x), torch.from_numpy(traj[i:i+1,:]) ) )
			spectrum_learned.append(np.concatenate(spectrum_traj))

		# plot learned spectrum
		# ex_setup.plot_stab(args.theta, xlim=(-2,2), ylim=(-2,2))
		for i in range(0,len(spectrum_learned),2):
			plt.plot(spectrum_learned[i][:,0],    spectrum_learned[i][:,1],   'r.', markersize=1)
			plt.plot(spectrum_learned[i+1][:,0],  spectrum_learned[i+1][:,1], 'b.', markersize=1)
		plt.legend(['validation', 'training'], loc='upper right')
		# plt.gca().axis('off')
		plt.xlim(-2, 2)
		plt.ylim(-2, 2)
		plt.savefig(Path(images_output+"_spectrum.jpg"), bbox_inches='tight', pad_inches=0.0, dpi=300)
		# plt.savefig(images_output+"_spectrum.jpg", bbox_inches='tight', pad_inches=0.0, dpi=300)
