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


	def correct_rhs(t,x):
		alpha, beta, gamma, delta = 2./3., 4./3., 1., 1.
		z = np.ones_like(x)
		z[...,0] = alpha*x[...,0] - beta*x[...,0]*x[...,1]
		z[...,1] = delta*x[...,0]*x[...,1] - gamma*x[...,1]
		return z


	def get_data(train_valid='training', T=args.T, steps=args.steps, y0=None, rhs=correct_rhs):
		from scipy.integrate import solve_ivp

		# data points
		if y0 is None:
			size = 5
			y0 = np.ones((size,2))
			if train_valid=='training':
				y0[:,1] = 1.0 + 0.2*np.arange(size)
			elif train_valid=='validation':
				y0[:,1] = 0.9 + 0.2*np.arange(size)

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

	t_train, y0_train, y_train = get_data('training')
	t_valid, y0_valid, y_valid = get_data('validation')

	dataset     = torch.utils.data.TensorDataset( y0_train, y_train)
	val_dataset = torch.utils.data.TensorDataset( y0_valid, y_valid)


	#########################################################################################
	#########################################################################################
	# Loss


	loss_fn = lambda input, target: (input[:,::args.steps//args.datasteps,:]-target[:,::args.steps//args.datasteps,:]).pow(2).flatten().mean()


	#########################################################################################
	#########################################################################################
	# NN model


	########################################################
	class ode_block(torch.nn.Module):
		def __init__(self, rhs, T, steps, theta):
			super().__init__()
			self.ode = theta_solver(rhs, T, steps, theta, tol=args.tol)

		def forward(self, y0, t0=0):
			return self.ode.sequence(y0)[1]
	########################################################

	rhs   = rhs_mlp(2, args.width, args.depth, T=1, num_steps=1, activation=args.sigma, learn_scales=args.learn_scales, learn_shift=args.learn_shift)
	model = ode_block( rhs, args.T, args.steps, args.theta )



	#########################################################################################
	#########################################################################################
	# init/train/plot model

	# uncommenting this line will enable debug mode and lead to increased cost and memory leaking
	# torch.autograd.set_detect_anomaly(True)


	model = ex_setup.load_model(model, args, _device)


	if args.mode=="train":
		optimizer   = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.alpha['wdecay'])
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True, threshold=1.e-5, threshold_mode='rel', cooldown=50, min_lr=1.e-6, eps=1.e-8)
		# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.7, last_epoch=-1)
		train_model = utils.TrainingLoop(model, loss_fn, dataset, args.batch, optimizer, val_dataset=val_dataset, scheduler=scheduler, val_freq=50, stat_freq=10)

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
		from implicitresnet.utils.spectral import eigenvalues, spectralnorm
		#########################################################################################
		fig_no = 0


		images_output = "%s/%s"%(Path(paths['output_images']), args.name)
		data_output   = "%s/%s"%(Path(paths['output_data']),   args.name)

		model.eval()
		rhs_obj = model.ode.rhs

		# extrapolation periods
		periods = 20


		#########################################################################################
		# prepare data

		y_train = y_train.detach().numpy()

		# learned discrete solution
		model.ode.T         = periods*args.T
		model.ode.num_steps = periods*args.steps
		y_learned = model(y0_train).detach().numpy()
		t_learned = model.ode._t.detach().numpy()

		# continuous solution with learned vector field
		model.ode.T         = periods*args.T
		model.ode.num_steps = periods*1000
		model.ode.theta     = 0.5
		y_learned_ode = model(y0_train).detach().numpy()
		t_learned_ode = model.ode._t.detach().numpy()

		# save solutions as tables such that x,y solution components come in pairs (total num_trajectories pairs) along the second dimension
		np.savetxt( Path(paths['output_data'],'training_data.csv'),          np.hstack( (t_train.reshape((-1,1)),                np.concatenate(y_train,  axis=1))                 ), delimiter=',')
		np.savetxt( Path(data_output+'_learned_solution_1period.csv'),       np.hstack( (t_learned.reshape((-1,1))[:args.steps], np.concatenate(y_learned,axis=1)[:args.steps,...])), delimiter=',')
		np.savetxt( Path(data_output+'_learned_solution_20periods.csv'),     np.hstack( (t_learned.reshape((-1,1)),              np.concatenate(y_learned,axis=1))                 ), delimiter=',')
		np.savetxt( Path(data_output+'_learned_ode_solution_20periods.csv'), np.hstack( (t_learned_ode.reshape((-1,1))[::10,:],  np.concatenate(y_learned_ode,axis=1)[::10,...])   ), delimiter=',')


		###############################################
		# plot vector field

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

		# exact vector field
		fig = plt.figure(fig_no); fig_no += 1
		plt.quiver(X, Y, UV_ode[:,0], UV_ode[:,1], angles='xy')
		plt.xlim(0, 3)
		plt.ylim(0, 2)
		plt.gca().axes.xaxis.set_visible(False)
		plt.gca().axes.yaxis.set_visible(False)
		plt.gca().axis('off')
		plt.savefig(Path(paths['output_images'],'true_vector_field.pdf'), bbox_inches='tight', pad_inches=0.0)


		# plot training data
		_, _, y_ode = get_data('training', periods*args.T, periods*1000)
		y_ode = y_ode.detach().numpy()
		for i in range(len(y_ode)):
			plt.plot(y_ode[i,:,0],  y_ode[i,:,1],  '-b')
			plt.plot(y0_train[i,0], y0_train[i,1], 'or', markersize=5.0)
		plt.xlim(0, 3)
		plt.ylim(0, 2)
		plt.gca().axes.xaxis.set_visible(False)
		plt.gca().axes.yaxis.set_visible(False)
		plt.gca().axis('off')
		plt.savefig(Path(paths['output_images'],'training_data.pdf'), bbox_inches='tight', pad_inches=0.0)


		# learned vector field
		fig = plt.figure(fig_no); fig_no += 1
		plt.quiver(X, Y, UV[:,0], UV[:,1], angles='xy')
		plt.xlim(0, 3)
		plt.ylim(0, 2)
		plt.gca().axes.xaxis.set_visible(False)
		plt.gca().axes.yaxis.set_visible(False)
		plt.gca().axis('off')
		plt.savefig(Path(images_output+'_learned_vector_field.pdf'), bbox_inches='tight', pad_inches=0.0)


		###############################################
		fig = plt.figure(fig_no); fig_no += 1

		# evaluate spectrum
		spectrum_train = []
		xmax = ymax = 4
		for i, t in enumerate(t_train[:args.steps+1]):
			y = (1-args.theta) * y_learned[:,i,:] + args.theta * y_learned[:,i+1,:]
			spectrum_train.append( eigenvalues( lambda x: rhs_obj(t,x), torch.from_numpy(y) ) )
			xmax = max(np.amax(np.abs(spectrum_train[-1][:,0])),xmax)
			ymax = max(np.amax(np.abs(spectrum_train[-1][:,1])),ymax)
		spectrum_train = np.concatenate(spectrum_train)

		# plot spectrum
		ex_setup.plot_stab(args.theta, xlim=(-xmax,xmax), ylim=(-ymax,ymax))
		plt.plot(spectrum_train[:,0], spectrum_train[:,1],'bo', markersize=4)
		plt.savefig(images_output+"_spectrum.pdf", bbox_inches='tight', pad_inches=0.0)
