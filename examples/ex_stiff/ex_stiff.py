import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter

import ex_setup
import layers
import utils




def theta_solver(fun, T, steps, theta, t0, y0):
	from scipy.optimize import minimize
	h = T / steps
	tt = []
	yy = []
	for i in range(len(t0)):
		t = np.linspace(t0[i], T, steps+1)
		y = [y0[i]]
		for step in range(steps):
			x = y[-1]
			if theta>0:
				residual = lambda z: np.sum( (z - x - h*fun(t[step]+theta*h, (1-theta)*x+theta*z))**2 )
				res = minimize(residual, y[-1]+h*fun(t[step], y[-1]))
				y.append( res.x )
			else:
				y.append( x + h*fun(t[step], x) )
		y = np.hstack(y)
		tt.append(t.reshape(1,-1))
		yy.append(y.reshape(1,-1))
	return np.vstack(tt), np.vstack(yy)





if __name__ == '__main__':
	#########################################################################################
	#########################################################################################


	args = ex_setup.parse_args()
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
	from scipy.integrate import solve_ivp
	np.random.seed(args.seed)


	correct_rhs   = lambda t,y: -20*(y-np.cos(t if t<args.T else args.T))
	piecewise_rhs = lambda t,y: -20*(y-np.cos(int(t*args.steps/args.T)*args.T/args.steps if t<args.T else args.T))


	def get_data(size, steps=args.steps, t0=None, y0=None, rhs=correct_rhs):
		# data points
		if t0 is not None and y0 is not None:
			blocks  = 1
			samples = len(t0)
			t0 = np.array(t0)
			y0 = np.array(y0)
		else:
			assert size>=steps
			size = steps*(size//steps)
			h = args.T / steps

			blocks  = steps
			samples = size//steps
			t0 = []
			time_grid = np.linspace(0,args.T,steps+1)
			for step in range(steps):
				t0.append( (step+np.random.rand(size//steps)) * h )
			t0 = np.hstack(t0)
			y0 = 2*np.random.rand(*t0.shape) - 0.5
			# y0 = 4*np.random.rand(*t0.shape) - 2
			# y0 = 1.5*np.random.rand(*t0.shape)
			t0.ravel()[0] = 0
			y0.ravel()[0] = 0

		t = []
		y = []
		for tt0, yy0 in zip(t0,y0):
			sol = solve_ivp(rhs, t_span=[tt0, 2*args.T], y0=[yy0], t_eval=np.linspace(tt0,tt0+args.T,steps+1))
			t.append( sol.t )
			y.append( sol.y )
		t = np.vstack(t)
		y = np.vstack(y)

		t  = t.reshape((blocks,samples,steps+1))
		y  = y.reshape((blocks,samples,steps+1,1))
		t0 = t0.reshape(blocks,samples,1)
		y0 = y0.reshape(blocks,samples,1)

		t  = torch.from_numpy(t).float()
		y  = torch.from_numpy(y).float()
		t0 = torch.from_numpy(t0).float()
		y0 = torch.from_numpy(y0).float()

		return t0, y0, t, y


	# # plot exact solution
	# t0 = np.linspace(0,args.T,40)
	# y0 = 1.5*np.ones_like(t0)
	# t0 = np.hstack((t0,t0))
	# y0 = np.hstack((y0,np.zeros_like(y0)))
	# t0_train, y0_train, t_train, y_train = get_data(args.datasize, t0=t0, y0=y0)
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


	loss_fn = lambda input, target: (input[:,:,::args.steps//args.datasteps,:]-target[:,:,::args.steps//args.datasteps,:]).pow(2).flatten().mean()


	#########################################################################################
	#########################################################################################
	# NN model

	########################################################
	class ode_rhs(ex_setup.rhs_mlp):
		def __init__(self, data_dim, args):
			super().__init__(data_dim, args)

		def t2ind(self, t):
			# h = args.T/args.steps
			# tint = (t/h).to(torch.int).flatten()
			# assert (tint==tint[0]).all()
			# print(tint)
			if isinstance(t,float):
				return super().t2ind(t)
			else:
				return super().t2ind(t.flatten()[0].item())
	########################################################


	########################################################
	class ode_block(ex_setup.ode_block_base):
		def __init__(self):
			super().__init__(args)
			self.ode = layers.theta_solver( ode_rhs(1, args), args.T, args.steps, args.theta, tol=args.tol )

		def forward(self, y0, t0):
			# print(super().forward(y0, t0, evolution=True).movedim(0,1).shape)
			# exit()
			return super().forward(y0, t0, evolution=True).movedim(0,2)
			# return super().forward(y0, t0.detach().numpy(), evolution=True)
			# return super().forward(y0, 1, evolution=True)
	########################################################



	#########################################################################################
	#########################################################################################
	# init/train/plot model

	# uncommenting this line will enable debug mode and lead to increased cost and memory leaking
	# torch.autograd.set_detect_anomaly(True)


	paths = ex_setup.create_paths(args)


	model     = ex_setup.load_model(ode_block(), args, _device)
	optimizer = ex_setup.get_optimizer('adam', model, args.lr, wdecay=args.wdecay)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=50, verbose=True, threshold=1.e-4, threshold_mode='rel', cooldown=100, min_lr=1.e-6, eps=1.e-8)
	# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.7, last_epoch=-1)
	# lr_schedule = np.linspace(args.lr, args.lr/100, args.epochs)
	# checkpoint={'epochs':1000, 'name':"models/"+script_name+"/sim_"+str(sim)+'_'+file_name[:]}
	train_obj = utils.training_loop(model, dataset, val_dataset, args.batch, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn, write_hist=False, history=False, checkpoint=None)


	if args.mode=="train":
		assert args.epochs>0, 'number of epochs must be positive'
		writer = SummaryWriter(Path("logs",file_name))
		losses = []

		# save initial model and train
		torch.save( model.state_dict(), Path(paths['checkpoints_0'],file_name) )
		try:
			losses.append(train_obj(args.epochs, writer=writer))
		except:
			raise
		finally:
			torch.save( model.state_dict(), Path(paths['checkpoints'],file_name) )

		writer.close()

	elif args.mode=="plot":
		fig_no = 0

		# sim_name = ("%s_th%4.2f_T%d_data%d_adiv%4.2f_eigs%d_%4.2f"%(args.sigma[0], args.theta, args.T, args.datasize, args.adiv, -1 if math.isnan(args.eigs[0]) else args.eigs[0], -1 if math.isnan(args.eigs[1]) else args.eigs[1] )) #.replace('.','')
		sim_name = ("th%4.2f_T%3.1f_steps%d_data_%d_%d_eigs%4.2f_%4.2f"%(args.theta, args.T, args.steps, args.datasize, args.datasteps, args.eigs[0], args.eigs[1] ))


		Path(paths['output_data'],  sim_name).mkdir(parents=True, exist_ok=True)
		Path(paths['output_images'],sim_name).mkdir(parents=True, exist_ok=True)

		data_sim_output   = Path(paths['output'],'data',  sim_name)
		images_sim_output = Path(paths['output'],'images',sim_name)


		#########################################################################################
		#########################################################################################


		for model_location in [ Path(paths['checkpoints'],file_name) ]:
			###############################################
			# load model
			model = ex_setup.load_model(ode_block(), args, _device, location=model_location)
			model.eval()


			rhs_obj = model.ode.rhs

			with np.printoptions(precision=2, suppress=True):
				print("eigenvalue spread: ", rhs_obj.scales[0].detach().numpy())
				print("eigenvalue shift:  ", rhs_obj.eigshift[0].detach().numpy())
			f = open(Path(paths['output_data'],'eigspread_datasteps%d.csv'%args.datasteps),'ab')
			np.savetxt( f, np.array([args.theta, rhs_obj.scales[0].detach().numpy()]).reshape((1,2)), delimiter=',')
			f.close()


			#########################################################################################
			#########################################################################################
			###############################################
			# prepare data

			t0_train = t0_train.detach().numpy().flatten()
			y0_train = y0_train.detach().numpy().flatten()
			t_train  = t_train.reshape((-1,args.steps+1))
			y_train  = y_train.reshape((-1,args.steps+1))
			t_valid  = t_valid.reshape((-1,args.steps+1))
			y_valid  = y_valid.reshape((-1,args.steps+1))

			t0, y0, t_true, y_true = get_data(args.steps, t0=[0], y0=[0])
			# t0, y0, t_true, y_true = get_data(args.steps)
			t_true = t_true.reshape((-1,args.steps+1))
			y_true = y_true.reshape((-1,args.steps+1))

			y_learned = []
			for i in range(len(t0)):
				y_learned.append( model( y0[i,...], t0[i,...]) )
			y_learned = torch.cat(y_learned).detach().numpy()
			y_learned = y_learned.reshape((-1,args.steps+1))

			np.savetxt( Path(paths['output_data'],'ode_solution.csv'),  np.vstack((t_true[0],y_true[0])).T,    delimiter=',')
			np.savetxt( Path(data_sim_output,'solution.csv'),     np.vstack((t_true[0],y_learned[0])).T, delimiter=',')


			# ###############################################
			# # plot training data
			# fig = plt.figure(fig_no); fig_no += 1
			# steps=1000
			# _, _, t_init, y_init = get_data(1, steps=steps, t0=t0_train, y0=y0_train)
			# t_init = t_init.detach().numpy().reshape((len(t0_train),steps+1))
			# y_init = y_init.detach().numpy().reshape((len(t0_train),steps+1))
			# for i in range(len(t_init)):
			# 	plt.plot(t_init[i],  y_init[i],  '-k', linewidth=1.0)
			# 	plt.plot(t_train[i], y_train[i], 'ok', markersize=5.0)
			# # plt.plot(tt, yy, 'or')
			# plt.xlim(0, args.T)
			# plt.ylim(-0.5, 1.5)
			# plt.gca().axes.xaxis.set_visible(False)
			# plt.gca().axes.yaxis.set_visible(False)
			# plt.gca().axis('off')
			# plt.savefig("out/images/training_data.pdf", bbox_inches='tight', pad_inches=0.0)


			###############################################
			# continuous solution
			fig = plt.figure(fig_no); fig_no += 1

			steps = 5000
			h     = args.T / steps

			# true ode and true piecewise constant ode
			_, _, t_ode, y_true_ode  = get_data(steps, steps=steps, t0=t0.detach().numpy().ravel(), y0=y0.detach().numpy().ravel())
			_, _, _, y_piecewise_ode = get_data(steps, steps=steps, t0=t0.detach().numpy().ravel(), y0=y0.detach().numpy().ravel(), rhs=piecewise_rhs)
			t_ode      = t_ode.detach().numpy().reshape((len(t0),-1))
			y_true_ode = y_true_ode.detach().numpy().reshape((len(t0),-1))
			y_piecewise_ode = y_piecewise_ode.detach().numpy().reshape((len(t0),-1))


			# learned rhs
			y_learned_ode = []
			for i in range(len(t0)):
				y = [y0[i:i+1]]
				for step in range(steps):
					y.append( y[-1] + h*rhs_obj(t0[i]+h*step, y[-1]) )
				y_learned_ode.append( torch.stack(y).movedim(0,2).detach().numpy().reshape((-1,steps+1)) )
			y_learned_ode = np.vstack(y_learned_ode)

			np.savetxt( Path(paths['output_data'],'ode.csv'),           np.vstack((t_ode[0,::50],y_true_ode[0,::50])).T,      delimiter=',')
			np.savetxt( Path(paths['output_data'],'piecewise_ode.csv'), np.vstack((t_ode[0,::50],y_piecewise_ode[0,::50])).T, delimiter=',')
			np.savetxt( Path(data_sim_output,'learned_ode.csv'),              np.vstack((t_ode[0,::50],y_learned_ode[0,::50])).T,   delimiter=',')


			# theta scheme for exact rhs
			for theta in [0.0,0.25,0.5,0.75,1.0]:
				t_theta, y_theta = theta_solver(correct_rhs, args.T, args.steps, theta, t0.detach().numpy().ravel(), y0.detach().numpy().ravel())
				np.savetxt( Path(paths['output_data'],('ode_theta%4.2f'%(theta)).replace('.','')+'.csv'), np.vstack((t_theta[0],y_theta[0])).T, delimiter=',')


			step = args.steps//args.datasteps
			for i in range(len(t_ode)):
				plt.plot(t_ode[i,:], y_true_ode[i,:],       '-r')
				plt.plot(t_ode[i,:], y_piecewise_ode[i,:],  '-g')
				plt.plot(t_ode[i,:], y_learned_ode[i,:],    '-b')
				plt.plot(t_true[i,::step], y_true[i,::step],'or')
			plt.xlim(0, args.T)
			plt.ylim(-0.5, 1.5)
			plt.title('Continuous solutions')


			###############################################
			# plot trajectories
			fig = plt.figure(fig_no); fig_no += 1

			step = args.steps//args.datasteps
			for i in range(len(t_true)):
				plt.plot(t_true[i],        y_learned[i],    '-b')
				plt.plot(t_true[i,::step], y_true[i,::step],'or')
			plt.xlim(0, args.T)
			plt.ylim(-0.5, 1.5)
			plt.title('Trajectories')


			###############################################
			# plot vector field

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


			# exact vector field
			fig = plt.figure(fig_no); fig_no += 1
			plt.quiver(X, Y, np.ones_like(X), UV_ode.reshape(X.shape), angles='xy')
			plt.xlim(0, args.T)
			plt.ylim(-0.5, 1.5)
			plt.gca().axes.xaxis.set_visible(False)
			plt.gca().axes.yaxis.set_visible(False)
			plt.gca().axis('off')
			plt.savefig(Path(paths['output_images'],'vector_field.pdf'), bbox_inches='tight', pad_inches=0.0)

			# plot training data
			steps=1000
			_, _, t_init, y_init = get_data(1, steps=steps, t0=t0_train, y0=y0_train)
			t_init = t_init.detach().numpy().reshape((len(t0_train),steps+1))
			y_init = y_init.detach().numpy().reshape((len(t0_train),steps+1))
			for i in range(len(t_init)):
				plt.plot(t_init[i],  y_init[i],  '-b', linewidth=1.5)
				# plt.plot(t_train[i], y_train[i], 'ok', markersize=5.0)
			plt.plot(t0_train, y0_train, 'or', markersize=5.0)
			# plt.plot(tt, yy, 'or')
			plt.xlim(0, args.T)
			plt.ylim(-0.5, 1.5)
			plt.gca().axes.xaxis.set_visible(False)
			plt.gca().axes.yaxis.set_visible(False)
			plt.gca().axis('off')
			plt.savefig(Path(paths['output_images'],'training_data.pdf'), bbox_inches='tight', pad_inches=0.0)


			# learned vector field
			fig = plt.figure(fig_no); fig_no += 1
			plt.quiver(X, Y, np.ones_like(X), UV.reshape(X.shape), angles='xy')
			plt.xlim(0, args.T)
			plt.ylim(-0.5, 1.5)
			plt.gca().axes.xaxis.set_visible(False)
			plt.gca().axes.yaxis.set_visible(False)
			plt.gca().axis('off')
			plt.savefig(Path(images_sim_output,'vector_field.pdf'), bbox_inches='tight', pad_inches=0.0)

			# plt.show()
			# exit()









