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


	def data_shift(data, shift=0.5):
		size = data.shape[0]
		rad  = 6.5 * ( 1.0 - np.linspace(0,96,size)/104 ).reshape((-1,1))
		y = torch.cat((data*(1-shift/rad), data*(1+shift/rad)))
		return y.float()


	def get_data(size=194 , split_data=False, separate_labels=False):
		np.random.seed(args.seed)
		size  = size//2
		angle = np.linspace(0,96*np.pi/16.0,size)
		rad   = 6.5 * ( 1.0 - np.linspace(0,96,size)/104 )

		# data points
		xy1  =  rad * np.array( [ np.sin(angle), np.cos(angle) ], dtype=float )
		xy2  = -rad * np.array( [ np.sin(angle), np.cos(angle) ], dtype=float )
		data = np.hstack((xy1,xy2)).T
		# data = data + 0.3*np.random.randn(*data.shape)

		# data labels
		lbl1   = np.zeros((size,1), dtype=float)
		lbl2   = np.ones((size,1),  dtype=float)
		labels = np.vstack((lbl1,lbl2))

		if split_data:
			xtrain, ytrain = data[0::2,:], labels[0::2,:]
			xvalid, yvalid = data[1::2,:], labels[1::2,:]
			return torch.from_numpy(xtrain).float(), torch.from_numpy(ytrain.ravel()).float(), torch.from_numpy(xvalid).float(), torch.from_numpy(yvalid.ravel()).float()
		else:
			if separate_labels:
				return torch.from_numpy(data[:size,:]).float(), torch.from_numpy(data[size:,:]).float()
			else:
				return torch.from_numpy(data).float(), torch.from_numpy(labels.ravel()).float()
	# # plot dataset
	# import matplotlib.pyplot as plt
	# xtrain,  ytrain  = get_data(args.datasize, False)
	# xx = data_shift(xtrain[:args.datasize//2,:], 0.3)
	# yy = data_shift(xtrain[args.datasize//2:,:], 0.3)
	# print(xtrain.shape)
	# plt.scatter(xtrain[:args.datasize//2,0], xtrain[:args.datasize//2,1], c='blue', s=20)
	# plt.scatter(xtrain[args.datasize//2:,0], xtrain[args.datasize//2:,1], c='red',  s=20)
	# plt.plot(xx[:,0], xx[:,1], 'g')
	# plt.plot(yy[:,0], yy[:,1], 'brown')
	# plt.axis('equal')
	# plt.show()
	# exit()


	xtrain, ytrain, x_val, y_val = get_data(2*args.datasize, True)

	dataset     = torch.utils.data.TensorDataset( xtrain, ytrain )
	val_dataset = torch.utils.data.TensorDataset( x_val,  y_val  )


	#########################################################################################
	#########################################################################################
	# Loss


	loss_fn     = torch.nn.BCEWithLogitsLoss(reduction='mean')
	accuracy_fn = lambda input, target: (torch.sigmoid(input)>0.5).eq(target).sum().to(dtype=torch.float) / target.nelement()


	#########################################################################################
	#########################################################################################
	# NN model


	########################################################
	# augment original data with additional dimensions
	class augment(torch.nn.Module):
		def __init__(self):
			super().__init__()

		def forward(self, x):
			return torch.cat( (x, torch.zeros((x.size(0),args.codim))), 1 )
			# return torch.cat( (x, x**2), 1 )
	########################################################

	########################################################
	class linear_classifier(torch.nn.Linear):
		def __init__(self):
			super().__init__(2+args.codim, 1, bias=False)

		def forward(self, x):
			return torch.nn.functional.linear(x, self.weight, self.bias).squeeze(1)
	########################################################


	if args.name=="plain-10":
		T = 10
		rhs   = rhs_mlp(2+args.codim, args.width, args.depth, T=T, num_steps=T, activation=args.sigma, power_iters=0, spectral_limits=[-15,15])
		model = torch.nn.Sequential( augment(), theta_solver(rhs, T, T, 0.0), linear_classifier() )
	elif args.name=="1Lip-10":
		T = 10
		rhs   = rhs_mlp(2+args.codim, args.width, args.depth, T=T, num_steps=T, activation=args.sigma, power_iters=1, spectral_limits=[-1,1])
		model = torch.nn.Sequential( augment(), theta_solver(rhs, T, T, 0.0), linear_classifier() )
	elif args.name=="2Lip-10":
		T = 10
		rhs   = rhs_mlp(2+args.codim, args.width, args.depth, T=T, num_steps=T, activation=args.sigma, power_iters=1, spectral_limits=[-2,2])
		model = torch.nn.Sequential( augment(), theta_solver(rhs, T, T, 0.0), linear_classifier() )
	elif args.name=="2Lip-5-1Lip-5":
		T1 = 5
		T2 = 5
		rhs1    = rhs_mlp(2+args.codim, args.width, args.depth, T=T1, num_steps=T1, activation=args.sigma, power_iters=1, spectral_limits=[-1,1])
		rhs2    = rhs_mlp(2+args.codim, args.width, args.depth, T=T2, num_steps=T2, activation=args.sigma, power_iters=1, spectral_limits=[-1,1])
		solver1 = regularized_ode_solver( theta_solver(rhs1, T1, T1, 0.0))
		solver2 = regularized_ode_solver( theta_solver(rhs2, T2, T2, 0.0))
		model   = torch.nn.Sequential( augment(), solver1, solver2, linear_classifier() )
	elif args.name=="10-0":
		T = 10
		rhs    = rhs_mlp(2+args.codim, args.width, args.depth, T=T, num_steps=T, activation=args.sigma, power_iters=1, learn_shift=False)
		solver = regularized_ode_solver( theta_solver(rhs, T, T, 0.0), stability_limits=[-2.0,-1.0,0.0] )
		model  = torch.nn.Sequential( augment(), solver, linear_classifier() )
	elif args.name=="5-5-%.2f"%(args.theta):
		T1 = 5
		T2 = 5
		rhs1    = rhs_mlp(2+args.codim, args.width, args.depth, T=T1, num_steps=T1, activation=args.sigma, power_iters=1, learn_shift=False)
		rhs2    = rhs_mlp(2+args.codim, args.width, args.depth, T=T2, num_steps=T2, activation=args.sigma, power_iters=1, learn_shift=False, spectral_limits=[-1,1])
		solver1 = regularized_ode_solver( theta_solver(rhs1, T1, T1, 0.0),        stability_limits=[-2.0,-1.0, 0.0] )
		solver2 = regularized_ode_solver( theta_solver(rhs2, T2, T2, args.theta),  alpha={'div':0.0}, p=2 )
		model   = torch.nn.Sequential( augment(), solver1, solver2, linear_classifier() )



	#########################################################################################
	#########################################################################################
	# init/train/test model

	# uncommenting this line will enable debug mode and lead to increased cost and memory leaking
	# torch.autograd.set_detect_anomaly(True)


	model = ex_setup.load_model(model, args, _device)


	if args.mode=="train":
		optimizer   = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.alpha['wdecay'])
		# scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=50, verbose=True, threshold=1.e-4, threshold_mode='rel', cooldown=50, min_lr=1.e-6, eps=1.e-8)
		scheduler   = utils.optim.EvenReductionLR(optimizer, lr_reduction=0.1, gamma=0.9, epochs=args.epochs, last_epoch=-1)
		train_model = utils.TrainingLoop(model, loss_fn, dataset, args.batch, optimizer, val_dataset=val_dataset, scheduler=scheduler, accuracy_fn=accuracy_fn, val_freq=10, stat_freq=1)

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
		from matplotlib.colors import ListedColormap
		from implicitresnet.utils.spectral import eigenvalues

		ext = 'pdf'

		mycmap = plt.cm.get_cmap('Paired', 2)
		color2 = list(mycmap(1))
		color2[-1] = 0.7
		mycmap = ListedColormap([mycmap(0), color2])
		# mycmap = ListedColormap(["skyblue", "lightsalmon"])
		plot_step = 0.01


		def project_on_plane(x):
			if isinstance( x, np.ndarray ):
				x = torch.from_numpy(x).float()
			normal = torch.nn.functional.normalize( model[-1](torch.eye(2+args.codim)).reshape(1,-1), dim=1 )
			vector = model[-1](torch.ones(1,2+args.codim)).reshape(1,-1)
			planar = torch.nn.functional.normalize( vector - (vector*normal).sum(dim=1, keepdim=True) * normal, dim=1 )
			# project on the normal and in-plane vectors
			y0 = (planar*x).sum(dim=1, keepdim=True)
			y1 = (normal*x).sum(dim=1, keepdim=True)
			return torch.cat( (y0, y1), 1 ).detach().numpy()

		#########################################################################################
		fig_no = 0

		images_output = "%s/%s"%(Path(paths['output'],'images'), args.name)

		model.eval()


		###############################################
		# prepare data


		xtrain0, xtrain1   = get_data(args.datasize, separate_labels=True)
		xtest0,  xtest1    = get_data(1000,          separate_labels=True)
		# midtest0 = data_shift(xtest0, shift=0.3)
		# midtest1 = data_shift(xtest1, shift=0.3)


		###############################################
		# Nonlinear classifier
		plt.figure(fig_no); fig_no += 1

		x_min, x_max = xtrain.min() - 1, xtrain.max() + 1
		xx, yy = np.meshgrid( np.arange(x_min, x_max, plot_step), np.arange(x_min, x_max, plot_step) )
		xplot = np.hstack((xx.reshape(-1,1), yy.reshape(-1,1)))
		yplot = model(torch.from_numpy(xplot).float())
		# if args.theta==0:
		# 	yplot = model(torch.from_numpy(xplot).float())
		# else:
		# 	# LBFGS can't process large batches. Hence break into chunks. Why?
		# 	yplot = []
		# 	for chunk in np.split(xplot,xplot.shape[0]//200):
		# 		yplot.append( model(torch.from_numpy(chunk).float()) )
		# 	yplot = torch.stack(yplot)
		label = (torch.sigmoid(yplot)>0.5).reshape(xx.shape)

		plt.imshow(label, extent=(x_min,x_max,x_min,x_max), origin='lower', cmap=mycmap)# plt.cm.Paired)
		plt.scatter(xtrain0[:,0], xtrain0[:,1], c='blue', s=5);  #plt.plot(xtest0[:,0], xtest0[:,1], '-b')
		plt.scatter(xtrain1[:,0], xtrain1[:,1], c='red',  s=5);  #plt.plot(xtest1[:,0], xtest1[:,1], '-r')
		ex_setup.savefig(images_output+"_result", format=ext, aspect='equal')


		###############################################
		# Linear classifier
		plt.figure(fig_no); fig_no += 1

		# xT_train  = project_on_plane(model[:-1](xtrain))
		train0_2D = project_on_plane(model[:-1](xtrain0))
		train1_2D = project_on_plane(model[:-1](xtrain1))
		test0_2D  = project_on_plane(model[:-1](xtest0))
		test1_2D  = project_on_plane(model[:-1](xtest1))

		x_min = min(test0_2D.min(), test1_2D.min()) - 1
		x_max = max(test0_2D.max(), test1_2D.max()) + 1
		xx1, yy1 = np.meshgrid( np.arange(x_min, x_max, plot_step), np.arange(x_min, x_max, plot_step) )

		# plt.contourf(xx1, yy1, yy1>0, cmap=mycmap)# plt.cm.Paired)
		plt.imshow(yy1>0, extent=(x_min,x_max,x_min,x_max), origin='lower', cmap=mycmap)# plt.cm.Paired)
		plt.scatter(train0_2D[:, 0], train0_2D[:, 1], c='blue', s=10);  plt.plot(test0_2D[:,0], test0_2D[:,1], '-b')
		plt.scatter(train1_2D[:, 0], train1_2D[:, 1], c='red',  s=10);  plt.plot(test1_2D[:,0], test1_2D[:,1], '-r')
		ex_setup.savefig(images_output+"_result_linear", format=ext, aspect='equal')



		###############################################
		# Evolution of projection
		Path(images_output+"_evolution").mkdir(parents=True, exist_ok=True)
		plt.figure(fig_no); fig_no += 1

		out_train0, out_train1   = model[0](xtrain0),  model[0](xtrain1)
		out_test0,  out_test1    = model[0](xtest0),   model[0](xtest1)
		traj_train0, traj_train1 = [project_on_plane(out_train0)], [project_on_plane(out_train1)]
		traj_test0,  traj_test1  = [project_on_plane(out_test0)],  [project_on_plane(out_test1)]
		for m in model[1:-1]:
			m.ind_out = torch.arange(m.num_steps+1)
			ytrain0, ytrain1 = m(out_train0),  m(out_train1)
			ytest0,  ytest1  = m(out_test0),   m(out_test1)
			m.ind_out = None

			for t in range(1,m.num_steps+1):
				traj_train0.append( project_on_plane(ytrain0[:,t,:]) )
				traj_train1.append( project_on_plane(ytrain1[:,t,:]) )
				traj_test0.append(  project_on_plane(ytest0[:,t,:])  )
				traj_test1.append(  project_on_plane(ytest1[:,t,:])  )
			out_train0, out_train1 = ytrain0[:,-1,:], ytrain1[:,-1,:]
			out_test0,  out_test1  = ytest0[:,-1,:],  ytest1[:,-1,:]
		traj_train0, traj_train1 = np.stack(traj_train0), np.stack(traj_train1)
		traj_test0,  traj_test1  = np.stack(traj_test0),  np.stack(traj_test1)

		x_min = min( traj_train0.min(), traj_train1.min(), traj_test0.min(), traj_test1.min() )
		x_max = max( traj_train0.max(), traj_train1.max(), traj_test0.max(), traj_test1.max() )
		xx1, yy1 = np.meshgrid( np.arange(x_min, x_max, plot_step), np.arange(x_min, x_max, plot_step) )


		# plot evolution
		t0 = 0
		for t in range(len(traj_train0)):
			plt.cla()
			plt.imshow(yy1>0, extent=(x_min,x_max,x_min,x_max), origin='lower', cmap=mycmap)
			plt.scatter(traj_train0[t,:,0], traj_train0[t,:,1], c='blue', s=10);  plt.plot(traj_test0[t,:,0], traj_test0[t,:,1], '-b')
			plt.scatter(traj_train1[t,:,0], traj_train1[t,:,1], c='red',  s=10);  plt.plot(traj_test1[t,:,0], traj_test1[t,:,1], '-r')
			plt.xlim(x_min, x_max)
			plt.ylim(x_min, x_max)
			ex_setup.savefig(images_output+"_evolution/%d"%(t0+t), format=ext, aspect='equal')


		###############################################
		fig = plt.figure(fig_no); fig_no += 1

		out_train = model[0](xtrain)
		for i in range(1,len(model)-1):
			plt.cla()
			# evaluate trjectory
			model[i].ind_out = torch.arange(model[i].num_steps+1)
			ytrain = model[i](out_train)
			out_train = ytrain0[:,-1,:]
			model[i].ind_out = None

			# evaluate spectrum
			spectrum_train = []
			xmax = ymax = 4
			for t in range(model[i].num_steps):
				y = (1-model[i].theta) * ytrain[:,t,:] + model[i].theta * ytrain[:,t+1,:]
				spectrum_train.append( eigenvalues( lambda x: model[i].rhs(t,x), y ) )
				xmax = max(np.amax(np.abs(spectrum_train[-1][:,0])),xmax)
				ymax = max(np.amax(np.abs(spectrum_train[-1][:,1])),ymax)
			spectrum_train = np.concatenate(spectrum_train)

			# plot spectrum
			ex_setup.plot_stab(model[i].theta, xlim=(-xmax,xmax), ylim=(-ymax,ymax))
			plt.plot(spectrum_train[:,0], spectrum_train[:,1],'bo', markersize=4)
			ex_setup.savefig(images_output+"_spectrum_%d"%(i), format='jpg', aspect='equal')



		###############################################
		# model response to shift

		for std in np.linspace(0,0.5,10):
			out_noise = []
			labels    = []

			# evaluate model
			std_test0 = data_shift(xtest0, shift=std)
			std_test1 = data_shift(xtest1, shift=std)
			std_y0 = model(std_test0)
			std_y1 = model(std_test1)

			std_y = torch.cat((std_y0, std_y1))
			label = torch.cat((torch.zeros_like(std_y0), torch.ones_like(std_y1)))

			acc_noise = accuracy_fn(std_y, label)

			# save accuracy
			header = 'theta, accuracy'
			fname = Path(paths['output_data'],('acc_noise_std_%.2f.txt'%(std)))
			if not os.path.exists(fname):
				with open(fname, 'w') as f:
					np.savetxt( f, np.array([args.theta, acc_noise]).reshape((1,-1)), fmt='%.2f', delimiter=',', header=header)
			else:
				with open(fname, 'a') as f:
					np.savetxt( f, np.array([args.theta, acc_noise]).reshape((1,-1)), fmt='%.2f', delimiter=',')