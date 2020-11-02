# import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import argparse
from pathlib import Path

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from collections import OrderedDict

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

import ex_setup
import layers
import utils



if __name__ == '__main__':
	#########################################################################################
	#########################################################################################


	args = ex_setup.parse_args()


	#########################################################################################
	#########################################################################################
	# compose file name

	file_name   = ex_setup.make_name(args)
	script_name = sys.argv[0][:-3]

	# print(layers.ode_solver( ex_setup.rhs_mlp(2, args, final_activation=None) ))
	# print(type(layers.ode_solver))
	# exit()

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

	phi = lambda x: torch.zeros((x.size(0),args.codim))


	def get_data(size, split_data=False, separate_labels=False):
		angle = np.linspace(0,8*np.pi,size)
		rad   = np.linspace(0,4,size)

		# data points
		xy1  =       rad * np.array( [ np.cos(angle), np.sin(angle) ], dtype=float )
		xy2  = (rad+0.5) * np.array( [ np.cos(angle), np.sin(angle) ], dtype=float )
		data = np.hstack((xy1,xy2)).T

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


	# ntrain = args.datasize
	# angle  = np.linspace(0,8*np.pi,ntrain)
	# rad    = np.linspace(0,4,ntrain)

	# # data points
	# x1 = rad * np.array( [ np.cos(angle), np.sin(angle) ] ).astype(np.float)
	# x2 = (rad+0.5) * np.array( [ np.cos(angle), np.sin(angle) ] ).astype(np.float)
	# xtrain  = np.hstack((x1,x2)).T

	# # data labels
	# y1 = np.zeros((ntrain,1)).astype(np.float)
	# y2 = np.ones((ntrain,1)).astype(np.float)
	# ytrain  = np.vstack((y1,y2))

	# x_val = xtrain[0::2,:]
	# y_val = ytrain[0::2,:]
	# # x_val = xtrain[0:2:2,:]
	# # y_val = ytrain[0:2:2,:]
	# xtrain = xtrain[1::2,:]
	# ytrain = ytrain[1::2,:]
	# ytrain = ytrain.ravel()
	# y_val = y_val.ravel()

	xtrain, ytrain, x_val, y_val = get_data(args.datasize, True)

	# print(ytrain)
	# plt.plot(xtrain[:,0],xtrain[:,1],'-')
	# plt.show()
	# exit()

	dataset     = torch.utils.data.TensorDataset( xtrain, ytrain )
	val_dataset = torch.utils.data.TensorDataset( x_val,  y_val  )


	#########################################################################################
	#########################################################################################
	# NN parameters

	# loss_fn         = lambda input, target: (torch.sigmoid(input)[:,0]-target).pow(2).mean()
	# loss_fn         = lambda input, target: (input-target).pow(2).mean()
	loss_fn         = torch.nn.BCEWithLogitsLoss(reduction='mean')
	accuracy_fn     = lambda input, target: (torch.sigmoid(input)>0.5).eq(target).sum().to(dtype=torch.float) / target.nelement()
	optimizer_Adam  = lambda model, lr: torch.optim.Adam(model.parameters(),    lr=lr, weight_decay=args.wdecay)
	optimizer_RMS   = lambda model, lr: torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=args.wdecay)
	optimizer_SGD   = lambda model, lr: torch.optim.SGD(model.parameters(),     lr=lr, weight_decay=args.wdecay, momentum=0.5)
	optimizer_LBFGS = lambda model, lr: torch.optim.LBFGS(model.parameters(),   lr=1., max_iter=100, max_eval=None, tolerance_grad=1.e-9, tolerance_change=1.e-9, history_size=100, line_search_fn='strong_wolfe')
	scheduler_fn    = lambda optimizer: torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=50, verbose=True, threshold=1.e-4, threshold_mode='rel', cooldown=100, min_lr=1.e-6, eps=1.e-8)
	# scheduler_fn    = lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.7, last_epoch=-1)


	#########################################################################################
	#########################################################################################
	# NN model

	########################################################
	# augment original data with additional dimensions
	class augment(torch.nn.Module):
		def __init__(self):
			super().__init__()
			# self.net = torch.nn.Linear(2, 2+args.codim, bias=True)

		def forward(self, x):
			# return self.net(x)
			return torch.cat( (x, phi(x)), 1 )
	########################################################



	########################################################
	class output(torch.nn.Module):
		def __init__(self):
			super().__init__()
			# self.net = torch.nn.Sequential(
			# 	torch.nn.Linear(2+args.codim, 2, bias=False),
			# 	torch.nn.Linear(2, 1, bias=False)
			# 	)

		def forward(self, x):
			# return self.net(x)[:,0]
			# return x[:,-1]
			return x[:,:2]
	########################################################



	########################################################
	class classifier(torch.nn.Module):
		def __init__(self):
			super().__init__()
			self.net = torch.nn.Linear(2+args.codim, 1, bias=False)
			# self.net = torch.nn.Linear(2, 1, bias=False)

		def forward(self, x):
			return self.net(x)[:,0]
	########################################################



	########################################################
	class ode_block(ex_setup.ode_block_base):
		def __init__(self):
			super().__init__(args)
			# self.ode = layers.ode_solver( ex_setup.rhs_mlp(2, args, final_activation=None), args.T, args.T, args.theta, solver='cg', method=args.method, tol=args.tol )
			self.ode = layers.theta_solver( ex_setup.rhs_mlp(2, args), args.T, args.T, args.theta, tol=args.tol )
	########################################################



	########################################################
	class nn_model(torch.nn.Module):
		def __init__(self):
			super().__init__()
			# self.net = torch.nn.Sequential( augment(), ode_block(), output(), classifier() )
			self.net = torch.nn.Sequential( augment(), ode_block(), classifier() )

		def propagate(self, x):
			if isinstance( x, np.ndarray ):
				x = torch.from_numpy(x).float()
			out = {}

			# hidden state
			out['y0'] = self.net[0](x)
			out['yt'] = self.net[1](out['y0'], evolution=True)
			out['yT'] = out['yt'][-1]

			# final labels
			out['val']   = self.net[2](out['yT'])
			out['label'] = torch.sigmoid(out['val'])>0.5

			# # hidden state
			# out['y0'] = self.net[0](x)
			# out['yt'] = self.net[1](out['y0'], evolution=True)
			# out['yT'] = out['yt'][-1]

			# # first two dims
			# out['x0'] = x
			# out['xt'] = [ self.net[2](y) for y in out['yt'] ]
			# out['xT'] = out['xt'][-1]

			# # final labels
			# out['val']   = self.net[3](out['xT'])
			# out['label'] = torch.sigmoid(out['val'])>0.5

			# convert everything to numpy arrays
			for key, value in out.items():
				if isinstance(value, list):
					out[key] = [ val.detach().numpy() for val in value]
				else:
					out[key] = value.detach().numpy()
			return out

		def project_on_plane(self, x):
			if isinstance( x, np.ndarray ):
				x = torch.from_numpy(x).float()
			normal = torch.nn.functional.normalize( self.net[-1](torch.eye(2+args.codim)).reshape(1,-1), dim=1 )
			vector = self.net[-1](torch.ones(1,2+args.codim)).reshape(1,-1)
			planar = torch.nn.functional.normalize( vector - (vector*normal).sum(dim=1, keepdim=True) * normal, dim=1 )
			y0 = (planar*x).sum(dim=1, keepdim=True)
			y1 = (normal*x).sum(dim=1, keepdim=True)
			return torch.cat( (y0, y1), 1 ).detach().numpy()

		def forward(self, x):
			out = self.net(x.requires_grad_(True))
			return out
	########################################################


	def get_model(seed=None):
		if seed is not None: torch.manual_seed(seed)
		mod = nn_model().to(device=_device)
		return mod


	#########################################################################################
	torch.autograd.set_detect_anomaly(True)

	# subdir = ("init" if args.prefix is None else "init_"+args.prefix) if args.mode=="init" else ( "mlp" if args.prefix is None else args.prefix )
	subdir = "mlp" if args.prefix is None else args.prefix

	Path("checkpoints",subdir).mkdir(parents=True, exist_ok=True)
	Path("checkpoints",subdir,"epoch0").mkdir(parents=True, exist_ok=True)
	logdir = Path("logs",subdir,file_name)
	writer = SummaryWriter(logdir) if args.epochs>0 else None

	# if args.mode=="init":
	# 	writer = SummaryWriter(logdir)

	# 	model       = get_model(args.seed)
	# 	optimizer   = optimizer_Adam(model, args.lr)
	# 	scheduler   = None
	# 	regularizer = None

	# 	train_obj = utils.training_loop(model, dataset, val_dataset, args.batch, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn, accuracy_fn=accuracy_fn, regularizer=regularizer,
	# 				writer=writer, write_hist=True, history=False, checkpoint=None)

	# 	# initialize with analytic continuation
	# 	for theta in np.linspace(0,1,101):
	# 		model.apply(lambda m: setattr(m,'theta',theta))
	# 		train_obj(args.epochs)
	# 		torch.save( model.state_dict(), Path("checkpoints",subdir,"%4.2f"%(theta)) )

	# 	if writer is not None:
	# 		writer.close()

	if args.mode!="plot":
		writer = SummaryWriter(logdir)

		losses = []
		for sim in range(1):
			try:
				model     = get_model(args.seed+sim)
				optimizer = optimizer_Adam(model, args.lr)
				scheduler = scheduler_fn(optimizer)

				if args.init=="init":
					missing_keys, unexpected_keys = model.load_state_dict(torch.load(Path("checkpoints/init","%4.2f"%(args.theta)), map_location=_device))
					model.apply(lambda m: setattr(m,'theta',args.theta))
					print(missing_keys, unexpected_keys)
				elif args.init=="cont":
					missing_keys, unexpected_keys = model.load_state_dict(torch.load(Path("checkpoints",subdir,file_name), map_location=_device))
					model.apply(lambda m: setattr(m,'theta',args.theta))
					print(missing_keys, unexpected_keys)

				# save initial model
				torch.save( model.state_dict(), Path("checkpoints",subdir,'epoch0',file_name) )

				# lr_schedule = np.linspace(args.lr, args.lr/100, args.epochs)
				# checkpoint={'epochs':1000, 'name':"models/"+script_name+"/sim_"+str(sim)+'_'+file_name[:]}
				train_obj = utils.training_loop(model, dataset, val_dataset, args.batch, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn, accuracy_fn=accuracy_fn,
					writer=writer, write_hist=True, history=False, checkpoint=None)

				if args.mode=="train":
					# model.apply(lambda m: setattr(m,'theta',0.0))
					# losses.append(train_obj(50))
					# model.apply(lambda m: setattr(m,'theta',args.theta))
					# torch.save( model.state_dict(), Path("checkpoints",subdir,'epoch0',file_name) )
					losses.append(train_obj(args.epochs))
				elif args.mode=="init":
					import re
					# initialize with analytic continuation
					for theta in np.linspace(0,1,51):
						model.apply(lambda m: setattr(m,'theta',theta))
						train_obj(args.epochs if theta>0 else 2000)
						torch.save( model.state_dict(), Path("checkpoints",subdir,re.sub('theta_\d*.\d*', 'theta_'+str(theta), file_name)) )
			except:
				raise
			finally:
				torch.save( model.state_dict(), Path("checkpoints",subdir,file_name) )

		if writer is not None:
			writer.close()

	elif args.mode=="plot":
		from scipy import stats
		from itertools import combinations
		from matplotlib.colors import ListedColormap
		fig_no = 0

		sim_name = ("%s_th%4.2f_T%d_data%d_adiv%4.2f_eigs%d_%4.2f"%(args.sigma[0], args.theta, args.T, args.datasize, args.adiv, -1 if math.isnan(args.eigs[0]) else args.eigs[0], -1 if math.isnan(args.eigs[1]) else args.eigs[1] )).replace('.','')

		# with torch.no_grad():
		subdirs = [subdir] #[subdir+'/epoch0', subdir]
		for subdir in subdirs:
			Path("out",subdir,sim_name).mkdir(parents=True, exist_ok=True)

			data_name = "out/%s/%s/"%(subdir,sim_name)
			# data_name = ("out/%s/%s_th%4.2f_T%d_data%d_adiv%4.2f_ajdiag%4.2f"%(subdir, args.sigma, args.theta, args.T, args.datasize, args.adiv, args.ajdiag)).replace('.','')


			###############################################
			# load model
			model = get_model(args.seed)
			# print(model)
			# exit()
			missing_keys, unexpected_keys = model.load_state_dict(torch.load(Path("checkpoints",subdir,file_name), map_location=cpu))
			model.apply(lambda m: setattr(m,'theta',args.theta))
			print('Load model: ',subdir)
			print('\tmissing_keys:    ', missing_keys)
			print('\tunexpected_keys: ', unexpected_keys)
			model.eval()


			rhs_obj = model.net[1].ode.rhs

			with np.printoptions(precision=2, suppress=True):
				print("scales:   ", rhs_obj.scales[0].detach().numpy())
				print("eigshift: ", rhs_obj.eigshift[0].detach().numpy())


			#########################################################################################
			#########################################################################################
			# title_prefix = str(args.theta)+'_'+str(args.T)+'_'

			###############################################
			# prepare data

			# print(xtest.shape)
			# exit()

			xtest0, xtest1 = get_data(1000, separate_labels=True)

			# propagation of train data through layers
			# ytrain0  = model.net[0](torch.from_numpy(xtrain).float())
			# ytrain1  = model.net[1](ytrain0, evolution=True)
			# ytrainxy = model.net[2].net[0](ytrain1[-1])
			# ytrain2  = model.net[2](ytrain1[-1])


			out_train = model.propagate(xtrain)
			out_test0 = model.propagate(xtest0)
			out_test1 = model.propagate(xtest1)


			# propagation of test data through layers
			# ytest0  = model.net[0](torch.from_numpy(xtest).float())
			# ytest1  = model.net[1](ytest0, evolution=True)
			# ytestxy = model.net[2].net[0](ytest1[-1])
			# ytest2  = model.net[2](ytest1[-1])

			# convert to numpy arrays
			# ytrain0  = ytrain0.detach().numpy()
			# ytrain1  = ytrain1.detach().numpy()
			# ytrain2  = ytrain2.detach().numpy()
			# ytrainxy = ytrainxy.detach().numpy()
			# ytest0   = ytest0.detach().numpy()
			# ytest1   = ytest1.detach().numpy()
			# ytestxy  = ytestxy.detach().numpy()
			# ytest2 = ytest2.detach().numpy()

			# plt.plot(np.sort(ytrain2), '.')
			# plt.show()
			# exit()


			###############################################
			# Nonlinear classifier
			plt.figure(fig_no); fig_no += 1

			plot_step = 0.02

			x_min, x_max = xtrain[:, 0].min() - 1, xtrain[:, 0].max() + 1
			y_min, y_max = xtrain[:, 1].min() - 1, xtrain[:, 1].max() + 1
			# x_min, x_max = -1, 1
			# y_min, y_max = -1, 1
			xx, yy   = np.meshgrid( np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step) )
			xplot    = np.hstack((xx.reshape(-1,1), yy.reshape(-1,1)))
			out_plot = model.propagate(xplot)

			mycmap = plt.cm.get_cmap('Paired', 2)
			color2 = list(mycmap(1))
			color2[-1] = 0.7
			mycmap = ListedColormap([mycmap(0), color2])
			# mycmap = ListedColormap(["skyblue", "lightsalmon"])

			# ytest = (torch.sigmoid(torch.from_numpy(out_test['val']).float()).detach().numpy()>0.5) #* 1 - 0
			plt.contourf(xx, yy, out_plot['label'].reshape(xx.shape), cmap=mycmap)# plt.cm.Paired)
			idx0, idx1 = np.where(ytrain == 0), np.where(ytrain == 1)
			plt.scatter(xtrain[idx0, 0], xtrain[idx0, 1], c='blue', s=20)
			plt.scatter(xtrain[idx1, 0], xtrain[idx1, 1], c='red', s=20)
			plt.plot(xtest0[:,0], xtest0[:,1], '-b')
			plt.plot(xtest1[:,0], xtest1[:,1], '-r')
			plt.gca().axes.xaxis.set_visible(False)
			plt.gca().axes.yaxis.set_visible(False)
			plt.gca().axis('off')
			plt.savefig(data_name+"result.pdf", pad_inches=0.0, bbox_inches='tight')


			###############################################
			# Linear classifier
			plt.figure(fig_no); fig_no += 1

			# slices = list(combinations(np.arange(2+args.codim),2))
			# for ind in slices:
			# 	# Path("out",subdir,"slice%d_%d"%(ind[0],ind[1])).mkdir(parents=True, exist_ok=True)
			# 	plt.cla()
			# 	xT_train = out_train['yT'][:,ind]
			# 	xT_test0 = out_test0['yT'][:,ind]
			# 	xT_test1 = out_test1['yT'][:,ind]

			# 	x_min, x_max = xT_train[:, 0].min() - 1, xT_train[:, 0].max() + 1
			# 	y_min, y_max = xT_train[:, 1].min() - 1, xT_train[:, 1].max() + 1
			# 	xx1, yy1 = np.meshgrid( np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step) )
			# 	xplotlin = np.zeros((xx1.size, 2+args.codim))
			# 	xplotlin[:,ind[0]] = xx1.ravel()
			# 	xplotlin[:,ind[1]] = yy1.ravel()
			# 	yplotlin = model.net[2](torch.from_numpy(xplotlin).float())
			# 	yplotlin = torch.sigmoid(yplotlin).detach().numpy()>0.5

			# 	plt.contourf(xx1, yy1, yplotlin.reshape(xx1.shape), cmap=mycmap)# plt.cm.Paired)
			# 	plt.scatter(xT_train[idx0, 0], xT_train[idx0, 1], c='blue', s=20)
			# 	plt.scatter(xT_train[idx1, 0], xT_train[idx1, 1], c='red',  s=20)
			# 	plt.plot(xT_test0[:,0], xT_test0[:,1], '-b')
			# 	plt.plot(xT_test1[:,0], xT_test1[:,1], '-r')

			# 	plt.gca().axes.xaxis.set_visible(False)
			# 	plt.gca().axes.yaxis.set_visible(False)
			# 	plt.gca().axis('off')
			# 	# plt.savefig(data_name.replace(subdir,subdir+"/slice%d_%d"%(ind[0],ind[1]))+"_result_lin.pdf", pad_inches=0.0, bbox_inches='tight')
			# 	plt.savefig(data_name+"slice%d_%d"%(ind[0],ind[1])+".pdf", pad_inches=0.0, bbox_inches='tight')



			xT_train = model.project_on_plane(out_train['yT'])
			xT_test0 = model.project_on_plane(out_test0['yT'])
			xT_test1 = model.project_on_plane(out_test1['yT'])

			x_min, x_max = xT_train[:, 0].min() - 1, xT_train[:, 0].max() + 1
			y_min, y_max = xT_train[:, 1].min() - 1, xT_train[:, 1].max() + 1
			xx1, yy1 = np.meshgrid( np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step) )
			# xplotlin = np.hstack((xx1.reshape(-1,1), yy1.reshape(-1,1)))
			# yplotlin = model.net[2](torch.from_numpy(xplotlin).float())
			# yplotlin = torch.sigmoid(yplotlin).detach().numpy()>0.5

			# plt.contourf(xx1, yy1, yplotlin.reshape(xx1.shape), cmap=mycmap)# plt.cm.Paired)
			plt.contourf(xx1, yy1, yy1>0, cmap=mycmap)# plt.cm.Paired)
			plt.scatter(xT_train[idx0, 0], xT_train[idx0, 1], c='blue', s=20)
			plt.scatter(xT_train[idx1, 0], xT_train[idx1, 1], c='red', s=20)
			plt.plot(xT_test0[:,0], xT_test0[:,1], '-b')
			plt.plot(xT_test1[:,0], xT_test1[:,1], '-r')
			plt.gca().axes.xaxis.set_visible(False)
			plt.gca().axes.yaxis.set_visible(False)
			plt.gca().axis('off')
			plt.savefig(data_name+"result_lin.pdf", pad_inches=0.0, bbox_inches='tight')


			# x_min, x_max = out_train['xT'][:, 0].min() - 1, out_train['xT'][:, 0].max() + 1
			# y_min, y_max = out_train['xT'][:, 1].min() - 1, out_train['xT'][:, 1].max() + 1
			# xx1, yy1 = np.meshgrid( np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step) )
			# xplotlin = np.hstack((xx1.reshape(-1,1), yy1.reshape(-1,1)))
			# yplotlin = model.net[3](torch.from_numpy(xplotlin).float())
			# yplotlin = torch.sigmoid(yplotlin).detach().numpy()>0.5

			# plt.contourf(xx1, yy1, yplotlin.reshape(xx1.shape), cmap=mycmap)# plt.cm.Paired)
			# # for i, c in zip(range(2), ['blue', 'red']):
			# # 	idx = np.where(ytrain == i)
			# # 	plt.scatter(out_train['xT'][idx, 0], out_train['xT'][idx, 1], c=c, s=20)
			# plt.scatter(out_train['xT'][idx0, 0], out_train['xT'][idx0, 1], c='blue', s=20)
			# plt.scatter(out_train['xT'][idx1, 0], out_train['xT'][idx1, 1], c='red', s=20)
			# plt.plot(out_test0['xT'][:,0], out_test0['xT'][:,1], '-b')
			# plt.plot(out_test1['xT'][:,0], out_test1['xT'][:,1], '-r')
			# plt.gca().axes.xaxis.set_visible(False)
			# plt.gca().axes.yaxis.set_visible(False)
			# plt.gca().axis('off')
			# plt.savefig(data_name+"_result_lin.pdf", pad_inches=0.0, bbox_inches='tight')



			###############################################
			# Evolution of projection
			Path(data_name,"evol").mkdir(parents=True, exist_ok=True)
			plt.figure(fig_no); fig_no += 1

			x_min = y_min =  1e3
			x_max = y_max = -1e3
			for t in range(len(out_train['yt'])):
				yt0 = model.project_on_plane(out_test0['yt'][t])
				yt1 = model.project_on_plane(out_test1['yt'][t])
				x_min, x_max = min(x_min, yt0[:, 0].min()), max(x_max, yt0[:, 0].max())
				y_min, y_max = min(y_min, yt0[:, 1].min()), max(y_max, yt0[:, 1].max())
				x_min, x_max = min(x_min, yt1[:, 0].min()), max(x_max, yt1[:, 0].max())
				y_min, y_max = min(y_min, yt1[:, 1].min()), max(y_max, yt1[:, 1].max())
			x_min -= 1
			y_min -= 1
			x_max += 1
			y_max += 1
			xx1, yy1 = np.meshgrid( np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step) )
			for t in range(len(out_train['yt'])):
				plt.cla()
				plt.contourf(xx1, yy1, yy1>0, cmap=mycmap)
				yt  = model.project_on_plane(out_train['yt'][t])
				yt0 = model.project_on_plane(out_test0['yt'][t])
				yt1 = model.project_on_plane(out_test1['yt'][t])
				# plot training data
				idx0, idx1 = np.where(ytrain == 0), np.where(ytrain == 1)
				plt.scatter(yt[idx0, 0], yt[idx0, 1], c='blue', s=20)
				plt.scatter(yt[idx1, 0], yt[idx1, 1], c='red', s=20)
				# plot test data
				plt.plot(yt0[:,0], yt0[:,1], '-b')
				plt.plot(yt1[:,0], yt1[:,1], '-r')
				plt.xlim(x_min, x_max)
				plt.ylim(y_min, y_max)
				plt.gca().axes.xaxis.set_visible(False)
				plt.gca().axes.yaxis.set_visible(False)
				plt.gca().axis('off')
				# plt.savefig(data_name.replace(subdir,subdir+"/evol"+"/slice%d_%d"%(ind[0],ind[1]))+"_result_%d.pdf"%(t), pad_inches=0.0, bbox_inches='tight')
				plt.savefig(data_name+"evol/%d.pdf"%(t), pad_inches=0.0, bbox_inches='tight')



			###############################################
			# Evolution of the coordinate slices
			plt.figure(fig_no); fig_no += 1


			slices = list(combinations(np.arange(2+args.codim),2))
			for ind in slices:
				Path(data_name,"evol","slice%d_%d"%(ind[0],ind[1])).mkdir(parents=True, exist_ok=True)

				x_min = y_min =  1e3
				x_max = y_max = -1e3
				for t in range(len(out_train['yt'])):
					x_min, x_max = min(x_min, out_test0['yt'][t][:,ind][:, 0].min()), max(x_max, out_test0['yt'][t][:,ind][:, 0].max())
					y_min, y_max = min(y_min, out_test0['yt'][t][:,ind][:, 1].min()), max(y_max, out_test0['yt'][t][:,ind][:, 1].max())
					x_min, x_max = min(x_min, out_test1['yt'][t][:,ind][:, 0].min()), max(x_max, out_test1['yt'][t][:,ind][:, 0].max())
					y_min, y_max = min(y_min, out_test1['yt'][t][:,ind][:, 1].min()), max(y_max, out_test1['yt'][t][:,ind][:, 1].max())
				for t in range(len(out_train['yt'])):
					plt.cla()
					# plot training data
					idx0, idx1 = np.where(ytrain == 0), np.where(ytrain == 1)
					plt.scatter(out_train['yt'][t][:,ind][idx0, 0], out_train['yt'][t][:,ind][idx0, 1], c='blue', s=20)
					plt.scatter(out_train['yt'][t][:,ind][idx1, 0], out_train['yt'][t][:,ind][idx1, 1], c='red', s=20)
					# plot test data
					plt.plot(out_test0['yt'][t][:,ind][:,0], out_test0['yt'][t][:,ind][:,1], '-b')
					plt.plot(out_test1['yt'][t][:,ind][:,0], out_test1['yt'][t][:,ind][:,1], '-r')
					plt.xlim(x_min-1, x_max+1)
					plt.ylim(y_min-1, y_max+1)
					plt.gca().axes.xaxis.set_visible(False)
					plt.gca().axes.yaxis.set_visible(False)
					plt.gca().axis('off')
					# plt.savefig(data_name.replace(subdir,subdir+"/evol"+"/slice%d_%d"%(ind[0],ind[1]))+"_result_%d.pdf"%(t), pad_inches=0.0, bbox_inches='tight')
					plt.savefig(data_name+"evol"+"/slice%d_%d"%(ind[0],ind[1])+"/%d.pdf"%(t), pad_inches=0.0, bbox_inches='tight')

			# x_min = y_min =  1e3
			# x_max = y_max = -1e3
			# for t in range(len(out_train['xt'])):
			# 	x_min, x_max = min(x_min, out_test0['xt'][t][:, 0].min()), max(x_max, out_test0['xt'][t][:, 0].max())
			# 	y_min, y_max = min(y_min, out_test0['xt'][t][:, 1].min()), max(y_max, out_test0['xt'][t][:, 1].max())
			# for t in range(len(out_train['xt'])):
			# 	plt.cla()
			# 	# plot training data
			# 	idx0, idx1 = np.where(ytrain == 0), np.where(ytrain == 1)
			# 	plt.scatter(out_train['xt'][t][idx0, 0], out_train['xt'][t][idx0, 1], c='blue', s=20)
			# 	plt.scatter(out_train['xt'][t][idx1, 0], out_train['xt'][t][idx1, 1], c='red', s=20)
			# 	# plot test data
			# 	plt.plot(out_test0['xt'][t][:,0], out_test0['xt'][t][:,1], '-b')
			# 	plt.plot(out_test1['xt'][t][:,0], out_test1['xt'][t][:,1], '-r')
			# 	plt.xlim(x_min-1, x_max+1)
			# 	plt.ylim(y_min-1, y_max+1)
			# 	plt.gca().axes.xaxis.set_visible(False)
			# 	plt.gca().axes.yaxis.set_visible(False)
			# 	plt.gca().axis('off')
			# 	plt.savefig(data_name.replace(subdir,subdir+"/evol")+"_result_%d.pdf"%(t), pad_inches=0.0, bbox_inches='tight')



			###############################################
			# evaluate spectrum

			xeig, _ = get_data(args.datasize)
			out_eig = model.propagate(xeig)

			spectrum = []
			for t in range(len(out_eig['yt'])):
				spectrum.append( rhs_obj.spectrum(t, out_eig['yt'][t]) )
			spectrum = np.array(spectrum)

			xmax = ymax = 5
			xmax = max( np.amax(np.abs(spectrum[...,0])), xmax );  ymax = max( np.amax(np.abs(spectrum[...,1])), ymax )
			xmax = max(xmax,ymax); ymax = max(xmax,ymax)

			# np.savetxt( data_name+"_eig1.txt", np.sqrt(spectrum_test[-1,1::2,0]**2+spectrum_test[-1,1::2,1]**2), delimiter=',')
			# np.savetxt( data_name+"_eig2.txt", np.sqrt(spectrum_test[-1,::2,0]**2+spectrum_test[-1,::2,1]**2), delimiter=',')

			spectrum = np.concatenate(spectrum)


			# spectrum_train = []
			# spectrum_test  = []
			# ind = np.random.choice(xtest.shape[0],1,replace=True)
			# for t in range(len(out_train['yt'])):
			# 	spectrum_train.append( rhs_obj.spectrum(t, out_train['yt'][t]) )
			# 	spectrum_test.append( rhs_obj.spectrum(t, out_plot['yt'][t][ind,:]) )
			# spectrum_train = np.array(spectrum_train)
			# spectrum_test  = np.array(spectrum_test)

			# xmax = ymax = 5
			# xmax = max( np.amax(np.abs(spectrum_train[...,0])), xmax );  ymax = max( np.amax(np.abs(spectrum_train[...,1])), ymax )
			# xmax = max( np.amax(np.abs(spectrum_test[...,0])),  xmax );  ymax = max( np.amax(np.abs(spectrum_test[...,1])),  ymax )
			# xmax = max(xmax,ymax); ymax = max(xmax,ymax)

			# # np.savetxt( data_name+"_eig1.txt", np.sqrt(spectrum_test[-1,1::2,0]**2+spectrum_test[-1,1::2,1]**2), delimiter=',')
			# # np.savetxt( data_name+"_eig2.txt", np.sqrt(spectrum_test[-1,::2,0]**2+spectrum_test[-1,::2,1]**2), delimiter=',')

			# spectrum_train = np.concatenate(spectrum_train)
			# spectrum_test  = np.concatenate(spectrum_test)

			# kernel = stats.gaussian_kde(spectrum_train.T)
			# eig_alpha = kernel(spectrum_train.T).T
			# eig_alpha = eig_alpha / np.amax(eig_alpha)
			# fig = plt.figure(fig_no); fig_no += 1
			# zz = kernel(xtest.T).reshape(xx.shape)
			# print(zz.shape)
			# plt.imshow(zz, cmap=plt.cm.gist_earth_r, extent=[x_min, x_max, y_min, y_max])
			# plt.show()
			# exit()
			# np.savetxt( data_name+"_eig_train.txt", np.concatenate(spectrum_train), delimiter=',')
			# np.savetxt( data_name+"_eig_test.txt",  np.concatenate(spectrum_test),  delimiter=',')

			###############################################
			# plot stability function
			fig = plt.figure(fig_no); fig_no += 1

			theta_stab = lambda z, theta: abs((1+(1-theta)*z)/(1-theta*z))


			stab_val = theta_stab(spectrum[:,0]+1j*spectrum[:,1], args.theta)
			plt.hist( stab_val, bins='auto', histtype='step', density=True )
			# stab_val_train = theta_stab(spectrum_train[:,0]+1j*spectrum_train[:,1], args.theta)
			# stab_val_test  = theta_stab(spectrum_test[:,0]+1j*spectrum_test[:,1],   args.theta)
			# plt.hist( stab_val_train, bins='auto' )
			# plt.hist( stab_val_test, bins='auto', histtype='step', density=True )
			# val, edges, _ = plt.hist( stab_val_test, bins='auto', histtype='step' )
			# val = val + [val[-1]]

			# np.savetxt( data_name+"_stab_fun_train.txt", np., delimiter=',')

			# np.savetxt( data_name+"_stab_fun_train.txt", stab_val_train, delimiter=',')
			# np.savetxt( data_name+"_stab_fun_test.txt",  stab_val_test, delimiter=',')


			###############################################
			# plot spectrum
			fig = plt.figure(fig_no); fig_no += 1

			def plot_stab(theta, xlim=(-5,5), ylim=(-5,5), fname=None):
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
					levels = [0.14, 0.33, 0.5, 0.71, 0.83, 1.0, 1.2, 1.4, 2, 3, 7]

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
				# ax.add_artist( plt.Circle((0,0), args.maxrho, alpha=1.0, color='c', fill=False, linewidth=1) )
				# ax.add_artist( plt.Circle((0,0), 1, alpha=1.0, color='g', fill=False, linewidth=1) )
				# if args.theta>0: ax.add_artist( plt.Circle((0,0), 1/args.theta, alpha=1.0, color='b', fill=False, linewidth=1.5) )
				if not math.isnan(args.eigs[0]):
					plt.axvline(x=args.eigs[0], color='green')
				if math.isnan(args.eigs[1]) and args.theta>0.2:
					plt.axvline(x=1/args.theta, color='green')
				else:
					plt.axvline(x=args.eigs[1], color='green')


				cs = plt.contour(X,Y,Z, levels=levels, colors='black')
				cs.levels = [nf(val) for val in cs.levels]
				ax.clabel(cs, cs.levels, inline=True, fmt=r'%r', fontsize=15)

				if fname is not None:
					plt.savefig(fname, bbox_inches='tight')

			plot_stab(args.theta, xlim=(-xmax,xmax), ylim=(-ymax,ymax))

			# d = spectrum_test.shape[0] // len(ytrain1)
			# for t in range(len(ytrain1)):
			# 	plt.plot(spectrum_test[(t-1)*d:t*d,0],  spectrum_test[(t-1)*d:t*d,1], 'o', markersize=10-1.8*t, alpha=(t+1)/len(ytrain1)) #, markerfacecolor='none')
			# 	# plt.plot(spectrum_train[(t-1)*d:t*d,0], spectrum_train[(t-1)*d:t*d,1],'ro', markersize=8-0.5*t, alpha=(t+1)/(args.T+1))
			plt.plot(spectrum[:,0], spectrum[:,1], 'bo', markersize=4) #, markerfacecolor='none')
			# plt.plot(spectrum_test[:,0],  spectrum_test[:,1], 'bo', markersize=4) #, markerfacecolor='none')
			# plt.plot(spectrum_train[:,0], spectrum_train[:,1],'ro', markersize=4)
			# for i in range(len(spectrum_train)):
			# 	plt.plot(spectrum_train[i,0], spectrum_train[i,1],'ro', markersize=4, alpha=eig_alpha[i])

			plt.gca().axes.xaxis.set_visible(False)
			plt.gca().axes.yaxis.set_visible(False)
			plt.gca().axis('off')
			plt.savefig(data_name+"spectrum.pdf", bbox_inches='tight', pad_inches=0.0)


			###############################################


			# plt.show()
			# exit()


			# # Evolution of points
			# model[-1].cache_hidden=True
			# ytrained = model(torch.from_numpy(xtrain).float()).detach().numpy()
			# for step in range(args.steps+1):
			# 	plt.figure(fig_no); fig_no += 1
			# 	for i, n, c in zip([-1,1], class_names, plot_colors):
			# 		idx = np.where(ytrain == i)
			# 		plt.scatter(ytrained[step,:,2:][idx, 0], ytrained[step][:,2:][idx, 1], c=c, cmap=plt.cm.Paired, s=20, edgecolor='k', label="Class %s" % n)


			# ##############################################
			# # final approximation

			# ytrained = model.model[-1](torch.from_numpy(xtrain).float()).detach().numpy()
			# print(model.model[1])

			# plt.figure(fig_no); fig_no += 1

			# # Plot the training points
			# for i, n, c in zip(range(2), class_names, plot_colors):
			# 	idx = np.where(ytrain == i)
			# 	plt.scatter(ytrained[idx, 2], ytrained[idx, 3], c=c, cmap=plt.cm.Paired, s=20, label="Class %s" % n)
			# plt.title(r"$\sigma=0.025$", fontdict={'fontsize': 20})
			# # plt.savefig('ex_2_final_T_10_steps_10_theta_000_std_0025.eps', bbox_inches='tight')
			# # exit()







