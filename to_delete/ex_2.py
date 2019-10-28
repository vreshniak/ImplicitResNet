import time
import argparse
# import multiprocessing as omp
import torch.multiprocessing as omp

import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla
import scipy.optimize as opt
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import Module, Linear, ReLU, Sequential, Tanh
from torch.utils.tensorboard import SummaryWriter
from CustomLayers import Antisym, ResNet, TV_regularizer
import CustomLayers as custom



if __name__ == '__main__':
	#########################################################################################
	#########################################################################################




	parser = argparse.ArgumentParser()
	parser.add_argument("--theta",       type=np.float, default=0.0 )
	parser.add_argument("--lr",          type=np.float, default=0.01)
	parser.add_argument("--simulations", type=np.int,   default=1   )
	parser.add_argument("--layers",      type=np.int,   default=10  )
	parser.add_argument("--levels",      type=np.int,   default=1   )
	parser.add_argument("--nodes",       type=np.int,   default=1   )
	parser.add_argument("--h",           type=np.float, default=1   )
	parser.add_argument("--epochs",      type=np.int,   default=1000)
	parser.add_argument("--batch_size",  type=np.int,   default=-1  )
	parser.add_argument("--seed",        type=np.int,   default=np.random.randint(0,10000)  )
	parser.add_argument("--prefix",      default=None  )
	args = parser.parse_args()

	theta         = args.theta
	learning_rate = args.lr
	simulations   = args.simulations
	num_layers    = args.layers
	num_levels    = args.levels
	num_nodes     = args.nodes
	step_size     = args.h
	num_epochs    = args.epochs
	seed          = args.seed
	batch_size    = args.batch_size if args.batch_size > 0 else None

	print("        theta: "+str(theta)        )
	print("learning_rate: "+str(learning_rate))
	print("       layers: "+str(num_layers)   )
	print("        nodes: "+str(num_nodes)  )
	print("    step_size: "+str(step_size)    )
	print("       levels: "+str(num_levels)   )
	print("       epochs: "+str(num_epochs)   )
	print("   batch_size: "+str(batch_size)   )
	print("  simulations: "+str(simulations)  )



	if args.prefix is not None:
		file_name = 'ex_2/'+args.prefix+'_seed_'+str(seed)+'_theta_'+str(theta)+'_h_'+str(step_size)+'_lr_'+str(learning_rate)+\
					'_levels_'+str(num_levels)+'_layers_'+str(num_layers)+'_nodes_'+str(num_nodes)+'_batch_'+str(batch_size)
	else:
		file_name = 'ex_2/seed_'+str(seed)+'_theta_'+str(theta)+'_h_'+str(step_size)+'_lr_'+str(learning_rate)+\
					'_levels_'+str(num_levels)+'_layers_'+str(num_layers)+'_nodes_'+str(num_nodes)+'_batch_'+str(batch_size)



	gpu = torch.device('cuda')
	cpu = torch.device('cpu')
	device = cpu





	#########################################################################################
	#########################################################################################
	# Data


	np.random.seed(10)


	num_points = 100
	angle = np.linspace(0,4*np.pi,num_points)
	rad   = np.linspace(0,1,num_points)

	# data points
	x1 = rad * np.array( [ np.cos(angle), np.sin(angle) ] ).astype(np.float)
	x2 = (rad+0.2) * np.array( [ np.cos(angle), np.sin(angle) ] ).astype(np.float)
	x  = np.hstack((x1,x2)).T

	# data labels
	y1 = np.zeros((num_points,1)).astype(np.float)
	y2 = np.ones((num_points,1)).astype(np.float)
	y  = np.vstack((y1,y2))

	x_val = x[0::2,:]
	y_val = y[0::2,:]
	x = x[1::2,:]
	y = y[1::2,:]

	dataset     = torch.utils.data.TensorDataset( torch.from_numpy(x).float(),     torch.from_numpy(y).float()     )
	val_dataset = torch.utils.data.TensorDataset( torch.from_numpy(x_val).float(), torch.from_numpy(y_val).float() )


	#########################################################################################
	#########################################################################################
	# NN params

	loss_fn            = nn.BCELoss(reduction='sum')
	accuracy_fn        = lambda y_pred, y: ((y_pred.detach().numpy()>0.5)==y.numpy()).sum() / y.numpy().size
	weight_initializer = lambda w: torch.nn.init.xavier_uniform_(w,gain=1.0)
	bias_initializer   = torch.nn.init.zeros_
	optimizer_fn       = lambda model: torch.optim.Adam(model.parameters(), lr=learning_rate)
	regularizer        = lambda model: model[1].TV_weights(alpha=0.1)


	writer = SummaryWriter('logs/'+file_name)

	#########################################################################################
	#########################################################################################
	# NN model



	def sigma():
		return Sequential( Antisym(num_nodes,num_nodes,bias=True), Tanh() )

	def get_model(theta, writer, seed=None):
		if seed is not None:
			torch.manual_seed(seed)
		model = Sequential(	Linear(2,out_features=num_nodes,bias=True),
							ResNet(fun=sigma,num_layers=num_layers,h=step_size,theta=theta,writer=writer),
							Linear(num_nodes,out_features=1,bias=True),
							nn.Sigmoid()
							)
		custom.initialize( model, weight_initializer, bias_initializer )
		return model


	model     = get_model(theta, writer, seed )
	optimizer = optimizer_fn(model)
	scheduler = None


	for l in range(num_levels):
		print([model[1].num_layers, model[1].h])
		custom.train(model, optimizer, num_epochs, dataset, val_dataset, batch_size, loss_fn=loss_fn, accuracy_fn=accuracy_fn, regularizer=regularizer, writer=writer, scheduler=scheduler, epoch0=(l-1)*num_epochs)
		model     = custom.refine_model(model.cpu()).to(device)
		optimizer = optimizer_fn(model)


	# if theta >= 0:
	# 	custom.train(model, optimizer, num_epochs, dataset, val_dataset, batch_size, loss_fn=loss_fn, accuracy_fn=accuracy_fn, regularizer=regularizer, writer=writer, scheduler=scheduler)
	# else:
	# 	model[1].set_theta(0.0)
	# 	custom.train(model, optimizer, num_epochs//2, dataset, val_dataset, batch_size, loss_fn=loss_fn, accuracy_fn=accuracy_fn, regularizer=regularizer, writer=writer, scheduler=scheduler)
	# 	model[1].set_theta(np.abs(theta))
	# 	custom.train(model, optimizer, num_epochs//2, dataset, val_dataset, batch_size, loss_fn=loss_fn, accuracy_fn=accuracy_fn, regularizer=regularizer, writer=writer, scheduler=scheduler, epoch0=num_epochs//2)


	writer.close()

	# torch.save(model.state_dict(),'./models/'+file_name)


	#########################################################################################
	#########################################################################################


	# xx, yy = np.meshgrid( np.linspace(-1.5,1.5,200), np.linspace(-1.5,1.5,200) )
	# x_test = np.array([xx.ravel(),yy.ravel()]).T
	# labels = model(torch.from_numpy(x_test).float()).detach().numpy()>0.5
	# plt.contourf(xx, yy, labels.reshape(xx.shape))
	# plt.plot(x1[0,:],x1[1,:],'o')
	# plt.plot(x2[0,:],x2[1,:],'o')
	# plt.show()
