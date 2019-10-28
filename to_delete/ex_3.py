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
import torchvision
import torch.nn as nn
from torch.autograd import Function
from torch.nn import Module, Linear, ReLU, Sequential, Tanh, Conv2d
from torch.utils.tensorboard import SummaryWriter
from CustomLayers import Antisym, AntisymConv2d, ResNet, TV_regularizer
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
	parser.add_argument("--filters",     type=np.int,   default=1   )
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
	num_filters   = args.filters
	step_size     = args.h
	num_epochs    = args.epochs
	seed          = args.seed
	batch_size    = args.batch_size if args.batch_size > 0 else None

	print("        theta: "+str(theta)        )
	print("learning_rate: "+str(learning_rate))
	print("       layers: "+str(num_layers)   )
	print("      filters: "+str(num_filters)  )
	print("    step_size: "+str(step_size)    )
	print("       levels: "+str(num_levels)   )
	print("       epochs: "+str(num_epochs)   )
	print("   batch_size: "+str(batch_size)   )
	print("  simulations: "+str(simulations)  )



	if args.prefix is not None:
		file_name = 'ex_3/'+args.prefix+'_seed_'+str(seed)+'_theta_'+str(theta)+'_h_'+str(step_size)+'_lr_'+str(learning_rate)+\
					'_levels_'+str(num_levels)+'_layers_'+str(num_layers)+'_filters_'+str(num_filters)+'_batch_'+str(batch_size)
	else:
		file_name = 'ex_3/seed_'+str(seed)+'_theta_'+str(theta)+'_h_'+str(step_size)+'_lr_'+str(learning_rate)+\
					'_levels_'+str(num_levels)+'_layers_'+str(num_layers)+'_filters_'+str(num_filters)+'_batch_'+str(batch_size)



	gpu = torch.device('cuda')
	cpu = torch.device('cpu')
	device = gpu


	#########################################################################################
	#########################################################################################
	# Data


	transform = torchvision.transforms.Compose([ torchvision.transforms.ToTensor() ])


	dataset     = torchvision.datasets.MNIST("./MNIST", train=True,  transform=transform, target_transform=None, download=True)
	val_dataset = torchvision.datasets.MNIST("./MNIST", train=False, transform=transform, target_transform=None, download=True)
	# dataset     = torch.utils.data.Subset(dataset,     np.arange(1000))
	# val_dataset = torch.utils.data.Subset(val_dataset, np.arange(1000))
	# for i in range(5):
	# 	print(dataset[i][0][0])
	# 	plt.imshow(dataset[i][0][0])
	# 	plt.show()
	# exit()
	im_size = dataset[0][0].size()

	# account for unbalanced data
	weight = np.zeros(10,)
	for i in range(len(dataset)):
		weight[dataset[i][1]] += 1
	weight = 1. / weight
	weight = torch.from_numpy( weight/weight.sum() ).float().to(device=device)
	# print(weight)
	# exit()



	#########################################################################################
	#########################################################################################
	# NN params

	loss_fn            = nn.CrossEntropyLoss(weight=weight, reduction='sum')
	accuracy_fn        = lambda y_pred, y: y_pred.argmax(dim=1).eq(y).sum().to(dtype=torch.float) / y.nelement()
	weight_initializer = lambda w: torch.nn.init.xavier_uniform_(w,gain=1)
	bias_initializer   = torch.nn.init.zeros_
	optimizer_fn       = lambda model: torch.optim.Adam(model.parameters(), lr=learning_rate)
	regularizer        = lambda model: model[1].TV_weights(alpha=0.1)


	writer = SummaryWriter('logs/'+file_name)


	#########################################################################################
	#########################################################################################
	# NN model


	def sigma():
		return Sequential( AntisymConv2d(num_filters,num_filters,kernel_size=3,padding=1,bias=True), ReLU() )

	def get_model(theta, writer, seed=None):
		if seed is not None:
			torch.manual_seed(seed)
		model = Sequential(	Conv2d(1,num_filters,kernel_size=3,padding=1,bias=False),
							ResNet(fun=sigma,num_layers=num_layers,h=step_size,theta=theta,writer=writer),
							custom.Flatten(),
							Linear(num_filters*im_size[1]*im_size[2],out_features=10,bias=False)
							)
		custom.initialize( model, weight_initializer, bias_initializer )
		return model


	model     = get_model(theta, writer, seed).to(device)
	optimizer = optimizer_fn(model)
	scheduler = None


	if theta >= 0:
		for l in range(num_levels):
			print([l, model[1].num_layers, model[1].h])
			custom.train(model, optimizer, num_epochs, dataset, val_dataset, batch_size, loss_fn=loss_fn, accuracy_fn=accuracy_fn, regularizer=regularizer, writer=writer, scheduler=scheduler, epoch0=l*num_epochs)
			model     = custom.refine_model(model.cpu()).to(device)
			optimizer = optimizer_fn(model)
	else:
		model[1].set_theta(0.0)
		for l in range(num_levels):
			print([l, model[1].num_layers, model[1].h])
			custom.train(model, optimizer, num_epochs, dataset, val_dataset, batch_size, loss_fn=loss_fn, accuracy_fn=accuracy_fn, regularizer=regularizer, writer=writer, scheduler=scheduler, epoch0=l*num_epochs)
			model     = custom.refine_model(model.cpu()).to(device)
			optimizer = optimizer_fn(model)
		model[1].set_theta(np.abs(theta))
		custom.train(model, optimizer, num_epochs, dataset, val_dataset, batch_size, loss_fn=loss_fn, accuracy_fn=accuracy_fn, regularizer=regularizer, writer=writer, scheduler=scheduler, epoch0=num_levels*num_epochs)


	writer.close()

	# torch.save(model.state_dict(),'./models/'+file_name)