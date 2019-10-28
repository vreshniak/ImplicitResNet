import sys
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import CustomLayers as custom



if __name__ == '__main__':
	#########################################################################################
	#########################################################################################




	parser = argparse.ArgumentParser()
	parser.add_argument("--prefix",      default=None )
	parser.add_argument("--seed",        type=np.int,   default=np.random.randint(0,10000))
	parser.add_argument("--basis",       default=None               )
	parser.add_argument("--update_rule", default=None               )
	parser.add_argument("--init_level",  default=None               )
	parser.add_argument("--theta",       type=np.float, default=0.0 )
	parser.add_argument("--fp_iters",    type=np.int,   default=10  )
	parser.add_argument("--power_iters", type=np.int,   default=0  )
	parser.add_argument("--h",           type=np.float, default=1   )
	parser.add_argument("--lr",          type=np.float, default=0.01)
	parser.add_argument("--layers",      type=np.int,   default=10  )
	parser.add_argument("--filters",     type=np.int,   default=1   )
	parser.add_argument("--epochs",      type=np.int,   default=1000)
	parser.add_argument("--batch",       type=np.int,   default=-1  )
	parser.add_argument("--thresh",      type=np.float, default=-1  )
	parser.add_argument("--hard_thresh", type=np.float, default=1   )
	parser.add_argument("--alpha",       type=np.float, default=0   )
	parser.add_argument("--beta",        type=np.float, default=0   )
	parser.add_argument("--simulations", type=np.int,   default=1   )
	args = parser.parse_args()

	print("  simulations: "+str(args.simulations))
	print("        theta: "+str(args.theta) )
	print("learning_rate: "+str(args.lr)    )
	print("       layers: "+str(args.layers))
	print("      filters: "+str(args.filters) )
	print("    step_size: "+str(args.h)     )
	print("       epochs: "+str(args.epochs))
	print("   batch_size: "+str(args.batch) )
	print("        basis: "+str(args.basis) )
	print("  update_rule: "+str(args.update_rule) )

	file_name = ''
	for arg,value in vars(args).items():
		if value is not None:
			file_name += arg+"_"+str(value)+"__"
	file_name = file_name[:-2]

	script_name = sys.argv[0][:-3]




	#########################################################################################
	#########################################################################################



	gpu = torch.device('cuda')
	cpu = torch.device('cpu')
	device = gpu



	#########################################################################################
	#########################################################################################
	# Data


	np.random.seed(10)
	transform = torchvision.transforms.Compose([ torchvision.transforms.ToTensor() ])


	dataset     = torchvision.datasets.MNIST("./MNIST", train=True,  transform=transform, target_transform=None, download=True)
	val_dataset = torchvision.datasets.MNIST("./MNIST", train=False, transform=transform, target_transform=None, download=True)
	# dataset     = torch.utils.data.Subset(dataset,     np.arange(1000))
	# val_dataset = torch.utils.data.Subset(val_dataset, np.arange(1000))
	im_size = dataset[0][0].size()

	# account for unbalanced data
	weight = np.zeros(10,)
	for i in range(len(dataset)):
		weight[dataset[i][1]] += 1
	weight = 1. / weight
	weight = torch.from_numpy( weight/weight.sum() ).float().to(device=device)


	#########################################################################################
	#########################################################################################
	# NN parameters

	params = {'type':'Conv2d', 'channels':args.filters, 'kernel_size':3}

	loss_fn            = torch.nn.CrossEntropyLoss(weight=weight, reduction='mean')
	accuracy_fn        = lambda y_pred, y: y_pred.argmax(dim=1).eq(y).sum().to(dtype=torch.float) / y.nelement()
	weight_initializer = lambda w: torch.nn.init.xavier_uniform_(w,gain=1)
	bias_initializer   = torch.nn.init.zeros_
	optimizer_fn       = lambda model, lr: torch.optim.SGD(model.parameters(), lr=lr)
	# scheduler_fn       = lambda optimizer: torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
	# scheduler_fn       = lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5, last_epoch=-1)
	scheduler_fn       = lambda optimizer: None
	regularizer_fn     = lambda model: model[1].regularizer(alpha=args.alpha, beta=args.beta)
	regularizer_fn_0   = lambda model: model[1].regularizer(alpha=0.0)


	writer = SummaryWriter('logs1/'+script_name+'/'+file_name)


	#########################################################################################
	#########################################################################################
	# NN model

	def get_model(seed=None):
		if seed is not None: torch.manual_seed(seed)
		model = torch.nn.Sequential(
			torch.nn.Conv2d(1,args.filters,kernel_size=3,padding=1,bias=True),
			custom.ResNet(params=params, num_layers=args.layers, theta=args.theta, fixed_point_iters=args.fp_iters, power_iters=args.power_iters,
				h=args.h, basis=args.basis, writer=writer, init_level=args.init_level),
			torch.nn.Flatten(),
			torch.nn.Linear(args.filters*im_size[1]*im_size[2],out_features=10,bias=True) )
		custom.initialize( model, weight_initializer, bias_initializer )
		return model

	# torch.autograd.set_detect_anomaly(True)
	losses = []; W = []; b = []
	for sim in range(args.simulations):
		model     = get_model(args.seed+sim).to(device=device)
		optimizer = optimizer_fn(model, args.lr)
		training_loop = custom.training_loop(model, dataset, val_dataset, args.batch, optimizer=optimizer,
			scheduler=scheduler_fn(optimizer), loss_fn=loss_fn, accuracy_fn=accuracy_fn, regularizer=regularizer_fn, writer=writer, write_hist=True, history=True)

		losses.append(training_loop(args.epochs))
		W1, b1 = model[1].plot_weights(); W.append(W1); b.append(b1)
	losses = np.hstack(losses)


	# np.savetxt('./csv/losses_'+file_name, losses)
	# np.savetxt('./csv/W_'+file_name, np.array(W))
	# np.savetxt('./csv/b_'+file_name, np.array(b))

	writer.close()
	# torch.save(model.state_dict(),'./models1/ex_Haar/'+file_name)
