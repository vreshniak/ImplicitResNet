import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch.autograd.profiler as profiler

# import torchattacks
from skimage.util import random_noise

from implicitresnet import utils, theta_solver, regularized_ode_solver, rhs_conv2d, rhs_mlp
from implicitresnet.models.misc import MLP
from implicitresnet.utils import calc
import ex_setup



if __name__ == '__main__':
	#########################################################################################
	#########################################################################################


	args  = ex_setup.parse_args()
	paths = ex_setup.create_paths(args)
	if args.seed is not None: torch.manual_seed(args.seed)
	torch.backends.cudnn.deterministic = True

	# skip run if given stability limit is too small
	# if args.stabspread is not None:
	# 	if args.stabspread[0] < (ex_setup.theta_stability_fun(args.theta,-20.0)-0.05):
	# 		exit("Skip run for theta=%.2f with stabspread[0]=%.2f"%(args.theta, args.stabspread[0]))

	#########################################################################################
	#########################################################################################
	# compose file name


	runname   = ex_setup.make_name(args)
	# script_name = sys.argv[0][:-3]


	#########################################################################################
	#########################################################################################


	# cpu because model is too small to get speedup using gpu
	_device = ex_setup._cpu if torch.cuda.is_available() else ex_setup._cpu
	_dtype  = ex_setup._dtype


	#########################################################################################
	#########################################################################################
	# Data

	class AddGaussianNoise(object):
		'''
		Source: https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745/2
		'''
		def __init__(self, mean=0., std=1.):
			self.std = std
			self.mean = mean

		def __call__(self, tensor):
			return tensor + torch.randn(tensor.size()) * self.std + self.mean

		def __repr__(self):
			return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


	weight = None
	if args.dataset=='MNIST':
		transform   = torchvision.transforms.Compose([ torchvision.transforms.ToTensor(), torchvision.transforms.Pad(2, fill=0, padding_mode='constant') ]) #, torchvision.transforms.Normalize((0.1307,), (0.3081,))  ])
		dataset     = torchvision.datasets.MNIST("./MNIST", train=True,  transform=transform, target_transform=None, download=True)
		val_dataset = torchvision.datasets.MNIST("./MNIST", train=False, transform=transform, target_transform=None, download=True)
	elif args.dataset=='FashionMNIST':
		transform   = torchvision.transforms.Compose([ torchvision.transforms.ToTensor(), torchvision.transforms.Pad(2, fill=0, padding_mode='constant') ])
		dataset     = torchvision.datasets.FashionMNIST("./FashionMNIST", train=True,  transform=transform, target_transform=None, download=True)
		val_dataset = torchvision.datasets.FashionMNIST("./FashionMNIST", train=False, transform=transform, target_transform=None, download=True)
	elif args.dataset=='CIFAR10':
		transform   = torchvision.transforms.Compose([ torchvision.transforms.ToTensor(), torchvision.transforms.Grayscale() ])
		dataset     = torchvision.datasets.CIFAR10("./CIFAR10", train=True,  transform=transform, target_transform=None, download=True)
		val_dataset = torchvision.datasets.CIFAR10("./CIFAR10", train=False, transform=transform, target_transform=None, download=True)

	# in this case, redefine validation dataset as a traininig data corrupted with Gaussian noise
	if args.datasize>0:
		if args.dataset=='MNIST':
			val_transform = torchvision.transforms.Compose([ torchvision.transforms.ToTensor(), torchvision.transforms.Pad(2, fill=0, padding_mode='constant'), AddGaussianNoise(0., 0.3) ])
			val_dataset   = torchvision.datasets.MNIST("./MNIST", train=True, transform=val_transform, target_transform=None, download=True)
		elif args.dataset=='FashionMNIST':
			val_transform = torchvision.transforms.Compose([ torchvision.transforms.ToTensor(), torchvision.transforms.Pad(2, fill=0, padding_mode='constant'), AddGaussianNoise(0., 0.3) ])
			val_dataset   = torchvision.datasets.FashionMNIST("./FashionMNIST", train=True, transform=val_transform, target_transform=None, download=True)
		elif args.dataset=='CIFAR10':
			val_transform = torchvision.transforms.Compose([ torchvision.transforms.ToTensor(), torchvision.transforms.Grayscale(), AddGaussianNoise(0., 0.3) ])
			val_dataset   = torchvision.datasets.CIFAR10("./CIFAR10", train=True, transform=val_transform, target_transform=None, download=True)

		# subsets
		dataset     = torch.utils.data.Subset(dataset,     [i for i in range(args.datasize)])
		val_dataset = torch.utils.data.Subset(val_dataset, [i for i in range(args.datasize)])

		# for dat in dataset:
		# 	print(dat[0])
		# 	exit()

		# account for the unbalanced data
		weight = np.zeros(10,)
		for i in range(len(dataset)):
			weight[dataset[i][1]] += 1
		weight = 1. / weight
		weight = torch.from_numpy( weight/weight.sum() ).float().to(device=_device)


	# if args.dataset=='MNIST':
	# 	transform_train = torchvision.transforms.Compose([ torchvision.transforms.ToTensor(), torchvision.transforms.Pad(2, fill=0, padding_mode='constant') ]) #, torchvision.transforms.Normalize((0.1307,), (0.3081,))  ])
	# 	val_transform = transform_train if args.datasize<0 else torchvision.transforms.Compose([ torchvision.transforms.ToTensor(), torchvision.transforms.Pad(2, fill=0, padding_mode='constant'), AddGaussianNoise(0., 0.3) ])

	# 	# original data
	# 	dataset     = torchvision.datasets.MNIST("./MNIST", train=True,  transform=transform_train, target_transform=None, download=True)
	# 	val_dataset = torchvision.datasets.MNIST("./MNIST", train=False, transform=val_transform, target_transform=None, download=True)

	# 	# # data normalization
	# 	# for sample in torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False):
	# 	# 	data_mean, data_std = sample[0].float().mean(), sample[0].float().std()

	# 	# move data to device
	# 	# dataset     = torch.utils.data.TensorDataset(     dataset.data.unsqueeze(1).to(_device, dtype=_dtype)/255.,     dataset.targets.to(_device) )
	# 	# val_dataset = torch.utils.data.TensorDataset( val_dataset.data.unsqueeze(1).to(_device, dtype=_dtype)/255., val_dataset.targets.to(_device) )
	# 	# subset of data
	# 	# weight = None
	# 	# if args.datasize>0:
	# 	# 	dataset     = torch.utils.data.Subset(dataset,     [i for i in range(args.datasize)])
	# 	# 	val_dataset = torch.utils.data.Subset(val_dataset, [0])

	# 	# 	# account for the unbalanced data
	# 	# 	weight = np.zeros(10,)
	# 	# 	for i in range(len(dataset)):
	# 	# 		weight[dataset[i][1]] += 1
	# 	# 		# weight[dataset[i][1].item()] += 1
	# 	# 	weight = 1. / weight
	# 	# 	weight = torch.from_numpy( weight/weight.sum() ).float().to(device=_device)

	# elif args.dataset=='CIFAR10':
	# 	transform_train = torchvision.transforms.Compose([ torchvision.transforms.ToTensor(), torchvision.transforms.Grayscale() ])
	# 	val_transform = transform_train if args.datasize<0 else torchvision.transforms.Compose([ torchvision.transforms.ToTensor(), torchvision.transforms.Grayscale(), AddGaussianNoise(0., 0.1) ])

	# 	# original data
	# 	dataset     = torchvision.datasets.CIFAR10("./CIFAR10", train=True,  transform=transform_train, target_transform=None, download=True)
	# 	val_dataset = torchvision.datasets.CIFAR10("./CIFAR10", train=False, transform=val_transform, target_transform=None, download=True)

	# 	# # data normalization
	# 	# for sample in torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False):
	# 	# 	data_mean, data_std = sample[0].float().mean(), sample[0].float().std()

	# # subset of data
	# weight = None
	# if args.datasize>0:
	# 	dataset     = torch.utils.data.Subset(dataset,     [i for i in range(args.datasize)])
	# 	val_dataset = torch.utils.data.Subset(val_dataset, [i for i in range(args.datasize)])

	# 	# account for the unbalanced data
	# 	weight = np.zeros(10,)
	# 	for i in range(len(dataset)):
	# 		weight[dataset[i][1]] += 1
	# 		#weight[dataset[i][1].item()] += 1
	# 	weight = 1. / weight
	# 	weight = torch.from_numpy( weight/weight.sum() ).float().to(device=_device)


	#########################################################################################
	#########################################################################################
	# Loss function


	loss_fn     = torch.nn.CrossEntropyLoss(weight=weight, reduction='mean')
	accuracy_fn = lambda y_pred, y: y_pred.argmax(dim=1).eq(y).sum().float() / y.nelement()


	#########################################################################################
	#########################################################################################
	# NN model


	########################################################
	def ode_block(im_shape, kernel_size=3, theta=0.00, alpha={}):
		# eig_center = [ min(20, max(-20, ex_setup.theta_inv_stability_fun(theta, i))) for i in stab_center ]
		rhs = rhs_conv2d(im_shape, kernel_size=kernel_size, depth=2, bias=True, activation=args.sigma, power_iters=args.piters,
			T=1, num_steps=1, mode=args.rhs_mode, center=args.center, radius=args.radius, stabmin=args.stabmin, stabmax=args.stabmax, theta=theta)
			# power_iters=args.piters, spectral_limits=args.eiglim, learn_spectral_limits=args.learn_limits, learn_spectral_circle=args.learn_circle)
		solver = theta_solver( rhs, args.T, args.steps, theta, tol=args.tol )
		return regularized_ode_solver( solver, alpha, p=1, quadrature='rectangle', collect_rhs_stat=True )
		# return regularized_ode_solver( solver, alpha, stability_limits, args.mciters, p=1, stability_target=args.stabval, collect_rhs_stat=True )
	########################################################
	# def theta_inv_stability_fun(theta, y):
	# 	y = max(y, 1 - 1.0/(theta+1.e-12) + 1.e-6)
	# 	return (1-y) / ((1-y)*theta-1)
	# def theta_stability_fun(theta, x):
	# 	return (1+(1-theta)*x) / (1-theta*x)
	# for theta in [0.00, 0.25,0.50,0.75]:
	# 	print(theta,theta_stability_fun(theta,2*theta_inv_stability_fun(theta,0.0)))
	# exit()

	########################################################
	class Normalize(torch.nn.Module) :
		def __init__(self) :
			super().__init__()
		def forward(self, input):
			if args.dataset=='CIFAR10':
				return (input - 0.4808) / 0.2392
			elif args.dataset=='MNIST':
				return (input - 0.1307) / 0.3081
			elif args.dataset=='FashionMNIST':
				return (input - 0.2859) / 0.3530
			# return (input - data_mean) / data_std
	########################################################


	########################################################
	if args.model is None: args.model='123'
	# ch = args.codim + 1
	ch = 8 if args.dataset=='MNIST' else 16
	model = torch.nn.Sequential(
		######################################
		# Normalization
		# [0] 1x32x32
		Normalize(),
		######################################
		# Feature extraction
		# [1] chx16x16
		# torch.nn.Dropout(p=0.3), # if args.datasize<0 else torch.nn.Identity(),
		# [2] chx16x16
		# torch.nn.BatchNorm2d(1),
		# torch.nn.ReLU(),
		torch.nn.Conv2d(1, ch, 7, stride=2, padding=3, bias=True),
		# torch.nn.ReLU(),
		# torch.nn.BatchNorm2d(ch),
		# [3] chx16x16
		ode_block(im_shape=(ch,14,14), theta=args.theta, alpha=args.alpha) if args.model=='1' or args.model=='123' else torch.nn.Identity(),
		# torch.nn.ReLU(),
		######################################
		# [4] 2chx8x8
		# torch.nn.Dropout(p=0.3), # if args.datasize<0 else torch.nn.Identity(),
		# [5] 2chx8x8
		# torch.nn.BatchNorm2d(ch),
		# torch.nn.ReLU(),
		torch.nn.Conv2d(ch, 2*ch, 3, stride=2, padding=1, bias=True),
		# torch.nn.ReLU(),
		# torch.nn.BatchNorm2d(2*ch),
		# [6] 2chx8x8
		ode_block(im_shape=(2*ch,7,7), theta=args.theta, alpha=args.alpha) if args.model=='2' or args.model=='123' else torch.nn.Identity(),
		# torch.nn.ReLU(),
		######################################
		# [7] 4chx4x4
		# torch.nn.Dropout(p=0.3), # if args.datasize<0 else torch.nn.Identity(),
		# [8] 4chx4x4
		# torch.nn.BatchNorm2d(2*ch),
		# torch.nn.ReLU(),
		torch.nn.Conv2d(2*ch, 4*ch, 4, stride=2, padding=1, bias=True),
		# torch.nn.ReLU(),
		# torch.nn.BatchNorm2d(4*ch),
		# [9] 4chx4x4
		ode_block(im_shape=(4*ch,3,3), theta=args.theta, alpha=args.alpha) if args.model=='3' or args.model=='123' else torch.nn.Identity(),
		# torch.nn.ReLU(),
		######################################
		# Classifier
		# [10] 4chx1x1
		torch.nn.AvgPool2d(3),
		# [11] 4ch
		torch.nn.Flatten(),
		# [12] 10
		torch.nn.Linear(in_features=4*ch, out_features=10, bias=True),
	)
	########################################################


	#########################################################################################
	#########################################################################################
	# train/test/process model

	# uncommenting this line will enable debug mode and lead to increased cost and memory leaking
	# torch.autograd.set_detect_anomaly(True)

	model = ex_setup.load_model(model, args, _device)

	# attacks = [ torchattacks.FGSM(model, eps=(0.1*i)) for i in range(1,3) ] #+ [ torchattacks.GN(model, sigma=0.1*i) for i in range(1,4) ]
	# attacks = [ torchattacks.GN(model, sigma=(0.1*i)) for i in range(1,4) ]
	# def attack(x,y):
		# adv_x = attacks[np.random.randint(len(attacks))](x,y)
		# dx = (adv_x - x ) / 0.3
		# return torch.cat([ x + 0.1*i*dx for i in range(1,4) ])
		# return attacks[np.random.randint(len(attacks))](x,y)
		# return torch.cat([ attk(x,y) for attk in attacks ])

	# def attack(x,y):
	# 	eps = 0.3
	# 	delta = torch.zeros_like(x, requires_grad=True)
	# 	L  = loss_fn(model(x+delta),y)
	# 	dx = torch.autograd.grad(L, delta, create_graph=False, retain_graph=False, only_inputs=True)[0]
	# 	return (x+eps*dx.sign()).clamp(0,1).detach()


	dataloaders = {
		'train': torch.utils.data.DataLoader(dataset,     batch_size=max(args.batch,args.batch), shuffle=False),
		'valid': torch.utils.data.DataLoader(val_dataset, batch_size=max(args.batch,args.batch), shuffle=False)
		}
	attacks = {
		'GNclip': lambda x,y,eps: torch.from_numpy(random_noise(x.detach().numpy(), mode='gaussian', clip=True,  var=eps**2, seed=101)).float(),
		'GN':     lambda x,y,eps: torch.from_numpy(random_noise(x.detach().numpy(), mode='gaussian', clip=False, var=eps**2, seed=101)).float(),
		'SP':     lambda x,y,eps: torch.from_numpy(random_noise(x.detach().numpy(), mode='s&p',      clip=True,  amount=eps, seed=101)).float()
		# 'FGSM'
	}


	eps = 0.3
	if args.datanoise=='GN':
		attack = lambda x,y: torch.clamp(x + eps*torch.rand(1)[0]*torch.randn_like(x), 0., 1.)
		args.batch = args.batch // 2
	else:
		attack = None

	# import pickle
	# with open(Path(paths['output'],'name2accuracy2'),'rb') as f:
	# 	name2accuracy = pickle.load(f)
	# for attk_name, attk in attacks.items():
	# 	for key, loader in dataloaders.items():
	# 		print(" ")
	# 		for eps in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
	# 			name1 = '%s_%s_noise_%s_eps_%.1f'%(key, runname, attk_name, eps)
	# 			name2 = '%s%s%.1f-%s'%(key,attk_name,eps,runname)
	# 			print(name1, name2)
	# 			ex_setup.pickle_data( name2, name2accuracy[name1], Path(paths['output'],'name2accuracy') )
	# exit()


	if args.mode=="train":
		optimizer = torch.optim.Adam(model.parameters(),  lr=args.lr, weight_decay=args.alpha['wdecay'])
		# scheduler = utils.optim.EvenReductionLR(optimizer, lr_reduction=0.05, gamma=0.7, epochs=args.epochs, last_epoch=-1)
		# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, threshold=1.e-4, threshold_mode='rel', cooldown=5, min_lr=0, eps=1e-08, verbose=False)
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5, last_epoch=-1)
		# scheduler = None
		checkpoints = None
		# checkpoints = {'dir': paths['checkpoints'], 'each_nth': 5, 'name': runname}
		train_model = utils.TrainingLoop(model, loss_fn, dataset, args.batch, optimizer, data_augmentation=attack, val_dataset=val_dataset, val_batch_size=args.batch, scheduler=scheduler, accuracy_fn=accuracy_fn, val_freq=1, stat_freq=1, hist_freq=0, checkpoints=checkpoints, tol=1.e-4, min_epochs=0)

		# with profiler.profile(record_shapes=True,use_cuda=True) as prof:
		writer = SummaryWriter(Path(paths['logs'],runname))

		# save initial model and train
		torch.save( model.state_dict(), Path(paths['chkp_init'],runname) )
		try:
			# if args.pre_epochs>0:
			# 	for m in model.modules():
			# 		if isinstance(m,theta_solver):
			# 			m.tol = args.tol
			# 			for key in m.r_alpha.keys(): m.r_alpha[key] = 0.0
			# 	converged = train_model(args.pre_epochs, writer=writer)
			# 	for m in model.modules():
			# 		if isinstance(m,theta_solver):
			# 			m.tol = args.tol
			# 			for key in m.r_alpha.keys(): m.r_alpha[key] = 0.0

			# alpha = args.alpha.copy()
			# initial tolerance
			ini_tol = 1.e-4
			for m in model.modules():
				if isinstance(m,theta_solver):
					m.tol = ini_tol

			epoch = 0
			while epoch < args.epochs:
				epoch = epoch + 20
				converged = train_model(20, writer=writer)
				if converged: break
				for m in model.modules():
					if isinstance(m,theta_solver):
						# anneal tolerance
						m.tol = ini_tol * (args.tol/ini_tol)**(epoch/args.epochs)
						# if (hasattr(m,'reduce') and m.reduce) or epoch>=100:
						# if epoch>=100:
						# 	# m.reduce = False
						# 	# m.regularizer = {}
						# 	for key in m.alpha.keys():
						# 		# alpha[key] = alpha[key] * 0.8
						# 		m.alpha[key] = 0.8 * m.alpha[key]

			# while epoch < args.epochs:
			# 	epoch = epoch + 10
			# 	converged = train_model(10, writer=writer)
			# 	if epoch%30==0:
			# 		for m in model.modules():
			# 			if isinstance(m,theta_solver):
			# 				m.regularizer = {}
			# 				for key in m.alpha.keys():
			# 					m.alpha[key] = 0.0
			# 				m.rhs.freeze_shift()
			# 	if epoch%50==0:
			# 		for m in model.modules():
			# 			if isinstance(m,theta_solver):
			# 				m.regularizer = {}
			# 				for key in m.alpha.keys():
			# 					m.alpha[key] = alpha[key]
			# 				m.rhs.unfreeze_shift()
				# for m in model.modules():
				# 	if (isinstance(m,theta_solver) and hasattr(m,'reduce') and m.reduce) or epoch>80:
				# 		m.reduce = False
				# 		m.regularizer = {}
				# 		for key in m.alpha.keys():
				# 			alpha[key] = alpha[key] * 0.8
				# 			m.alpha[key] = alpha[key]
			# i = 0
			# relaxation = 10
			# lr = args.lr
			# while epoch < args.epochs:
			# 	if i%2==0:
			# 		converged = train_model(50 - relaxation, writer=writer) #, optimizer=torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.alpha['wdecay']))
			# 		for m in model.modules():
			# 			if isinstance(m,theta_solver):
			# 				m.regularizer = {}
			# 				for key in m.alpha.keys(): m.alpha[key] = 0.0
			# 				m.rhs.freeze_shift()
			# 		epoch = epoch + 50 - relaxation
			# 	else:
			# 		converged = train_model(relaxation, writer=writer) #, optimizer=torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.alpha['wdecay']))
			# 		for m in model.modules():
			# 			if isinstance(m,theta_solver):
			# 				m.regularizer = {}
			# 				for key in m.alpha.keys():
			# 					alpha[key] = alpha[key] * 0.5
			# 					m.alpha[key] = alpha[key]
			# 				m.rhs.unfreeze_shift()
			# 		epoch = epoch + relaxation
			# 		lr = lr * 0.5
			# 		if relaxation<50: relaxation += 10
			# 	i += 1
			# 	if converged: break
			# converged = train_model(args.epochs, writer=writer)

			# if not converged:
			# if args.relaxation is not None:
			# 	torch.save( model.state_dict(), Path(paths['chkp_final'],runname+'_relax_0') )
			# 	for m in model.modules():
			# 		if isinstance(m,theta_solver):
			# 			m.tol = args.tol
			# 			# m.regularizer = {}
			# 			for key in m.r_alpha.keys(): m.r_alpha[key] = 0.0
			# 			m.rhs.freeze_spectral_circle()
			# 	while epoch < args.epochs+args.relaxation:
			# 		epoch = epoch + 10
			# 		train_model(10, writer=writer, optimizer=optimizer, scheduler=None, tol=1.e-8)
			# 		torch.save( model.state_dict(), Path(paths['chkp_final'],runname+'_relax_%d'%(epoch-args.epochs)) )
		except:
			raise
		finally:
			torch.save( model.state_dict(), Path(paths['chkp_final'],runname) )

		writer.close()
		# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

	elif args.mode=="test":

		# # always use full validation dataset
		# val_dataset = torchvision.datasets.MNIST("./MNIST", train=False, transform=transform, target_transform=None, download=True)
		# val_dataset = torch.utils.data.TensorDataset( val_dataset.data.unsqueeze(1).to(_device, dtype=_dtype)/255., val_dataset.targets.to(_device) )

		#########################################################################################
		fig_no = 0

		# model.load_state_dict(torch.load(Path("checkpoints","best_loss",runname), map_location=_device))
		# model.eval()


		# from mpl_toolkits import mplot3d
		# ax = plt.axes(projection='3d')
		# kernel = model[2].rhs.F[0].net[3].weight[0,2,...].cpu().detach().numpy()
		# # plt.imshow(kernel, cmap='gray')
		# X, Y = np.meshgrid(np.arange(7), np.arange(7))
		# ax.plot_surface(X,Y,kernel,cmap='viridis', edgecolor='none')
		# plt.show()
		# exit()


		#########################################################################################
		# evaluate spectrum


		x = []
		for i in range(1):
			x.append(dataset[i][0])
		x = torch.stack(x)
		model_spectrum = ex_setup.eval_model_spectrum(model, x, batch=10)
		ex_setup.pickle_data( runname, model_spectrum, Path(paths['output'],'name2spectrum') )
		# exit()


		# ###############################################
		# # output of layers
		# fig = plt.figure(fig_no); fig_no += 1

		# np.set_printoptions(formatter={'all':lambda x: str("%6.2f"%(x))})


		# # attk = torchattacks.FGSM(model, eps=0.4)
		# # attk = torchattacks.GN(model, sigma=0.4)

		# # x = add_noise()(dataset[:1][0], 0.3)
		# x, y = dataset[1:2]
		# adv_x = attacks['GNclip'](x,y,0.3)
		# clean_conf_score = 100 * torch.nn.functional.softmax(model(x),dim=1).cpu().detach().numpy().ravel()
		# adver_conf_score = 100 * torch.nn.functional.softmax(model(adv_x),dim=1).cpu().detach().numpy().ravel()
		# print("%d->%d(%6.2f): "%(y[0].item(), clean_conf_score.argmax(), np.amax(clean_conf_score)), clean_conf_score)
		# print("%d->%d(%6.2f): "%(y[0].item(), adver_conf_score.argmax(), np.amax(adver_conf_score)), adver_conf_score)
		# fig = plt.figure(fig_no); fig_no += 1
		# # plt.imshow(adv_x[0,0,...].cpu().detach().numpy(), cmap='gray');  ex_setup.savefig(images_output+"_layer_0_ch_0_adv", format='jpg')
		# # plt.imshow(x[0,0,...].cpu().detach().numpy(),     cmap='gray');  ex_setup.savefig(images_output+"_layer_0_ch_0",     format='jpg')
		# collage = []
		# collage_adv = []
		# for l, m in enumerate(model):
		# 	if l>=5: break
		# 	adv_x = m(adv_x)
		# 	x = m(x)
		# 	if l>0:
		# 		collage.append( np.concatenate(x[0].cpu().detach().numpy(), axis=1) )
		# 		collage_adv.append( np.concatenate(adv_x[0].cpu().detach().numpy(), axis=1) )
		# 		# plt.imshow(np.concatenate(x_adv[0].cpu().detach().numpy(), axis=0), cmap='gray');  ex_setup.savefig(images_output+"_layer_%d_adv"%(l), format='jpg')
		# 		# plt.imshow(np.concatenate(x[0].cpu().detach().numpy(), axis=0),     cmap='gray');  ex_setup.savefig(images_output+"_layer_%d"%(l),     format='jpg')
		# 		# for ch in range(x.size(1)):
		# 		# 	plt.imshow(x[0,ch,...].cpu().detach().numpy(), cmap='gray');       ex_setup.savefig(images_output+"_layer_%d_ch_%d.jpg"%(l,ch), format='jpg')
		# 		# 	plt.imshow(x_clean[0,ch,...].cpu().detach().numpy(), cmap='gray'); ex_setup.savefig(images_output+"_layer_%d_ch_%d_clean.jpg"%(l,ch), format='jpg')
		# plt.imshow(np.concatenate(collage, axis=0));      ex_setup.savefig(images_output, format='jpg')
		# plt.imshow(np.concatenate(collage_adv, axis=0));  ex_setup.savefig(images_output+"_adv", format='jpg')
		# # print("Hello")
		# # exit()

		###############################################
		# model response to attacks

		for attk_name, attk in attacks.items():
			# Path(paths['out_data'],attk_name).mkdir(parents=True, exist_ok=True)
			for key, loader in dataloaders.items():
				print(" ")
				for eps in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
					start = time.time()

					#################################

					output = []
					labels = []
					for sample in loader:
						for _ in range(1):
							x, y = sample[0].to(_device), sample[1].to(_device)
							prob = torch.nn.functional.softmax(model(attk(x,y,eps)), dim=1)

							output.append(prob.detach().cpu())
							labels.append(y)
					output = torch.cat(output)
					labels = torch.cat(labels).reshape((-1,1))

					accuracy = ex_setup.topk_acc(output,labels,(1,2,3,4,5))

					#################################
					print("Evaluating "+attk_name+" attack with eps="+str(eps)+" for "+key+" data: "+str(int(time.time()-start))+" sec")

					# save accuracy to text file
					# fname  = Path(paths['out_data'],attk_name,(key+'_accuracy_eps_%.2f.txt'%(eps)))
					# header = 'theta,'+','.join(['top-%d'%(i) for i in range(1,6)]) if not os.path.exists(fname) else ''
					# with open(fname, 'a') as f:
					# 	np.savetxt( f, np.array([args.theta]+accuracy).reshape((1,-1)), fmt='%.2f', delimiter=',', header=header)

					# save accuracy vs current parameters
					# name = '%s_%s_noise_%s_eps_%.1f'%(key, runname, attk_name, eps)
					name = '%s_%s_%.1f-%s'%(key,attk_name,eps,runname)
					ex_setup.pickle_data( name, accuracy, Path(paths['output'],'name2accuracy') )


	elif args.mode=="process":
		import pickle
		import matplotlib.pyplot as plt
		# from implicitresnet.models.ode import theta_inv_stability_fun
		from ex_setup import theta_inv_stability_fun

		line_styles = ['-', '--', '-.']

		# images_output = "%s/%s"%(paths['out_images'], runname)

		fig_no = 0


		#########################################################################################
		# plot spectrum

		fig = plt.figure(fig_no); fig_no += 1

		# load spectrum
		with open(Path(paths['output'],'name2spectrum'),'rb') as f:
			spectrum = pickle.load(f)[runname]

		xlim=(-6,1)
		ylim=(-2,2)

		# location of the zero of the stability function
		x0 = theta_inv_stability_fun(args.theta, 0)

		all_eigenvalues        = []
		all_spectral_circles   = []
		all_gershgorin_circles = []

		# plot for each residual layer separately
		for i, (key, rhs_spectrum) in enumerate(spectrum.items()):
			plt.gca().clear()

			# plot stability region
			ex_setup.plot_stab(args.theta, xlim, ylim)

			# circle of spectral localization
			circle = rhs_spectrum['spectral_circle']
			patch = plt.Circle((circle[0], 0), circle[1], fill=False)
			patch.set_ec('b')
			patch.set_lw(1.5)
			patch.set_ls(line_styles[i])
			all_spectral_circles.append(patch)
			plt.gca().add_patch(patch)
			plt.plot([circle[0],circle[0]],[ylim[0],ylim[1]], 'b', ls=line_styles[i])

			# # Gershgorin circles
			# for patch in rhs_spectrum['gershgorin_circles']:
			# 	patch.set_ec('r')
			# 	patch.set_lw(1.5)
			# 	patch.set_ls(line_styles[i])
			# 	all_gershgorin_circles.append(patch)
			# 	plt.gca().add_patch(patch)

			plt.plot([x0,x0],[ylim[0],ylim[1]], 'black', ls='dotted')

			# plot eigenvalues
			# ex_setup.plot_spectrum(rhs_spectrum['eigenvalues'], args.theta, xlim=xlim, ylim=ylim, density=True, save_path=images_output+"_%s_spectrum.jpg"%(key))

			# collect all eigenvalues in a single array
			all_eigenvalues.append(rhs_spectrum['eigenvalues'])
		all_eigenvalues = np.concatenate(all_eigenvalues)

		# plot for all residual layers
		plt.gca().clear()
		ex_setup.plot_stab(args.theta, xlim, ylim)
		for patch in all_spectral_circles:   plt.gca().add_patch(patch)
		for patch in all_gershgorin_circles: plt.gca().add_patch(patch)
		plt.plot([x0,x0],[ylim[0],ylim[1]], 'black', ls='dotted')
		ex_setup.plot_spectrum(all_eigenvalues, args.theta, xlim=xlim, ylim=ylim, density=True, save_path=Path(paths['out_images'],"spectrum-"+runname+".jpg"))


		# plt.show()
		#########################################################################################
		# save accuracy vs eps in a format readable by pgfplots

		# fig = plt.figure(fig_no); fig_no += 1

		# # load accuracy
		# with open(Path(paths['output'],'name2accuracy'),'rb') as f:
		# 	accuracy = pickle.load(f)

		# stds  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
		# table = np.zeros((len(stds),2))

		# for key, loader in dataloaders.items():
		# 	for attk_name, attk in attacks.items():
		# 		Path(paths['out_data'],attk_name).mkdir(parents=True, exist_ok=True)
		# 		filename = '%s_acc_vs_eps-%s.txt'%(key,runname)
		# 		# name = '%s_%s_noise_%s'%(key, runname, attk_name)
		# 		# make table
		# 		for i, eps in enumerate(stds):
		# 			name = '%s_%s_%.1f-%s'%(key,attk_name,eps,runname)
		# 			# table[i,:] = eps, accuracy[name+'_eps_%.1f'%(eps)][0]
		# 			table[i,:] = eps, accuracy[name][0]
		# 		np.savetxt( Path(paths['out_data'],attk_name,filename), table, delimiter=',', fmt='%0.2f' )
		# 		# np.savetxt( Path(paths['output'],'data',name+'_vs_eps.txt'), table, delimiter=',', fmt='%0.2f' )

		#########################################################################################
		# save corrupted images

		# fig = plt.figure(fig_no); fig_no += 1

		# for attk_name, attk in attacks.items():
		# 	for eps in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
		# 		for sample in dataloaders['train']:
		# 			x, y  = sample[0].to(_device), sample[1].to(_device)
		# 			adv_x = attk(x,y,eps).cpu().numpy()[0,0,...]
		# 			plt.imshow(adv_x, cmap='gray')
		# 			ex_setup.savefig(str(Path(paths['out_images'],"%s_eps_%.1f"%(attk_name,eps))), format='jpg')
		# 			break