import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
import numpy as np

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch.autograd.profiler as profiler

import torchattacks

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


	#########################################################################################
	#########################################################################################
	# compose file name


	file_name   = ex_setup.make_name(args)
	script_name = sys.argv[0][:-3]


	#########################################################################################
	#########################################################################################


	_device = ex_setup._gpu if torch.cuda.is_available() else ex_setup._cpu
	_dtype  = ex_setup._dtype


	#########################################################################################
	#########################################################################################
	# Data


	np.random.seed(10)
	transform = torchvision.transforms.Compose([ torchvision.transforms.ToTensor() ]) #, torchvision.transforms.Normalize((0.1307,), (0.3081,))  ])

	# original data
	dataset     = torchvision.datasets.MNIST("./MNIST", train=True,  transform=transform, target_transform=None, download=True)
	val_dataset = torchvision.datasets.MNIST("./MNIST", train=False, transform=transform, target_transform=None, download=True)
	# move data to device
	dataset     = torch.utils.data.TensorDataset(     dataset.data.unsqueeze(1).to(_device, dtype=_dtype)/255.,     dataset.targets.to(_device) )
	val_dataset = torch.utils.data.TensorDataset( val_dataset.data.unsqueeze(1).to(_device, dtype=_dtype)/255., val_dataset.targets.to(_device) )
	# subset of data
	weight = None
	if args.datasize>0:
		dataset     = torch.utils.data.Subset(dataset,     np.arange(args.datasize))
		val_dataset = torch.utils.data.Subset(val_dataset, np.arange(args.datasize//6))

		# account for the unbalanced data
		weight = np.zeros(10,)
		for i in range(len(dataset)):
			# weight[dataset[i][1]] += 1
			weight[dataset[i][1].item()] += 1
		weight = 1. / weight
		weight = torch.from_numpy( weight/weight.sum() ).float().to(device=_device)
	im_size = dataset[0][0].size()


	#########################################################################################
	#########################################################################################
	# Loss function


	loss_fn     = torch.nn.CrossEntropyLoss(weight=weight, reduction='mean')
	accuracy_fn = lambda y_pred, y: y_pred.argmax(dim=1).eq(y).sum().float() / y.nelement()


	#########################################################################################
	#########################################################################################
	# NN model


	########################################################
	def ode_block(im_shape, theta=0.00, stability_limits=None, alpha={}, kernel_size=3):
		rhs    = rhs_conv2d(im_shape, kernel_size=kernel_size, depth=2, T=1, num_steps=1, power_iters=args.piters, spectral_limits=args.eiglim, learn_scales=args.learn_scales, learn_shift=args.learn_shift, bias=False, activation=args.sigma)
		solver = theta_solver( rhs, args.T, args.steps, theta, tol=args.tol )
		return regularized_ode_solver( solver, alpha, stability_limits, args.mciters, p=1 )
	########################################################

	########################################################
	class Normalize(torch.nn.Module) :
		def __init__(self) :
			super().__init__()
		def forward(self, input):
			return (input - 0.1307) / 0.3081
	########################################################


	if args.model is None: args.model='1'
	if args.model=='1':
		ch = args.codim + 1
		model = torch.nn.Sequential(
			Normalize(),															#1x28x28	[0]
			######################################
			torch.nn.Conv2d(1, ch, 7, stride=2, padding=3, bias=True),				#chx14x14	[1]
			ode_block((ch,14,14), args.theta, args.stablim, args.alpha),			#chx14x14	[2]
			######################################
			torch.nn.Conv2d(ch, 2*ch, 3, stride=2, padding=1, bias=True),			#2chx7x7	[3]
			ode_block((2*ch,7,7), args.theta, args.stablim, args.alpha),			#2chx7x7	[4]
			######################################
			torch.nn.Conv2d(2*ch, 4*ch, 4, stride=2, padding=1, bias=True),			#4chx3x3	[5]
			ode_block((4*ch,3,3), args.theta, args.stablim, args.alpha),			#4chx3x3	[6]
			######################################
			# Classifier
			torch.nn.AvgPool2d(3),													#4chx1x1	[7]
			torch.nn.Flatten(),														#4ch		[8]
			torch.nn.Linear(in_features=4*ch, out_features=10, bias=True),			#10			[9]
		)
	elif args.model=='2':
		ch = args.codim + 1
		model = torch.nn.Sequential(
			Normalize(),															#1x28x28	[0]
			######################################
			torch.nn.ConstantPad3d((0,0,0,0,0,args.codim), 0.0),					#chx28x28	[1]
			ode_block((ch,28,28), args.theta, args.stablim, args.alpha, 7),			#chx28x28	[2]
			######################################
			torch.nn.Conv2d(ch, ch, 7, stride=2, padding=3, bias=True),				#chx14x14	[3]
			ode_block((ch,14,14), args.theta, args.stablim, args.alpha),			#chx14x14	[4]
			######################################
			torch.nn.Conv2d(ch, 2*ch, 3, stride=2, padding=1, bias=True),			#2chx7x7	[5]
			ode_block((2*ch,7,7), args.theta, args.stablim, args.alpha),			#2chx7x7	[6]
			######################################
			torch.nn.Conv2d(2*ch, 4*ch, 4, stride=2, padding=1, bias=True),			#4chx3x3	[7]
			ode_block((4*ch,3,3), args.theta, args.stablim, args.alpha),			#4chx3x3	[8]
			######################################
			# Classifier
			torch.nn.AvgPool2d(3),													#4chx1x1	[9]
			torch.nn.Flatten(),														#4ch		[10]
			torch.nn.Linear(in_features=4*ch, out_features=10, bias=True),			#10			[11]
		)


	#########################################################################################
	#########################################################################################
	# train/test model

	# uncommenting this line will enable debug mode and lead to increased cost and memory leaking
	# torch.autograd.set_detect_anomaly(True)

	model = ex_setup.load_model(model, args, _device)

	attacks = [ torchattacks.FGSM(model, eps=(0.1*i)) for i in range(1,3) ] #+ [ torchattacks.GN(model, sigma=0.1*i) for i in range(1,4) ]
	# attacks = [ torchattacks.GN(model, sigma=(0.1*i)) for i in range(1,4) ]
	def attack(x,y):
		# adv_x = attacks[np.random.randint(len(attacks))](x,y)
		# dx = (adv_x - x ) / 0.3
		# return torch.cat([ x + 0.1*i*dx for i in range(1,4) ])
		return attacks[np.random.randint(len(attacks))](x,y)
		# return torch.cat([ attk(x,y) for attk in attacks ])

	# def attack(x,y):
	# 	eps = 0.3
	# 	delta = torch.zeros_like(x, requires_grad=True)
	# 	L  = loss_fn(model(x+delta),y)
	# 	dx = torch.autograd.grad(L, delta, create_graph=False, retain_graph=False, only_inputs=True)[0]
	# 	return (x+eps*dx.sign()).clamp(0,1).detach()


	if args.mode=="train":
		optimizer = torch.optim.Adam(model.parameters(),  lr=args.lr, weight_decay=args.alpha['wdecay'])
		scheduler = utils.optim.EvenReductionLR(optimizer, lr_reduction=0.05, gamma=0.5, epochs=args.epochs, last_epoch=-1)
		# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, threshold=1.e-4, threshold_mode='rel', cooldown=5, min_lr=0, eps=1e-08, verbose=False)
		checkpoints = None
		# checkpoints = {'dir': paths['checkpoints'], 'each_nth': 5, 'name': file_name}
		train_model = utils.TrainingLoop(model, loss_fn, dataset, args.batch, optimizer, data_augmentation=None, val_dataset=val_dataset, val_batch_size=args.batch, scheduler=scheduler, accuracy_fn=accuracy_fn, val_freq=1, stat_freq=1, checkpoints=checkpoints, tol=1.e-4)

		# with profiler.profile(record_shapes=True,use_cuda=True) as prof:
		writer = SummaryWriter(Path("logs",file_name))

		# save initial model and train
		torch.save( model.state_dict(), Path(paths['chkp_init'],file_name) )
		try:
			alpha = args.alpha.copy()
			epoch = 0
			i = 0
			while epoch < args.epochs:
				if i%2==0:
					epoch = epoch + 30
					i += 1
					train_model(30, writer=writer)
					for m in model.modules():
						if isinstance(m,theta_solver):
							m.regularizer = {}
							for key in m.alpha.keys(): m.alpha[key] = 0.0
							m.rhs.freeze_shift()
				else:
					epoch = epoch + 20
					i += 1
					train_model(20, writer=writer)
					for m in model.modules():
						if isinstance(m,theta_solver):
							m.regularizer = {}
							for key in m.alpha.keys(): m.alpha[key] = alpha[key]
							m.rhs.unfreeze_shift()
			# train_model(args.epochs, writer=writer)
			for m in model.modules():
				if isinstance(m,theta_solver):
					m.regularizer = {}
					for key in m.alpha.keys(): m.alpha[key] = 0.0
					m.rhs.freeze_shift()
			train_model(100, writer=writer, optimizer=optimizer, scheduler=None)
		except:
			raise
		finally:
			torch.save( model.state_dict(), Path(paths['chkp_final'],file_name) )

		writer.close()
		# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

	elif args.mode=="test":
		import matplotlib.pyplot as plt
		from implicitresnet.utils.spectral import eigenvalues, spectralnorm
		from collections import deque
		import pickle
		#########################################################################################
		fig_no = 0

		images_output = "%s/%s"%(paths['out_images'], args.name)

		# model.load_state_dict(torch.load(Path("checkpoints","best_loss",file_name), map_location=_device))
		model.eval()


		# from mpl_toolkits import mplot3d
		# ax = plt.axes(projection='3d')
		# kernel = model[2].rhs.F[0].net[3].weight[0,2,...].cpu().detach().numpy()
		# # plt.imshow(kernel, cmap='gray')
		# X, Y = np.meshgrid(np.arange(7), np.arange(7))
		# ax.plot_surface(X,Y,kernel,cmap='viridis', edgecolor='none')
		# plt.show()
		# exit()


		#########################################################################################
		# plot spectrum
		fig = plt.figure(fig_no); fig_no += 1

		# evaluate spectrum
		spectralnrm  = []
		rhs_spectrum = []
		x = dataset[:5][0]
		for m in model:
			if isinstance(m, theta_solver):
				spectrum_i = []
				m.ind_out = torch.arange(m.num_steps+1)
				odesol = m(x)
				m.ind_out = None
				for t in range(odesol.size(1)-1):
					spectrum_i.append( eigenvalues( lambda x: m.rhs(t,x), (1-m.theta.item())*odesol[:,t,...]+m.theta.item()*odesol[:,t+1,...]) )
					spectralnrm.append(np.amax(ex_setup.theta_stab(spectrum_i[-1][:,0]+1j*spectrum_i[-1][:,1],m.theta.item())))
				rhs_spectrum.append(np.concatenate(np.array(spectrum_i)))
				# plot eigenvalues
				xmax = ymax = 4
				plt.gca().clear()
				ex_setup.plot_stab(m.theta.item(), xlim=(-xmax,xmax), ylim=(-ymax,ymax))
				# print(rhs_spectrum[-1].shape, np.amax(np.abs(rhs_spectrum[-1])))
				plt.plot(rhs_spectrum[-1][:,0], rhs_spectrum[-1][:,1], 'o', markersize=4) #, markerfacecolor='none')
				plt.savefig(images_output+"_%s_spectrum.jpg"%(m.name), bbox_inches='tight', pad_inches=0.0, dpi=300)
			else:
				spectralnrm.append( spectralnorm(m,x).mean().item() )
			x = m(x)

		# save spectral norm of layers
		fname  = Path(paths['out_data'],('spectral_norm.txt'))
		header = 'layer-1,'+','.join(['layer-%d'%(i) for i in range(2,len(spectralnrm))]) if not os.path.exists(fname) else ''
		with open(fname, 'a') as f:
			np.savetxt( f, np.array(spectralnrm).reshape((1,-1)), fmt='%.2e', delimiter=',', header=header)


		###############################################
		# output of layers
		fig = plt.figure(fig_no); fig_no += 1

		np.set_printoptions(formatter={'all':lambda x: str("%6.2f"%(x))})


		# attk = torchattacks.FGSM(model, eps=0.4)
		attk = torchattacks.GN(model, sigma=0.4)

		# x = add_noise()(dataset[:1][0], 0.3)
		x, y = val_dataset[1:2]
		adv_x = attk(x,y)
		clean_conf_score = 100 * torch.nn.functional.softmax(model(x),dim=1).cpu().detach().numpy().ravel()
		adver_conf_score = 100 * torch.nn.functional.softmax(model(adv_x),dim=1).cpu().detach().numpy().ravel()
		print("%d->%d(%6.2f): "%(y[0].item(), clean_conf_score.argmax(), np.amax(clean_conf_score)), clean_conf_score)
		print("%d->%d(%6.2f): "%(y[0].item(), adver_conf_score.argmax(), np.amax(adver_conf_score)), adver_conf_score)
		fig = plt.figure(fig_no); fig_no += 1
		plt.imshow(adv_x[0,0,...].cpu().detach().numpy(), cmap='gray');  ex_setup.savefig(images_output+"_layer_0_ch_0_adv", format='jpg')
		plt.imshow(x[0,0,...].cpu().detach().numpy(),     cmap='gray');  ex_setup.savefig(images_output+"_layer_0_ch_0",     format='jpg')
		collage = []
		collage_adv = []
		for l, m in enumerate(model):
			if l>=5: break
			adv_x = m(adv_x)
			x = m(x)
			if l>0:
				collage.append( np.concatenate(x[0].cpu().detach().numpy(), axis=1) )
				collage_adv.append( np.concatenate(adv_x[0].cpu().detach().numpy(), axis=1) )
				# plt.imshow(np.concatenate(x_adv[0].cpu().detach().numpy(), axis=0), cmap='gray');  ex_setup.savefig(images_output+"_layer_%d_adv"%(l), format='jpg')
				# plt.imshow(np.concatenate(x[0].cpu().detach().numpy(), axis=0),     cmap='gray');  ex_setup.savefig(images_output+"_layer_%d"%(l),     format='jpg')
				# for ch in range(x.size(1)):
				# 	plt.imshow(x[0,ch,...].cpu().detach().numpy(), cmap='gray');       ex_setup.savefig(images_output+"_layer_%d_ch_%d.jpg"%(l,ch), format='jpg')
				# 	plt.imshow(x_clean[0,ch,...].cpu().detach().numpy(), cmap='gray'); ex_setup.savefig(images_output+"_layer_%d_ch_%d_clean.jpg"%(l,ch), format='jpg')
		plt.imshow(np.concatenate(collage, axis=0));      ex_setup.savefig(images_output, format='jpg')
		plt.imshow(np.concatenate(collage_adv, axis=0));  ex_setup.savefig(images_output+"_adv", format='jpg')


		###############################################
		# model response to adversarial attacks

		dataloaders = {
			'train': torch.utils.data.DataLoader(dataset,     batch_size=args.batch, shuffle=False),
			'valid': torch.utils.data.DataLoader(val_dataset, batch_size=args.batch, shuffle=False)
			}

		for std in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
			attacks = [torchattacks.GN(model, sigma=std)] #, torchattacks.FGSM(model, eps=std)]
			for attk in attacks:
				attk_name = attk.__class__.__name__
				Path(paths['out_data'],attk_name).mkdir(parents=True, exist_ok=True)
				for key, loader in dataloaders.items():
					adv_out   = deque([])
					score     = deque([])
					adv_score = deque([])
					labels    = deque([])
					for sample in loader:
						for _ in range(1 if attk_name!='GN' else 5):
							x, y = sample[0].to(_device), sample[1].to(_device)
							prob     = torch.nn.functional.softmax(model(x),         dim=1)
							adv_prob = torch.nn.functional.softmax(model(attk(x,y)), dim=1)

							adv_out.append(adv_prob.detach().cpu())
							score.append(prob.amax(dim=1).detach().cpu())
							adv_score.append(adv_prob.amax(dim=1).detach().cpu())
							labels.append(y)
					adv_out   = torch.cat(list(adv_out))
					labels    = torch.cat(list(labels)).reshape((-1,1))
					score     = torch.cat(list(score))
					adv_score = torch.cat(list(adv_score))

					accuracy  = ex_setup.topk_acc(adv_out,labels,(1,2,3,4,5))

					# save accuracy
					fname = Path(paths['out_data'],attk.__class__.__name__,(key+'_accuracy_eps_%.2f.txt'%(std)))
					# fname = Path(paths['output_data'],('accuracy_eps_%.2f.txt'%(std)))
					header = 'theta,'+','.join(['top-%d'%(i) for i in range(1,6)]) if not os.path.exists(fname) else ''
					with open(fname, 'a') as f:
						np.savetxt( f, np.array([args.theta]+accuracy).reshape((1,-1)), fmt='%.2f', delimiter=',', header=header)

					if os.path.exists(Path(paths['output'],'name2accuracy')):
						with open(Path(paths['output'],'name2accuracy'),'rb') as f:
							name2accuracy = pickle.load(f)
						name2accuracy[file_name+key+attk_name+str(std)] = accuracy
					else:
						name2accuracy = {file_name+key+attk_name+str(std): accuracy}
					with open(Path('output','name2accuracy'),'wb') as f:
						pickle.dump(name2accuracy, f)
