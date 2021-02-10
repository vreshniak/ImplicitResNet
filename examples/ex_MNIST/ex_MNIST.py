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

from implicitresnet import utils, theta_solver, regularized_ode_solver, rhs_conv2d
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
	if args.datasize>0:
		dataset     = torch.utils.data.Subset(dataset,     np.arange(args.datasize))
		val_dataset = torch.utils.data.Subset(val_dataset, np.arange(args.datasize//6))
	im_size = dataset[0][0].size()

	# account for the unbalanced data
	weight = np.zeros(10,)
	for i in range(len(dataset)):
		# weight[dataset[i][1]] += 1
		weight[dataset[i][1].item()] += 1
	weight = 1. / weight
	weight = torch.from_numpy( weight/weight.sum() ).float().to(device=_device)


	#########################################################################################
	#########################################################################################
	# Loss function


	loss_fn     = torch.nn.CrossEntropyLoss(weight=weight, reduction='mean')
	accuracy_fn = lambda y_pred, y: y_pred.argmax(dim=1).eq(y).sum().float() / y.nelement()


	#########################################################################################
	#########################################################################################
	# NN model


	########################################################
	def ode_block(im_shape, theta=0.00, stability_limits=None, alpha={}):
		rhs    = rhs_conv2d(im_shape, kernel_size=3, depth=2, T=args.T, num_steps=args.steps, power_iters=args.piters, learn_scales=args.learn_scales, learn_shift=args.learn_shift)
		solver = theta_solver( rhs, args.T, args.steps, theta, tol=args.tol )
		return regularized_ode_solver( solver, alpha, stability_limits, args.mciters )
	########################################################

	########################################################
	class add_noise(torch.nn.Module):
		def __init__(self):
			super().__init__()

		def forward(self, y, std=args.noisesize):
			return torch.clamp(y + std * torch.randn_like(y), 0, 1) if (self.training and std>0) else y
	########################################################

	########################################################
	ch = args.codim+1
	if args.name=='plain' or args.name=='1Lip':
		model = torch.nn.Sequential(
			######################################									#1x28x28	[input]
			add_noise(),															#1x28x28	[0]
			######################################
			ode_block((1,28,28)),													#1x28x28	[1]
			######################################
			torch.nn.Conv2d(1, ch, 7, stride=2, padding=3, bias=False),				#chx14x14	[2]
			torch.nn.MaxPool2d(3, stride=2),										#chx6x6		[3]
			ode_block((ch,6,6)),													#chx6x6		[4]
			######################################
			torch.nn.Conv2d(ch, 2*ch, 3, stride=2, padding=1, bias=False),			#2chx3x3	[5]
			ode_block((2*ch,3,3)),													#2chx3x3	[6]
			######################################
			torch.nn.Conv2d(2*ch, 4*ch, 3, stride=2, padding=1, bias=False),		#4chx2x2	[7]
			ode_block((4*ch,2,2)),													#4chx2x2	[8]
			######################################
			# Classifier
			torch.nn.AvgPool2d(2),													#4chx1x1	[9]
			torch.nn.Flatten(),														#4ch		[10]
			torch.nn.Linear(in_features=4*ch, out_features=10, bias=True),			#10			[11]
		)
	else:
		model = torch.nn.Sequential(
			######################################									#1x28x28	[input]
			add_noise(),															#1x28x28	[0]
			######################################
			ode_block((1,28,28), args.theta, args.stablim, args.alpha),				#1x28x28	[1]
			######################################
			torch.nn.Conv2d(1, ch, 7, stride=2, padding=3, bias=False),				#chx14x14	[2]
			torch.nn.MaxPool2d(3, stride=2),										#chx6x6		[3]
			ode_block((ch,6,6), args.theta, args.stablim, args.alpha),				#chx6x6		[4]
			######################################
			torch.nn.Conv2d(ch, 2*ch, 3, stride=2, padding=1, bias=False),			#2chx3x3	[5]
			ode_block((2*ch,3,3), args.theta, args.stablim, args.alpha),			#2chx3x3	[6]
			######################################
			torch.nn.Conv2d(2*ch, 4*ch, 3, stride=2, padding=1, bias=False),		#4chx2x2	[7]
			# ode_block((4*ch,2,2), 0.00, [0.0,1.0,2.0], {}),							#4chx2x2	[8]
			ode_block((4*ch,2,2), args.theta, args.stablim, args.alpha),			#4chx2x2	[8]
			######################################
			# Classifier
			torch.nn.AvgPool2d(2),													#4chx1x1	[9]
			torch.nn.Flatten(),														#4ch		[10]
			torch.nn.Linear(in_features=4*ch, out_features=10, bias=True),			#10			[11]
		)
	########################################################



	#########################################################################################
	#########################################################################################
	# train/test model

	# uncommenting this line will enable debug mode and lead to increased cost and memory leaking
	# torch.autograd.set_detect_anomaly(True)


	model = ex_setup.load_model(model, args, _device)


	if args.mode=="train":
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.alpha['wdecay'])
		scheduler = utils.optim.EvenReductionLR(optimizer, lr_reduction=0.1, gamma=0.9, epochs=args.epochs, last_epoch=-1)
		train_model = utils.TrainingLoop(model, loss_fn, dataset, args.batch, optimizer, val_dataset=val_dataset, scheduler=scheduler, accuracy_fn=accuracy_fn, val_freq=1, stat_freq=1)

		# with profiler.profile(record_shapes=True,use_cuda=True) as prof:
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
		# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

	elif args.mode=="test":
		import matplotlib.pyplot as plt
		from implicitresnet.utils.spectral import eigenvalues, spectralnorm
		#########################################################################################
		fig_no = 0

		images_output = "%s/%s"%(Path(paths['output'],'images'), args.name)

		model.eval()


		#########################################################################################
		# plot spectrum
		fig = plt.figure(fig_no); fig_no += 1

		# evaluate spectrum
		spectralnrm  = []
		rhs_spectrum = []
		x = dataset[:10][0]
		for m in model:
			if not isinstance(m, theta_solver):
				spectralnrm.append( spectralnorm(m,x).mean().item() )
			else:
				spectrum_i = []
				m.ind_out = torch.arange(m.num_steps+1)
				odesol = m(x)
				m.ind_out = None
				for t in range(odesol.size(1)-1):
					spectrum_i.append( eigenvalues( lambda x: m.rhs(t,x), (1-args.theta)*odesol[:,t,...]+args.theta*odesol[:,t+1,...]) )
					spectralnrm.append(np.amax(ex_setup.theta_stab(spectrum_i[-1][:,0]+1j*spectrum_i[-1][:,1],args.theta)))
				rhs_spectrum.append(np.concatenate(np.array(spectrum_i)))
			x = m(x)

		# plot eigenvalues
		xmax = ymax = 4
		ex_setup.plot_stab(args.theta, xlim=(-xmax,xmax), ylim=(-ymax,ymax))
		for eigs in rhs_spectrum:
			plt.plot(eigs[:,0], eigs[:,1], 'o', markersize=4) #, markerfacecolor='none')
		plt.savefig(images_output+"_spectrum.jpg", bbox_inches='tight', pad_inches=0.0)

		# save spectral norm of layers
		header = 'layer-1'
		for i in range(2,len(spectralnrm)): header+=',layer-%d'%(i)
		fname = Path(paths['output_data'],('spectral_norm.txt'))
		if not os.path.exists(fname):
			with open(fname, 'w') as f:
				np.savetxt( f, np.array(spectralnrm).reshape((1,-1)), fmt='%.2e', delimiter=',', header=header)
		else:
			with open(fname, 'a') as f:
				np.savetxt( f, np.array(spectralnrm).reshape((1,-1)), fmt='%.2e', delimiter=',')


		###############################################
		# model response to corrupted images

		def topk_acc(input, target, topk=(1,)):
			"""Computes the precision@k for the specified values of k"""
			maxk = max(topk)
			batch_size = target.size(0)

			_, pred = input.topk(k=maxk, dim=1, largest=True, sorted=True)
			pred = pred.t()
			correct = pred.eq(target.view(1, -1).expand_as(pred))

			res = []
			for k in topk:
				# correct_k = correct[:k].view(-1).float().sum(0)
				correct_k = correct[:k].flatten().float().sum(0)
				res.append(float("{:.1f}".format(correct_k.mul_(100.0 / batch_size).detach().numpy())))
			return res


		for std in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
			out_noise = []
			labels    = []

			# evaluate model
			for _ in range(10):
				dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=False)
				for batch_ndx, sample in enumerate(dataloader):
					x, y = sample[0].to(_device), sample[1].cpu()
					out_noise.append(torch.nn.functional.softmax(model(add_noise()(x,std)),dim=1).detach().cpu())
					labels.append(y)

			# evaluate accuracy
			out_noise = torch.cat(out_noise)
			labels    = torch.cat(labels).reshape((-1,1))
			acc_noise = topk_acc(out_noise,labels,(1,2,3,4,5))

			# save accuracy
			header = 'theta'
			for i in range(1,6): header+=',top-%d'%(i)
			fname = Path(paths['output_data'],('acc_noise_std_%.2f.txt'%(std)))
			if not os.path.exists(fname):
				with open(fname, 'w') as f:
					np.savetxt( f, np.array([args.theta]+acc_noise).reshape((1,-1)), fmt='%.2f', delimiter=',', header=header)
			else:
				with open(fname, 'a') as f:
					np.savetxt( f, np.array([args.theta]+acc_noise).reshape((1,-1)), fmt='%.2f', delimiter=',')


		# ###############################################
		# # output of layers
		# # x = add_noise()(dataset[:1][0], 0.3)
		# x = dataset[:1][0]
		# plt.imshow(x[0,0,...].cpu().detach().numpy(), cmap='gray')
		# plt.gca().axes.xaxis.set_visible(False)
		# plt.gca().axes.yaxis.set_visible(False)
		# plt.gca().axis('off')
		# plt.savefig(images_output+"_layer_0_ch_0.jpg", bbox_inches='tight', pad_inches=0.0)
		# for l, m in enumerate(model):
		# 	if l>=3: break
		# 	x = m(x)
		# 	for ch in range(x.size(1)):
		# 		plt.imshow(x[0,ch,...].cpu().detach().numpy(), cmap='gray')
		# 		plt.gca().axes.xaxis.set_visible(False)
		# 		plt.gca().axes.yaxis.set_visible(False)
		# 		plt.gca().axis('off')
		# 		plt.savefig(images_output+"_layer_%d_ch_%d.jpg"%(l+1,ch), bbox_inches='tight', pad_inches=0.0)