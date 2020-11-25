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
import torchvision
from torch.utils.tensorboard import SummaryWriter

import ex_setup
import layers
import utils



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


	_device = ex_setup._cpu #ex_setup._gpu if torch.cuda.is_available() else ex_setup._cpu
	_dtype  = ex_setup._dtype


	#########################################################################################
	#########################################################################################
	# Data
	np.random.seed(10)
	transform = torchvision.transforms.Compose([ torchvision.transforms.ToTensor() ]) #, torchvision.transforms.Normalize((0.1307,), (0.3081,))  ])

	dataset     = torchvision.datasets.MNIST("./MNIST", train=True,  transform=transform, target_transform=None, download=True)
	val_dataset = torchvision.datasets.MNIST("./MNIST", train=False, transform=transform, target_transform=None, download=True)
	if args.datasize>0:
		dataset     = torch.utils.data.Subset(dataset,     np.arange(args.datasize))
		val_dataset = torch.utils.data.Subset(val_dataset, np.arange(args.datasize//6))
	im_size = dataset[0][0].size()


	# account for the unbalanced data
	weight = np.zeros(10,)
	for i in range(len(dataset)):
		weight[dataset[i][1]] += 1
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
	class ode_block(ex_setup.ode_block_base):
		def __init__(self, channels):
			super().__init__(args)
			self.ode = layers.theta_solver( ex_setup.rhs_conv2d(channels, args), args.T, args.steps, args.theta, tol=args.tol )
	########################################################


	########################################################
	class model(torch.nn.Module):
		def __init__(self):
			super().__init__()

			ch = args.codim+1
			self.net = torch.nn.Sequential(
				######################################
				torch.nn.Conv2d(1, ch, 7, stride=2, padding=3, bias=False),			#chx14x14	[0]
				torch.nn.MaxPool2d(3, stride=2),									#chx7x7		[1]
				ode_block(ch),														#chx7x7		[2]
				######################################
				torch.nn.Conv2d(ch, 2*ch, 3, stride=2, padding=1, bias=False),		#2chx4x4	[3]
				ode_block(2*ch),													#2chx4x4	[4]
				######################################
				torch.nn.Conv2d(2*ch, 4*ch, 3, stride=2, padding=1, bias=False),	#4chx2x2	[5]
				ode_block(4*ch),													#			[6]
				######################################
				# Classifier
				torch.nn.AvgPool2d(2),												#4chx1x1	[7]
				torch.nn.Flatten(),													#4chx1		[8]
				torch.nn.Linear(in_features=4*ch, out_features=10, bias=True),		#10x1		[9]
			)

		def forward(self, x):
			out = self.net(x.requires_grad_(True))
			return out
	########################################################




	#########################################################################################
	#########################################################################################
	# init/train/plot model

	# uncommenting this line will enable debug mode and lead to increased cost and memory leaking
	# torch.autograd.set_detect_anomaly(True)


	paths = ex_setup.create_paths(args)


	model     = ex_setup.load_model(model(), args, _device)
	optimizer = ex_setup.get_optimizer('adam', model, args.lr, wdecay=args.wdecay)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=10, verbose=True, threshold=1.e-6, threshold_mode='rel', cooldown=10, min_lr=1.e-6, eps=1.e-8)
	# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7, last_epoch=-1)
	# lr_schedule = np.linspace(args.lr, args.lr/100, args.epochs)
	# checkpoint={'epochs':1000, 'name':"models/"+script_name+"/sim_"+str(sim)+'_'+file_name[:]}
	train_obj = utils.training_loop(model, dataset, val_dataset, args.batch, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn, accuracy_fn=accuracy_fn, write_hist=False, history=False, checkpoint=None)


	# #########################################################################################
	# # torch.autograd.set_detect_anomaly(True)

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

	elif args.mode=="test":

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


		def add_noise(image, std=0.0):
			if std==0:
				return image
			else:
				return image + std * torch.randn_like(image)
			# return image + std * (2*torch.rand_like(image)-1)
			# img = image.detach()
			# img[image==0] = img[image==0] + std * torch.rand_like(image).abs()[image==0]
			# img[image==1] = img[image==1] - std * torch.rand_like(image).abs()[image==1]
			# return img


		#########################################################################################
		#########################################################################################


		model.eval()


		###############################################
		# model response to corrupted images

		for std in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
			out_noise = []
			labels    = []

			dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=False)
			for batch_ndx, sample in enumerate(dataloader):
				x, y = sample[0].to(_device), sample[1].cpu()
				out_noise.append(torch.nn.functional.softmax(model(add_noise(x,std)),dim=1).detach().cpu())
				labels.append(y)

			out_noise = torch.cat(out_noise)
			labels    = torch.cat(labels).reshape((-1,1))

			acc_noise = topk_acc(out_noise,labels,(1,2,3,4,5))

			np.savetxt( Path(paths['output_data'],('acc_std%3.1f_theta%4.2f'%(std,args.theta)).replace('.','')+'.txt'), np.array(acc_noise).reshape((1,-1)), delimiter=',')


	# elif args.mode=="plot":
	# 	from skimage.util import montage

	# 	def add_noise(image, std=0.6):
	# 		# return image + std * (2*torch.rand_like(image).to(image.dtype)-1)
	# 		return image + std * torch.rand_like(image).to(image.dtype)

	# 	model = get_model(args.seed)
	# 	missing_keys, unexpected_keys = model.load_state_dict(torch.load(Path("checkpoints","init",file_name), map_location=cpu))
	# 	# missing_keys, unexpected_keys = model.load_state_dict(torch.load(Path("initialization","0.00"), map_location=cpu))
	# 	model.eval()

	# 	model = model.net

	# 	for image, label in dataset:
	# 		fig_no = 0
	# 		out_plain = image.reshape(1,*image.shape)
	# 		out_noise = add_noise(out_plain)
	# 		# original and noise image
	# 		plt.figure(fig_no); fig_no += 1;
	# 		plt.subplot(211); plt.imshow(out_plain.detach().numpy()[0,0,...], cmap='gray')
	# 		plt.subplot(212); plt.imshow(out_noise.detach().numpy()[0,0,...], cmap='gray')
	# 		# layer outputs
	# 		old_l = 0
	# 		for l in [1,2]: #[1,2,5,6,9,10]
	# 			out_plain = model[old_l:l+1](out_plain)
	# 			out_noise = model[old_l:l+1](out_noise); old_l = l+1
	# 			out_channels = out_plain.shape[1]
	# 			plt.figure(fig_no); fig_no += 1;
	# 			plt.subplot(211); plt.imshow( montage( [out_plain.detach().numpy()[0,i-1,:,:] for i in range(1,out_channels+1)], grid_shape=(1,out_channels), multichannel=False ), cmap='gray')
	# 			plt.subplot(212); plt.imshow( montage( [out_noise.detach().numpy()[0,i-1,:,:] for i in range(1,out_channels+1)], grid_shape=(1,out_channels), multichannel=False ), cmap='gray')
	# 		print(model[old_l:](out_plain).argmax(dim=1).detach().numpy()[0], model[old_l:](out_noise).argmax(dim=1).detach().numpy()[0], label)
	# 		plt.show()
	# 		exit()



