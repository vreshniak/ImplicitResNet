import sys
import os
import time
import argparse
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from jacobian import JacobianReg

import custom_layers as layers
import custom_utils  as utils



if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	####################
	parser.add_argument("--prefix",      default=None )
	parser.add_argument("--mode",        type=str,   default="train", choices=["initialize", "train", "plot", "test"] )
	parser.add_argument("--seed",        type=int,   default=np.random.randint(0,10000))
	####################
	parser.add_argument("--nodes",       type=int,   default=2   )
	####################
	parser.add_argument("--epochs",      type=int,   default=1000)
	parser.add_argument("--lr",          type=float, default=0.01)
	parser.add_argument("--datasize",    type=int,   default=500 )
	parser.add_argument("--batch",       type=int,   default=-1  )
	####################
	parser.add_argument("--w_decay",     type=float, default=0   )
	parser.add_argument("--alpha_jac",   type=float, default=0   )
	args = parser.parse_args()

	print("\n-------------------------------------------------------------------")
	file_name = str(args.prefix)+"__" if args.prefix is not None and args.prefix!="_" else ''
	max_len = 0
	for arg in vars(args):
		length  = len(arg)
		max_len = length if length>max_len else max_len
	max_len += 1
	noise_args = ""
	for arg,value in vars(args).items():
		if value is not None:
			print("{0:>{length}}: {1}".format(arg,str(value),length=max_len))
			if arg!='prefix' and arg!='mode' and arg!='robust' and arg!='init' and "noise" not in arg:
				file_name += arg+"_"+str(value)+"__"
	print("-------------------------------------------------------------------")

	file_name  = file_name[:-2]
	# if args.method!='theta':
	# 	file_name = args.method+'__T_'+str(args.T)+'__steps_'+str(args.steps)
	script_name = sys.argv[0][:-3]


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

	fun  = lambda x: np.sin(x) #+ 0.1*np.cos(19.0*x)


	# training data
	ntrain = args.datasize
	xtrain = np.linspace(-5, 5, ntrain).reshape((ntrain,1))
	ytrain = fun(xtrain)
	# xtrain = np.vstack([xtrain]+[xtrain + 0.2*(2*np.random.rand(xtrain.shape[0],1)-1) for _ in range(5)])
	# ytrain = np.vstack([ytrain]+[ytrain for _ in range(5)])

	# validation data
	nval = 2*ntrain
	xval = np.linspace(-5, 5, nval).reshape((nval,1))
	yval = fun(xval)

	# training and validation datasets
	dataset     = torch.utils.data.TensorDataset( torch.from_numpy(xtrain).to(_dtype), torch.from_numpy(ytrain).to(_dtype) )
	val_dataset = torch.utils.data.TensorDataset( torch.from_numpy(xval).to(_dtype),   torch.from_numpy(yval).to(_dtype)   )


	#########################################################################################
	#########################################################################################
	# NN parameters

	loss_fn         = torch.nn.MSELoss(reduction='mean')
	optimizer_Adam  = lambda model, lr: torch.optim.Adam(model.parameters(),    lr=lr, weight_decay=args.w_decay)
	optimizer_RMS   = lambda model, lr: torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=args.w_decay)
	optimizer_SGD   = lambda model, lr: torch.optim.SGD(model.parameters(),     lr=lr, weight_decay=args.w_decay, momentum=0.5)
	optimizer_LBFGS = lambda model, lr: torch.optim.LBFGS(model.parameters(),   lr=1., max_iter=100, max_eval=None, tolerance_grad=1.e-9, tolerance_change=1.e-9, history_size=100, line_search_fn='strong_wolfe')
	scheduler_fn    = lambda optimizer: torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=True, threshold=1.e-5, threshold_mode='rel', cooldown=200, min_lr=1.e-6, eps=1.e-8)
	# scheduler_fn    = lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.7, last_epoch=-1)


	#########################################################################################
	#########################################################################################
	# NN model

	########################################################
	class model(torch.nn.Module):
		def __init__(self):
			super(model, self).__init__()

			nodes = args.nodes

			self.regularizers = {}
			# activation = torch.nn.ReLU()
			activation = torch.nn.GELU()
			# activation = torch.nn.Softsign()
			# activation = torch.nn.Tanhshrink()
			# activation = torch.nn.CELU()

			self.model = torch.nn.Sequential(
				torch.nn.Linear(1,          args.nodes, bias=True), activation,
				torch.nn.Linear(args.nodes, args.nodes, bias=True), activation,
				torch.nn.Linear(args.nodes, args.nodes, bias=True), activation,
				torch.nn.Linear(args.nodes, args.nodes, bias=True), activation,
				torch.nn.Linear(args.nodes, 1,          bias=True)
				)

			self.reg = JacobianReg(n=1)

		def forward(self, x):
			out = self.model(x.requires_grad_(True))
			if self.training and args.alpha_jac>0:
				self.regularizers['model_Jac'] = args.alpha_jac * self.reg(x, out)
			return out

		@property
		def regularizer(self):
			return {} if args.alpha_jac==0 else self.regularizers
	########################################################


	def get_model(seed=None):
		if seed is not None: torch.manual_seed(seed)
		return model().to(device=_device)



	#########################################################################################
	# torch.autograd.set_detect_anomaly(True)

	if args.mode=="train":
		logdir = Path("logs",file_name)
		writer = SummaryWriter(logdir)

		losses = []
		for sim in range(1):
			try:
				model       = get_model(args.seed+sim)
				optimizer   = optimizer_Adam(model, args.lr)
				scheduler   = scheduler_fn(optimizer)

				train_obj = utils.training_loop(model, dataset, val_dataset, args.batch, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn,
					writer=writer, write_hist=True, history=False, checkpoint=None)

				losses.append(train_obj(args.epochs))
			except:
				raise
			finally:
				Path("checkpoints/").mkdir(parents=True, exist_ok=True)
				torch.save( model.state_dict(), Path("checkpoints",file_name[:]) )

		writer.close()

	elif args.mode=="plot":
		model = get_model(args.seed)
		missing_keys, unexpected_keys = model.load_state_dict(torch.load(Path("checkpoints",file_name[:]), map_location=cpu))
		model.eval()

		# plot data
		nplot = 1000
		xplot = np.linspace(-5, 5, nplot).reshape((nplot,1))
		ytrue = fun(xplot)

		# propagation of input through layers
		ypred = model.model(torch.from_numpy(xplot).float()).detach().numpy()


		fig_no = 0
		###############################################
		# plot prediction
		fig = plt.figure(fig_no); fig_no += 1

		# plt.plot(xplot,ytrue)
		plt.plot(xplot,ypred)
		plt.plot(xtrain,ytrain,'o')
		plt.show()
