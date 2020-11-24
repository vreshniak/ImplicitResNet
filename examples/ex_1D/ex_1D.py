import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import torch

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

	#########################################################################################
	#########################################################################################


	_device = ex_setup._cpu
	_dtype  = ex_setup._dtype


	#########################################################################################
	#########################################################################################
	# Data
	np.random.seed(args.seed)

	fun  = (lambda x: np.sin(x) + 0.0*np.cos(19.0*x))
	# initialization of codimensions
	phi  = lambda x: torch.zeros((x.size(0),args.codim))
	# phi  = lambda x: torch.cat( [x for _ in range(args.codim)], 1 )
	# phi  = lambda x: torch.from_numpy(fun(x.detach().numpy()))


	# training data
	ntrain = args.datasize
	xtrain = np.linspace(-5, 5, ntrain).reshape((ntrain,1)) if ntrain>1 else np.array([[1]])
	ytrain = fun(xtrain)
	# xtrain = np.vstack( [np.linspace(-5, -1, ntrain//2).reshape((-1,1)), np.linspace(1, 5, ntrain//2).reshape((-1,1))] )
	# xtrain = np.vstack([xtrain + 0.1*(2*np.random.rand(xtrain.shape[0],1)-1) for _ in range(5)])
	# ytrain = np.vstack([ytrain for _ in range(5)])


	# validation data
	nval = 2*ntrain
	xval = np.linspace(-5, 5, nval).reshape((nval,1))
	yval = fun(xval)

	# training and validation datasets
	dataset     = torch.utils.data.TensorDataset( torch.from_numpy(xtrain).to(_dtype), torch.from_numpy(ytrain).to(_dtype) )
	val_dataset = torch.utils.data.TensorDataset( torch.from_numpy(xval).to(_dtype),   torch.from_numpy(yval).to(_dtype)   )


	#########################################################################################
	#########################################################################################
	# Loss and scheduler

	loss_fn      = torch.nn.MSELoss(reduction='mean')
	scheduler_fn = lambda optimizer: torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=50, verbose=True, threshold=1.e-4, threshold_mode='rel', cooldown=50, min_lr=1.e-6, eps=1.e-8)
	# scheduler_fn = lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.7, last_epoch=-1)


	#########################################################################################
	#########################################################################################
	# NN model


	########################################################
	# augment original data with additional dimensions
	class augment(torch.nn.Module):
		def __init__(self):
			super().__init__()

		def forward(self, x, val=0):
			if val==0:
				return torch.cat( (x, phi(x)), 1 )
			else:
				return torch.cat( (x, val*torch.ones((x.size(0),args.codim))), 1 )
	########################################################


	########################################################
	# last component as the function value
	class output(torch.nn.Module):
		def __init__(self):
			super().__init__()

		def forward(self, x):
			return x[:,-1:]
	########################################################


	########################################################
	class ode_block(ex_setup.ode_block_base):
		def __init__(self):
			super().__init__(args)
			self.ode = layers.theta_solver( ex_setup.rhs_mlp(1, args), args.T, args.steps, args.theta, tol=args.tol )
	########################################################


	########################################################
	class model(torch.nn.Module):
		def __init__(self):
			super().__init__()
			self.net = torch.nn.Sequential( augment(), ode_block(), output() )

		def forward(self, x):
			return self.net(x.requires_grad_(True))
	########################################################


	def get_model(seed=None):
		if seed is not None: torch.manual_seed(seed)
		mod = model().to(device=_device)
		return mod


	#########################################################################################
	# uncommenting this will lead to the increased cost and memory leaking
	# torch.autograd.set_detect_anomaly(True)


	# if args.mode=="init":
	# 	# create directories for the checkpoints and logs
	# 	chkp_dir = "initialization"
	# 	logdir = Path("logs","init__"+file_name)
	# 	Path(chkp_dir).mkdir(parents=True, exist_ok=True)
	# 	writer = SummaryWriter(logdir)

	# 	model       = get_model(args.seed)
	# 	optimizer   = optimizer_SGD(model, args.lr)
	# 	scheduler   = None
	# 	regularizer = None

	# 	train_obj = utils.training_loop(model, dataset, val_dataset, args.batch, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn, regularizer=regularizer,
	# 				writer=writer, write_hist=True, history=False, checkpoint=None)

	# 	# initialize with analytic continuation
	# 	for theta in np.linspace(0,1,101):
	# 		model.apply(lambda m: setattr(m,'theta',theta))
	# 		train_obj(10)
	# 		torch.save( model.state_dict(), Path(chkp_dir,"%4.2f"%(theta)) )

	# 	writer.close()


	checkpoint_dir_init, checkpoint_dir, out_dir, writer = ex_setup.create_paths(args, file_name)

	if args.mode!="plot":
		losses = []
		for sim in range(1):
			try:
				model     = get_model(args.seed+sim)
				optimizer = ex_setup.get_optimizer('adam', model, args.lr, wdecay=0)
				scheduler = scheduler_fn(optimizer)

				# lr_schedule = np.linspace(args.lr, args.lr/100, args.epochs)
				# checkpoint={'epochs':1000, 'name':"models/"+script_name+"/sim_"+str(sim)+'_'+file_name[:]}
				train_obj = utils.training_loop(model, dataset, val_dataset, args.batch, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn,
					writer=writer, write_hist=True, history=False, checkpoint=None)

				torch.save( model.state_dict(), checkpoint_dir_init )
				losses.append(train_obj(args.epochs))

				# if args.mode=="train":
				# 	torch.save( model.state_dict(), checkpoint_dir_init )
				# 	losses.append(train_obj(args.epochs))
				# elif args.mode=='init':
				# 	# initialize with analytic continuation
				# 	for theta in np.linspace(0,1,101):
				# 		model.apply(lambda m: setattr(m,'theta',theta))
				# 		train_obj(10)
				# 		torch.save( model.state_dict(), Path("checkpoints","init_"+args.prefix,"%4.2f"%(theta)) )
			except:
				raise
			finally:
				torch.save( model.state_dict(), checkpoint_dir )

		if writer is not None:
			writer.close()

	elif args.mode=="plot":
		fig_no = 0


		data_name = ("%s/th%4.2f_T%d_data%d_adiv%4.2f"%(out_dir, args.theta, args.T, args.datasize, args.adiv)).replace('.','')


		###############################################
		# load model
		model = get_model(args.seed)
		missing_keys, unexpected_keys = model.load_state_dict(torch.load(checkpoint_dir, map_location=ex_setup._cpu))
		model.eval()


		rhs_obj = model.net[1].ode.rhs

		with np.printoptions(precision=2, suppress=True):
			print("scales: ", rhs_obj.scales[0].detach().numpy())


		###############################################
		# prepare data

		# test data
		ntest = 200
		xtest = np.linspace(-6, 6, ntest).reshape((ntest,1))
		ytrue = fun(xtest)

		# propagation of train data through layers
		ytrain0 = model.net[0](torch.from_numpy(xtrain).float())
		ytrain1 = model.net[1](ytrain0, evolution=True)
		# for _ in range(2):
		# 	ytrain1 = torch.cat((ytrain1, model.net[1](ytrain1[-1], evolution=True)))
		# ytrain11 = torch.cat((ytrain1, model.net[1](ytrain1[-1], evolution=True)))
		ytrain2 = model.net[2](ytrain1[-1])

		# propagation of test data through layers
		ytest0 = model.net[0](torch.from_numpy(xtest).float())
		ytest1 = model.net[1](ytest0, evolution=True)
		ytest2 = model.net[2](ytest1[-1])

		# upper bound
		std = 0.2
		yup0 = model.net[0](torch.from_numpy(xtest).float(), std)
		yup1 = model.net[1](yup0, evolution=True)
		yup2 = model.net[2](yup1[-1])
		ydown0 = model.net[0](torch.from_numpy(xtest).float(), -std)
		ydown1 = model.net[1](ydown0, evolution=True)
		ydown2 = model.net[2](ydown1[-1])

		# convert to numpy arrays
		ytest0 = ytest0.detach().numpy()
		ytest1 = ytest1.detach().numpy()
		ytest2 = ytest2.detach().numpy()
		ytrain0 = ytrain0.detach().numpy()
		ytrain1 = ytrain1.detach().numpy()
		ytrain2 = ytrain2.detach().numpy()
		yup0 = yup0.detach().numpy()
		yup1 = yup1.detach().numpy()
		yup2 = yup2.detach().numpy()
		ydown0 = ydown0.detach().numpy()
		ydown1 = ydown1.detach().numpy()
		ydown2 = ydown2.detach().numpy()

		# ytrain11 = ytrain11.detach().numpy()


		###############################################
		# plot function
		fig = plt.figure(fig_no); fig_no += 1

		# plt.plot(xplot,ytrue)
		plt.plot(xtest, ytest2)
		plt.plot(xtrain, ytrain,'o')
		# plt.show()

		# np.savetxt( "out/data"+str(args.datasize)+"_ytrain"+".txt", np.hstack((xtrain,ytrain)), delimiter=',')
		# np.savetxt( data_name+"_ytest.txt", np.hstack((xtest,ytest2)), delimiter=',')


		###############################################
		# evaluate spectrum

		spectrum_train = []
		spectrum_test  = []
		xmax = ymax = 3
		# for t in range(args.T+1):
		for t in range(len(ytrain1)):
			spectrum_train.append( rhs_obj.spectrum(t, ytrain1[t]) )
			xmax = max(np.amax(np.abs(spectrum_train[-1][:,0])),xmax)
			ymax = max(np.amax(np.abs(spectrum_train[-1][:,1])),ymax)

			spectrum_test.append( rhs_obj.spectrum(t,ytest1[t]) )
			xmax = max(np.amax(np.abs(spectrum_test[-1][:,0])),xmax)
			ymax = max(np.amax(np.abs(spectrum_test[-1][:,1])),ymax)
		spectrum_train = np.array(spectrum_train)
		spectrum_test  = np.array(spectrum_test)

		# np.savetxt( data_name+"_eig1.txt", np.sqrt(spectrum_test[-1,1::2,0]**2+spectrum_test[-1,1::2,1]**2), delimiter=',')
		# np.savetxt( data_name+"_eig2.txt", np.sqrt(spectrum_test[-1,::2,0]**2+spectrum_test[-1,::2,1]**2), delimiter=',')

		spectrum_train = np.concatenate(spectrum_train)
		spectrum_test  = np.concatenate(spectrum_test)

		# np.savetxt( data_name+"_eig_train.txt", np.concatenate(spectrum_train), delimiter=',')
		# np.savetxt( data_name+"_eig_test.txt",  np.concatenate(spectrum_test),  delimiter=',')

		###############################################
		# plot stability function
		fig = plt.figure(fig_no); fig_no += 1

		theta_stab = lambda z, theta: abs((1+(1-theta)*z)/(1-theta*z))


		stab_val_train = theta_stab(spectrum_train[:,0]+1j*spectrum_train[:,1], args.theta)
		stab_val_test  = theta_stab(spectrum_test[:,0]+1j*spectrum_test[:,1],   args.theta)
		# plt.hist( stab_val_train, bins='auto' )
		plt.hist( stab_val_test, bins='auto', histtype='step' )
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
		plt.plot(spectrum_test[:,0],  spectrum_test[:,1], 'bo', markersize=4) #, markerfacecolor='none')
		plt.plot(spectrum_train[:,0], spectrum_train[:,1],'ro', markersize=4)

		plt.gca().axes.xaxis.set_visible(False)
		plt.gca().axes.yaxis.set_visible(False)
		plt.gca().axis('off')
		plt.savefig(data_name+"_spectrum.pdf", bbox_inches='tight', pad_inches=0.0)


		###############################################
		# plot vector field
		from matplotlib.widgets import Slider

		fig = plt.figure(fig_no); fig_no += 1
		plot_ax   = plt.axes([0.1, 0.2, 0.8, 0.65])
		slider_ax = plt.axes([0.1, 0.05, 0.8, 0.05])

		# vector fields
		X = np.linspace(-7, 7, 25)
		Y = np.linspace(-2, 2, 25)
		# X = np.linspace(np.amin(ytest1[...,0]),     np.amax(ytest1[...,0]),     25)
		# Y = np.linspace(np.amin(ytest1[...,1]) - 1, np.amax(ytest1[...,1]) + 1, 25)
		X,Y = np.meshgrid(X,Y)
		XY = np.hstack((X.reshape(-1,1), Y.reshape(-1,1)))
		# X = ytrain1[...,0].reshape((-1,1))
		# Y = ytrain1[...,1].reshape((-1,1))
		# XY = np.hstack((X, Y))
		UV = []
		for t in range(len(ytrain1)):
			UV.append(rhs_obj(t, torch.from_numpy(XY).float()).detach().numpy())
		UV = np.array(UV)


		def plot_step(t, ax=plot_ax):
			ax.clear()
			if t<args.T:
				ax.quiver(X, Y, UV[t][:,0], UV[t][:,1], angles='xy')
				# ax.quiver(X, Y, (1-args.theta)*UV[t][:,0] + args.theta*UV[t+1][:,0], (1-args.theta)*UV[t][:,1] + args.theta*UV[t+1][:,1])
				############
				# connecting lines
				for point in range(ytrain1.shape[1]):
					xx = [ ytrain1[t,point,0], ytrain1[t+1,point,0] ]
					yy = [ ytrain1[t,point,1], ytrain1[t+1,point,1] ]
					ax.annotate("", xy=(xx[1], yy[1]), xytext=(xx[0],yy[0]), arrowprops=dict(arrowstyle="->"))
					# plot_ax.plot(xx, yy, '-b')
				############
				ax.plot(ytest1[t,:,0],    ytest1[t,:,1],    '-r')
				ax.plot(ytest1[t+1,:,0],  ytest1[t+1,:,1],  '-b')
				ax.plot(ytrain1[t,:,0],   ytrain1[t,:,1],   '.r')
				ax.plot(ytrain1[t+1,:,0], ytrain1[t+1,:,1], '.b')
				############
				ax.set_xlim(np.amin(X),np.amax(X))
				ax.set_ylim(np.amin(Y),np.amax(Y))
			elif t==args.T:
				ax.quiver(X, Y, UV[t-1][:,0], UV[t-1][:,1], angles='xy')
				ax.plot(ytest1[t,:,0], ytest1[t,:,1], '-b')
				ax.set_xlim(np.amin(X),np.amax(X))
				ax.set_ylim(np.amin(Y),np.amax(Y))
			fig.canvas.draw_idle()

		# initial plot
		plot_step(0)
		# create the slider
		a_slider = Slider( slider_ax, label='step', valmin=0, valmax=len(ytrain1)-1, valinit=0, valstep=1 )
		def update(step):
			plot_step(int(step))
		a_slider.on_changed(update)


		fig = plt.figure(fig_no); fig_no += 1
		plt.quiver(X, Y, UV[0][:,0], UV[0][:,1], angles='xy')
		for t in range(len(ytrain1)-1):
			for point in range(ytrain1.shape[1]):
				xx = [ ytrain1[t,point,0], ytrain1[t+1,point,0] ]
				yy = [ ytrain1[t,point,1], ytrain1[t+1,point,1] ]
				plt.plot(xx, yy, '-k', linewidth=1.5)
				plt.gca().annotate("", xy=(xx[1], yy[1]), xytext=(xx[0],yy[0]), arrowprops=dict(arrowstyle="->"))
		plt.fill( np.concatenate((ydown1[0,:,0], yup1[0,-1::-1,0])),  np.concatenate((ydown1[0,:,1], yup1[0,-1::-1,1])),  'b', alpha=0.4 )
		plt.fill( np.concatenate((ydown1[-1,:,0],yup1[-1,-1::-1,0])), np.concatenate((ydown1[-1,:,1],yup1[-1,-1::-1,1])), 'r', alpha=0.4 )
		plt.plot(ytest1[0,:,0],   ytest1[0,:,1],   '-b', linewidth=2.5)
		plt.plot(ytest1[-1,:,0],  ytest1[-1,:,1],  '-r', linewidth=2.5)
		plt.plot(ytrain1[0,:,0],  ytrain1[0,:,1],  '.b', markersize=8)
		plt.plot(ytrain1[-1,:,0], ytrain1[-1,:,1], '.r', markersize=8)
		plt.xlim(-7,7)
		plt.ylim(-2,2)
		# plt.plot(ytrain11[-1,:,0], ytrain11[-1,:,1], '.g', markersize=4)
		plt.gca().axes.xaxis.set_visible(False)
		plt.gca().axes.yaxis.set_visible(False)
		plt.gca().axis('off')
		plt.savefig(data_name+"_traj.pdf", pad_inches=0.0, bbox_inches='tight')



		# plt.show()


		# fig = plt.figure(fig_no); fig_no += 1
		# for t in range(args.T):
		# 	plot_step(t, plt.gca())
		# 	plt.savefig(data_name+"_step"+str(t)+".pdf", bbox_inches='tight')


		###############################################
















