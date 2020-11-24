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
from torch.utils.tensorboard import SummaryWriter

import ex_setup
import layers
import utils




def theta_solver(fun, T, steps, theta, t0, y0):
	from scipy.optimize import minimize
	h = T / steps
	tt = []
	yy = []
	for i in range(len(t0)):
		t = np.linspace(t0[i], T, steps+1)
		y = [y0[i]]
		for step in range(steps):
			x = y[-1]
			if theta>0:
				residual = lambda z: np.sum( (z - x - h*fun(t[step]+theta*h, (1-theta)*x+theta*z))**2 )
				res = minimize(residual, y[-1]+h*fun(t[step], y[-1]))
				y.append( res.x )
			else:
				y.append( x + h*fun(t[step], x) )
		y = np.hstack(y)
		tt.append(t.reshape(1,steps+1,-1))
		yy.append(y.reshape(1,steps+1,-1))
	return np.vstack(tt), np.vstack(yy)





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


	_device = ex_setup._cpu
	_dtype  = ex_setup._dtype


	#########################################################################################
	#########################################################################################
	# Data
	from scipy.integrate import solve_ivp
	np.random.seed(args.seed)


	def correct_rhs(t,x):
		alpha, beta, gamma, delta = 2./3., 4./3., 1., 1.
		z = np.ones_like(x)
		z[...,0] = alpha*x[...,0] - beta*x[...,0]*x[...,1]
		z[...,1] = delta*x[...,0]*x[...,1] - gamma*x[...,1]
		return z


	def get_data(T=args.T, steps=args.steps, y0=None, rhs=correct_rhs):
		size = 10

		# T = 2*np.pi/(2./3.)
		# args.T = 2*np.pi/(2./3.)

		# data points
		if y0 is None:
			h = T / steps

			t0 = np.linspace(0,T,steps+1)
			y0 = np.ones((size,2))
			y0[:,0] = 1
			y0[:,1] = 0.9 + 0.1*np.arange(10)

		t = np.linspace(0,T,steps+1)

		y = []
		for tt0, yy0 in zip(t0,y0):
			# print(yy0)
			sol = solve_ivp(rhs, t_span=[0, T], y0=yy0, t_eval=np.linspace(0,T,steps+1), rtol=1.e-6)
			y.append( sol.y.T[np.newaxis,...] )
		y = np.vstack(y)

		y  = torch.from_numpy(y).float()
		y0 = torch.from_numpy(y0).float()

		return t, y0, y


	# # plot exact solution
	# t_train, y0_train, y_train = get_data()
	# for i in range(len(y0_train)):
	# 	plt.plot(t_train, y_train[i,:,0], '-r')
	# 	plt.plot(t_train, y_train[i,:,1], '-b')
	# plt.show()

	# # exact phase plot
	# for i in range(len(y0_train)):
	# 	plt.plot(y_train[i,:,0], y_train[i,:,1], '-')
	# plt.show()
	# exit()


	t_train, y0_train, y_train = get_data()
	y0_train, y_train = y0_train[1::2,...], y_train[1::2,...]
	y0_valid, y_valid = y0_train[0::2,...], y_train[0::2,...]

	dataset     = torch.utils.data.TensorDataset( y0_train, y_train)
	val_dataset = torch.utils.data.TensorDataset( y0_valid, y_valid)


	#########################################################################################
	#########################################################################################
	# NN parameters


	loss_fn = lambda input, target: (input[...,::args.steps//args.datasteps,:]-target[...,::args.steps//args.datasteps,:]).pow(2).flatten().mean()


	#########################################################################################
	#########################################################################################
	# NN model


	########################################################
	class ode_block(ex_setup.ode_block_base):
		def __init__(self):
			super().__init__(args)
			self.ode = layers.theta_solver( ex_setup.rhs_mlp(2, args), args.T, args.steps, args.theta, tol=args.tol )

		def forward(self, y0):
			return super().forward(y0, evolution=True).movedim(0,1)
	########################################################



	#########################################################################################
	#########################################################################################
	# init/train/plot model

	# uncommenting this line will enable debug mode and lead to increased cost and memory leaking
	# torch.autograd.set_detect_anomaly(True)


	paths = ex_setup.create_paths(args)


	model     = ex_setup.load_model(ode_block(), args, _device)
	optimizer = ex_setup.get_optimizer('adam', model, args.lr, wdecay=args.wdecay)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True, threshold=1.e-5, threshold_mode='rel', cooldown=50, min_lr=1.e-6, eps=1.e-8)
	# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.7, last_epoch=-1)
	# lr_schedule = np.linspace(args.lr, args.lr/100, args.epochs)
	# checkpoint={'epochs':1000, 'name':"models/"+script_name+"/sim_"+str(sim)+'_'+file_name[:]}
	train_obj = utils.training_loop(model, dataset, val_dataset, args.batch, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn, write_hist=False, history=False, checkpoint=None)


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

	elif args.mode=="plot":
		fig_no = 0

		# sim_name = ("%s_th%4.2f_T%d_data%d_adiv%4.2f_eigs%d_%4.2f"%(args.sigma[0], args.theta, args.T, args.datasize, args.adiv, -1 if math.isnan(args.eigs[0]) else args.eigs[0], -1 if math.isnan(args.eigs[1]) else args.eigs[1] )) #.replace('.','')
		# sim_name = ("th%4.2f_T%3.1f_steps%d_data_%d_%d_eigs%4.2f_%4.2f"%(args.theta, args.T, args.steps, args.datasize, args.datasteps, args.eigs[0], args.eigs[1] ))
		sim_name = ("th%4.2f_T%d_steps%d_data_%d_%d"%(args.theta, args.T, args.steps, args.datasize, args.datasteps ))

		Path(paths['output_data'],  sim_name).mkdir(parents=True, exist_ok=True)
		Path(paths['output_images'],sim_name).mkdir(parents=True, exist_ok=True)

		data_sim_output   = Path(paths['output'],'data',  sim_name)
		images_sim_output = Path(paths['output'],'images',sim_name)


		# extrapolation periods
		periods = 20


		#########################################################################################
		#########################################################################################
		###############################################
		# load model
		model.eval()


		rhs_obj = model.ode.rhs

		with np.printoptions(precision=2, suppress=True):
			print("Scales: ", rhs_obj.scales.detach().numpy())
			# print("eigenvalue shift:  ", rhs_obj.eigshift[0].detach().numpy())
		# f = open("out/data/eigspread_datasteps%d.csv"%args.datasteps,'ab')
		# np.savetxt( f, np.array([args.theta, rhs_obj.scales[0][0].item(), rhs_obj.scales[0][1].item()]).reshape((1,3)), delimiter=',')
		# f.close()


		#########################################################################################
		#########################################################################################
		###############################################
		# prepare data

		y0_train = y0_train.reshape((-1,2)) #.detach().numpy().flatten()
		y0_valid = y0_valid.reshape((-1,2))
		t_train  = t_train.reshape((-1,args.steps+1))
		y_train  = y_train.reshape((-1,args.steps+1,2))
		y_valid  = y_valid.reshape((-1,args.steps+1,2))


		t_true, y0, y_true = get_data(T=periods*args.T, steps=periods*args.steps)
		y0, y_true = y0[1::2,...], y_true[1::2,...]


		y_learned = model(y0_train).detach().numpy()
		np.savetxt( Path(data_sim_output,'solution.csv'),           np.hstack( (t_train.reshape((-1,1)), np.concatenate(y_learned,axis=1)) ), delimiter=',')
		np.savetxt( Path(paths['output_data'],'training_data.csv'), np.hstack( (t_train.reshape((-1,1)), np.concatenate(y_train.detach().numpy(),axis=1)) ),   delimiter=',')
		for _ in range(periods-1):
			y00 = torch.from_numpy( y_learned[:,-1,:] )
			y_learned = np.concatenate( (y_learned, model(y00).detach().numpy()[:,1:,:]), axis=1 )
		# np.savetxt( "out/data/ode_solution.csv",  np.vstack((t_true,y_true[4,:,0],y_true[4,:,1])).T,       delimiter=',', header='t,y,z', comments='')
		np.savetxt( Path(data_sim_output,'extrap_solution.csv'), np.hstack( (t_true.reshape((-1,1)), np.concatenate(y_learned,axis=1)) ), delimiter=',')


		###############################################
		# continuous solution
		# fig = plt.figure(fig_no); fig_no += 1

		steps = periods*1000
		h     = periods*args.T / steps

		# true ode
		t_ode, y0_true_ode, y_true_ode = get_data(periods*args.T, steps)
		y0_true_ode, y_true_ode = y0_true_ode[1::2,...], y_true_ode[1::2,...]
		y_true_ode = y_true_ode.detach().numpy().reshape((len(y_true_ode),-1,2))


		# with learned rhs
		y_learned_ode = []
		for i in range(len(y_true_ode)):
			y = [y0_true_ode[i:i+1]]
			for step in range(steps):
				y.append( y[-1] + h*rhs_obj(0.0, y[-1]) )
			y_learned_ode.append( torch.stack(y).movedim(0,1).detach().numpy() )
		y_learned_ode = np.vstack(y_learned_ode)
		# np.savetxt( "out/data/ode.csv",           np.vstack((t_ode[::10],y_true_ode[4,::10,0],y_true_ode[4,::10,1])).T,         delimiter=',', header='t,y,z', comments='')
		# np.savetxt( data_name+"learned_ode.csv",  np.vstack((t_ode[::10],y_learned_ode[4,::10,0],y_learned_ode[4,::10,1])).T,   delimiter=',', header='t,y,z', comments='')
		np.savetxt( Path(paths['output_data'],'ode.csv'),    np.hstack( (t_ode.reshape((-1,1))[::10,:], np.concatenate(y_true_ode,axis=1)[::10,...]) ),    delimiter=',')
		np.savetxt( Path(data_sim_output,'learned_ode.csv'), np.hstack( (t_ode.reshape((-1,1))[::10,:], np.concatenate(y_learned_ode,axis=1)[::10,...]) ), delimiter=',')
		# exit()

		# theta scheme for exact rhs
		fig = plt.figure(fig_no); fig_no += 1
		# for theta in [0.0,0.25,0.5,0.75,1.0]:
		for theta in [0.0, 0.5, 1.0]:
			t_theta, y_theta = theta_solver(correct_rhs, 10, 50, theta, np.zeros_like(y0.detach().numpy()), y0.detach().numpy())
			# plt.plot(y_theta[0,:,0], y_theta[0,:,1])
			# plt.plot(y_theta[-1,:,0], y_theta[-1,:,1])
		# 	np.savetxt( ("out/data/ode_theta%4.2f"%(theta)).replace('.','')+".csv", np.vstack((t_theta[0],y_theta[0])).T, delimiter=',')
		# plt.show()
		# exit()

		step = args.steps//args.datasteps
		for i in range(len(y_true)):
			plt.plot(t_ode, y_true_ode[i,:,0],       '-r')
			plt.plot(t_ode, y_learned_ode[i,:,0],    '-b')
			plt.plot(t_true[::step], y_true[i,::step,0],'or')
		# plt.xlim(0, args.T)
		# plt.ylim(-0.5, 1.5)
		plt.title('Continuous solutions')


		fig = plt.figure(fig_no); fig_no += 1
		for i in range(len(y_true)):
			plt.plot(y_true_ode[i,:,0],    y_true_ode[i,:,1],       '-r')
			plt.plot(y_learned_ode[i,:,0], y_learned_ode[i,:,1],    '-b')
			plt.plot(y_true[i,::step,0],   y_true[i,::step,1],'or')



		###############################################
		# plot trajectories
		fig = plt.figure(fig_no); fig_no += 1

		step = args.steps//args.datasteps
		plt.plot(t_true, y_true[4,:,0],      '-k')
		plt.plot(t_true, y_learned[4,...,0], '-r')
		# for i in range(len(y_true)):
		# 	plt.plot(t_true, y_true[i,:,0],      '-k')
		# 	plt.plot(t_true, y_learned[i,...,0], '-r')
			# plt.plot(t_true[::step], y_true[i,::step,0], 'or')
		# plt.xlim(0, periods*args.T)
		# plt.ylim(-0.5, 1.5)
		plt.title('Trajectories')


		fig = plt.figure(fig_no); fig_no += 1
		for i in range(len(y_true)):
			plt.plot(y_true_ode[i,:,0],  y_true_ode[i,:,1],       '-r')
			plt.plot(y_learned[i,:,0],   y_learned[i,:,1],    '-b')
			plt.plot(y_true[i,::step,0], y_true[i,::step,1],'or')


		###############################################
		# plot vector field

		# evaluate vector fields
		X = np.linspace(0, 3, 25).reshape(-1,1)
		Y = np.linspace(0, 2, 25).reshape(-1,1)
		X,Y = np.meshgrid(X,Y)
		XY  = np.hstack((X.reshape(-1,1), Y.reshape(-1,1)))
		UV_ode = []
		UV     = []
		for i in range(len(X)):
			UV_ode.append(correct_rhs(0, XY).reshape((-1,2)))
			UV.append(rhs_obj(0.0, torch.from_numpy(XY).float()).detach().numpy().reshape((-1,2)))
		UV_ode = np.hstack(UV_ode)
		UV     = np.hstack(UV)

		# exact vector field
		fig = plt.figure(fig_no); fig_no += 1
		plt.quiver(X, Y, UV_ode[:,0], UV_ode[:,1], angles='xy')
		plt.xlim(0, 3)
		plt.ylim(0, 2)
		plt.gca().axes.xaxis.set_visible(False)
		plt.gca().axes.yaxis.set_visible(False)
		plt.gca().axis('off')
		plt.savefig(Path(paths['output_images'],'vector_field.pdf'), bbox_inches='tight', pad_inches=0.0)



		# plot training data
		steps = 1000
		t_ode, y0_true_ode, y_true_ode = get_data(args.T, steps)
		y0_true_ode, y_true_ode = y0_true_ode[1::2,...], y_true_ode[1::2,...]
		y_true_ode = y_true_ode.detach().numpy().reshape((len(y_true_ode),-1,2))
		y0_true_ode, y_true_ode = y0_true_ode[1::2,...], y_true_ode[1::2,...]
		for i in range(len(y_true_ode)):
			plt.plot(y_true_ode[i,:,0], y_true_ode[i,:,1], '-b')
			plt.plot(y0_train[i,0],     y0_train[i,1],     'or', markersize=5.0)
			# plt.plot(y_train[i,:,0],    y_train[i,:,1],    'or')
		plt.xlim(0, 3)
		plt.ylim(0, 2)
		plt.gca().axes.xaxis.set_visible(False)
		plt.gca().axes.yaxis.set_visible(False)
		plt.gca().axis('off')
		plt.savefig(Path(paths['output_images'],'training_data.pdf'), bbox_inches='tight', pad_inches=0.0)


		# learned vector field
		fig = plt.figure(fig_no); fig_no += 1
		plt.quiver(X, Y, UV[:,0], UV[:,1], angles='xy')
		plt.xlim(0, 3)
		plt.ylim(0, 2)
		plt.gca().axes.xaxis.set_visible(False)
		plt.gca().axes.yaxis.set_visible(False)
		plt.gca().axis('off')
		plt.savefig(Path(images_sim_output,'vector_field.pdf'), bbox_inches='tight', pad_inches=0.0)




		###############################################
		# evaluate spectrum

		t_true, y0, y_true = get_data()
		y0, y_true = y0[1::2,...], y_true[1::2,...]
		y_true = y_true.movedim(0,1)

		spectrum = []
		for t in range(len(y_true)):
			spectrum.append( rhs_obj.spectrum(t, y_true[t]) )
		spectrum = np.concatenate(np.array(spectrum))

		xmax = ymax = 3
		xmax = max( np.amax(np.abs(spectrum[...,0])), xmax );  ymax = max( np.amax(np.abs(spectrum[...,1])), ymax )
		xmax = max(xmax,ymax); ymax = max(xmax,ymax)

		# np.savetxt( data_name+"_eig1.txt", np.sqrt(spectrum_test[-1,1::2,0]**2+spectrum_test[-1,1::2,1]**2), delimiter=',')
		# np.savetxt( data_name+"_eig2.txt", np.sqrt(spectrum_test[-1,::2,0]**2+spectrum_test[-1,::2,1]**2), delimiter=',')


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
			# if not math.isnan(args.eigs[0]):
			# 	plt.axvline(x=args.eigs[0], color='green')
			# if math.isnan(args.eigs[1]) and args.theta>0.2:
			# 	plt.axvline(x=1/args.theta, color='green')
			# else:
			# 	plt.axvline(x=args.eigs[1], color='green')


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
		plt.savefig(Path(images_sim_output,'spectrum.pdf'), bbox_inches='tight', pad_inches=0.0)



		# plt.show()
		# exit()









