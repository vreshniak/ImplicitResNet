import time
import math
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla
import scipy.optimize as opt
import matplotlib.pyplot as plt

import torch
from torch.autograd import Function
from torch.nn import Linear, ReLU, Conv2d, Module, Sequential
from torch.nn.functional import linear, conv2d, conv_transpose2d
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter





_dtype  = torch.float
_device = torch.device("cpu")

# default_writer = SummaryWriter()



_sqrt_2 = 1.414213562373095048801688724209698078569671875376948073176



###############################################################################
###############################################################################

class training_loop:
	def __init__(self,model,dataset,val_dataset,batch_size,loss_fn,accuracy_fn=None,optimizer=None,regularizer=None,scheduler=None,writer=None,write_hist=False,history=False):
		self.model = model
		self.optimizer = optimizer
		# batch_shuffle = True
		# if batch_size is None or batch_size<0:
		# 	batch_size = len(val_dataset)
		# 	batch_shuffle = False
		# self.dataloader     = torch.utils.data.DataLoader(dataset,     batch_size=batch_size,       shuffle=batch_shuffle)
		# self.val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
		self.batch_size = batch_size
		self.dataset = dataset
		self.val_dataset = val_dataset
		self.device = model.parameters().__next__().device
		self.loss_fn = loss_fn
		self.accuracy_fn = accuracy_fn
		self.regularizer = regularizer
		self.scheduler = scheduler
		self.writer = writer
		self.write_hist = write_hist
		self.history = history
		self.curr_epoch = 0

	def __call__(self,num_epochs,optimizer=None,scheduler=None,regularizer=None):
		self.curr_epoch += num_epochs
		if optimizer is not None:   self.optimizer   = optimizer
		if scheduler is not None:   self.scheduler   = scheduler
		if regularizer is not None: self.regularizer = regularizer
		if self.history:
			return train(self.model,self.optimizer,num_epochs,self.dataset,self.val_dataset,self.batch_size,self.loss_fn,self.accuracy_fn,self.regularizer,self.scheduler,self.writer,self.write_hist,self.curr_epoch-num_epochs,self.history)
		else:
			train(self.model,self.optimizer,num_epochs,self.dataset,self.val_dataset,self.batch_size,self.loss_fn,self.accuracy_fn,self.regularizer,self.scheduler,self.writer,self.write_hist,self.curr_epoch-num_epochs,self.history)



def train(model,optimizer,num_epochs,dataset,val_dataset,batch_size,loss_fn,accuracy_fn=None,regularizer=None,scheduler=None,writer=None,write_hist=False,epoch0=0,history=False):
	val_freq  = 1
	hist_freq = 10
	if history:
		hist_i = 0
		loss_history = -1*np.ones((3,np.arange(0,num_epochs+1,val_freq).size))
		# loss_history     = -1*np.ones((num_epochs,))
		# val_loss_history = -1*np.ones((num_epochs,))
		# val_freq = 1

	batch_shuffle = True
	if batch_size is None or batch_size<0:
		batch_size = len(val_dataset)
		batch_shuffle = False
	dataloader     = torch.utils.data.DataLoader(dataset,     batch_size=batch_size,       shuffle=batch_shuffle)
	val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
	device = model.parameters().__next__().device
	for epoch in range(num_epochs):
		# initial validation loss/accuracy
		if epoch==0: sec = 0

		#######################################################
		# training loop
		start = time.time()
		model = model.train()
		for batch_ndx, sample in enumerate(dataloader):
			optimizer.zero_grad()

			x, y   = sample[0].to(device), sample[1].to(device)
			y_pred = model.forward(x)
			loss = loss_fn( y_pred, y )
			if regularizer is not None:
				reg_loss = regularizer(model)
				(loss + reg_loss).backward()
			# if regularizer is not None:
			# 	(loss + regularizer(model)).backward()
			else:
				loss.backward()
			optimizer.step()

		if scheduler is not None:
			if type(scheduler)==torch.optim.lr_scheduler.ReduceLROnPlateau:
				scheduler.step(loss)
			else:
				scheduler.step()
		sec += (time.time() - start)
		#######################################################

		# validation error/accuracy
		model = model.eval()
		if epoch%val_freq==0 or epoch==num_epochs-1:
			for batch_ndx, sample in enumerate(val_dataloader):
				val_x, val_y = sample[0].to(device), sample[1].to(device)
				val_y_pred   = model.forward(val_x)
				val_loss     = loss_fn( val_y_pred, val_y )
				if accuracy_fn is not None:
					acc     = accuracy_fn( y_pred,     y     )
					val_acc = accuracy_fn( val_y_pred, val_y )
			if writer is not None:
				writer.add_scalar('loss/train',       loss,     epoch+epoch0)
				writer.add_scalar('loss/validation',  val_loss, epoch+epoch0)
				if regularizer is not None:
					writer.add_scalar('loss/regularizer', reg_loss, epoch+epoch0)
				if accuracy_fn is not None:
					writer.add_scalar('accuracy/train',      acc,     epoch+epoch0)
					writer.add_scalar('accuracy/validation', val_acc, epoch+epoch0)
			if accuracy_fn is not None:
				print("Epoch %d: %4.2f sec, loss %5.2e, val_loss %5.2e, acc %4.2f, val_acc %4.2f "%(epoch, sec, loss, val_loss, acc, val_acc))
			else:
				print("Epoch %d: %4.2f sec, loss %5.2e, val_loss %5.2e"%(epoch, sec, loss, val_loss))
			if history:
				loss_history[0,hist_i] = epoch
				loss_history[1,hist_i] = loss
				loss_history[2,hist_i] = val_loss
				hist_i += 1
			sec = 0.0
		if write_hist:
			if epoch%hist_freq==0 or epoch==num_epochs-1:
				if writer is not None:
					for name, weight in model.named_parameters():
						writer.add_histogram('parameters/'+name, weight,      epoch+epoch0, bins='tensorflow')
						writer.add_histogram('gradients/'+name,  weight.grad, epoch+epoch0, bins='tensorflow')
	if history:
		return loss_history


def initialize(model,weight_initializer=torch.nn.init.xavier_uniform_, bias_initializer=torch.nn.init.zeros_):
	for name, weight in model.named_parameters():
		if 'weight' in name:
			weight_initializer(weight)
		elif 'bias' in name:
			bias_initializer(weight)



###############################################################################
###############################################################################




# class TV_regularizer:
# 	def __init__(self,model,alpha):
# 		self.model = model
# 		self.alpha = alpha

# 	def __call__(self):
# 		R = None
# 		for name, weight in self.model.named_parameters():
# 			if 'weight' in name:
# 				if R is not None:
# 					R += torch.norm( weight - old_weight, p='fro', dim=None, keepdim=False )
# 				else:
# 					# R = torch.norm( weight - weight, p=1, dim=None, keepdim=False )
# 					R = 0
# 				old_weight = weight
# 		return self.alpha * R




###############################################################################
###############################################################################





# class implicit_correction(Function):
# 	@staticmethod
# 	def forward(ctx,obj,y):
# 		ctx.save_for_backward(y)
# 		ctx.obj = obj
# 		return y

# 	@staticmethod
# 	def backward(ctx,dy):
# 		y, = ctx.saved_tensors
# 		batch_size, noutputs = y.size()
# 		with torch.enable_grad():
# 			yb = y.unsqueeze(1).repeat(1,noutputs,1)
# 			z  = ctx.obj.StepFun(yb)
# 			# z  = yb - ctx.obj.theta * ctx.obj.StepFun(yb)
# 		# A    = torch.autograd.grad(z, yb, grad_outputs=torch.eye(noutputs).repeat(batch_size,1,1))[0]
# 		A    = ctx.obj.Id - ctx.obj.theta * torch.autograd.grad(z, yb, grad_outputs=ctx.obj.Id)[0]
# 		x, _ = torch.solve( dy.unsqueeze(2), A.transpose(-1,-2) )
# 		# print( A.detach().numpy() )
# 		# x = la.solve( A.detach().numpy(), dy.unsqueeze(2).detach().numpy(), transposed=True )
# 		return None, x.squeeze(2)


# def nsolve(self,y0,explicit):
# 	# print( omp.current_process().pid )
# 	return self.nsolve(y0,explicit)


# class ResNet_step(Module):
# 	def __init__(self, fun=ReLU, h=1, theta=0.5, maxniter=10):
# 		super(ResNet_step,self).__init__()
# 		self.h        = h
# 		self.theta    = theta
# 		self.F        = fun()
# 		self.maxniter = maxniter

# 	def set_theta(self,theta):
# 		self.theta = theta

# 	class implicit_correction(Function):
# 		@staticmethod
# 		def forward(ctx,obj,y):
# 			ctx.save_for_backward(y)
# 			ctx.obj = obj
# 			return y

# 		@staticmethod
# 		def backward(ctx,dy):
# 			y,     = ctx.saved_tensors
# 			ndof   = y.flatten().size()[0]
# 			device = ctx.obj.parameters().__next__().device

# 			with torch.enable_grad():
# 				z = ctx.obj.StepFun(y)
# 			def mv(v):
# 				v0 = torch.from_numpy(v).view_as(y).to(device=device,dtype=_dtype)
# 				Av = v0 - ctx.obj.theta * torch.autograd.grad(z, y, grad_outputs=v0, retain_graph=True, only_inputs=True)[0]
# 				return Av.cpu().detach().numpy().ravel()
# 			A = sla.LinearOperator(dtype=np.float, shape=(ndof,ndof), matvec=mv)

# 			dy_cpu = (dy + ctx.obj.theta * torch.autograd.grad(z, y, grad_outputs=dy, retain_graph=True, only_inputs=True)[0]).cpu().detach().numpy().ravel()
# 			# dy_cpu = dy.cpu().detach().numpy().ravel()
# 			dx, info = sla.lgmres( A, dy_cpu, x0=dy_cpu, maxiter=100, atol=1.e-5, M=None )
# 			return None, torch.from_numpy(dx).view_as(dy).to(device=device,dtype=_dtype)
# 			# return None, (dy + ctx.obj.theta * torch.autograd.grad(z, y, grad_outputs=dy, retain_graph=False, only_inputs=True)[0])

# 	def fixed_point(self,y0,explicit):
# 		def functional(z0):
# 			z1 = explicit + self.theta * self.StepFun(z0)
# 			return z1, ((z1-z0)**2).sum()

# 		iters = 0
# 		y = y0
# 		while True:
# 			y, residual = functional(y)
# 			iters += 1
# 			if residual<=1.e-4:
# 				break
# 		return y, residual, iters

# 	def nsolve(self,y0,explicit):
# 		def functional(z):
# 			with torch.enable_grad():
# 				F = self.StepFun(z)
# 			residual = z - explicit - self.theta * F
# 			fun  = 0.5 * torch.sum( residual**2 )
# 			dfun = (residual - self.theta*torch.autograd.grad(F, z, grad_outputs=residual)[0])
# 			return fun, dfun

# 		alpha = 1.e-1
# 		y = y0
# 		y.requires_grad_(True)
# 		for i in range(10):
# 			f, df = functional(y)
# 			y = y - alpha * df
# 		f, df = functional(y)

# 		return y, f, 10

# 	def scipy_nsolve(self,y0,explicit):
# 		device = self.parameters().__next__().device
# 		def functional(z):
# 			z0 = torch.from_numpy(z).view_as(explicit).to(device=device,dtype=_dtype).requires_grad_(True)
# 			F  = self.StepFun(z0)
# 			residual = z0 - explicit - self.theta * F
# 			fun  = 0.5 * (residual**2).sum()
# 			# dfun = residual - self.theta * torch.autograd.grad(F, z0, grad_outputs=residual, retain_graph=True)[0]
# 			dfun = torch.autograd.grad(fun, z0)[0]
# 			return fun.cpu().detach().numpy(), dfun.cpu().detach().numpy().ravel()
# 		opt_res = opt.minimize(functional, y0.cpu().numpy(), method='CG', jac=True, tol=1.e-4, options={'maxiter': self.maxniter,'disp': False})
# 		return torch.from_numpy(opt_res.x).view_as(y0).to(device=device,dtype=_dtype), opt_res.fun, opt_res.nit

# 	def StepFun(self, x):
# 		return self.h * self.F(x)

# 	def forward(self, x):
# 		batch_size = x.size()[0]

# 		explicit = x + (1-self.theta) * self.StepFun(x)
# 		if self.theta>0:
# 			y, fmin, iters = self.scipy_nsolve(y0=(x+self.StepFun(x)).detach(),explicit=explicit.detach())
# 			# y, fmin, iters = self.fixed_point(y0=explicit.detach(),explicit=explicit.detach())
# 			# y, fmin, iters = self.nsolve(y0=explicit.detach(),explicit=explicit.detach())

# 			# y    = torch.empty_like(x, dtype=_dtype)
# 			# y_np = y.detach().numpy()
# 			#
# 			# fmin  = np.zeros((batch_size,))
# 			# iters = np.zeros((batch_size,))
# 			# for batch in range(batch_size):
# 			# 	y_np[batch,...], fmin[batch] = self.nsolve(y0=x.detach()[batch],explicit=explicit.detach()[batch].unsqueeze(0))
# 			# with omp.Pool(processes=torch.get_num_threads()) as pool:
# 			# with omp.Pool() as pool:
# 			# 	for batch in range(batch_size):
# 			# 		y_np[batch,...], fmin = pool.apply_async( nsolve, args=(self,x.detach()[batch],explicit.detach()[batch],) ).get()

# 			return ResNet_step.implicit_correction.apply(self, explicit + self.theta*self.StepFun(y).detach() ), fmin, iters
# 			# return explicit + self.theta*self.StepFun(y).detach(), fmin, iters
# 		else:
# 			return explicit, 0, 0


# class TV_regularizer:
# 	def __init__(self,model,alpha):
# 		self.model = model
# 		self.alpha = alpha

# 	def __call__(self):
# 		R = None
# 		for name, weight in self.model.named_parameters():
# 			if 'weight' in name:
# 				if R is not None:
# 					R += torch.norm( weight - old_weight, p='fro', dim=None, keepdim=False )
# 				else:
# 					# R = torch.norm( weight - weight, p=1, dim=None, keepdim=False )
# 					R = 0
# 				old_weight = weight
# 		return self.alpha * R


# class ImplicitNet(Module):
# 	def __init__(self, fun=None, num_layers=10, h=1, theta=0.0, writer=None, alpha=0):
# 		super(ImplicitNet,self).__init__()
# 		self.fun        = fun
# 		self.h          = h
# 		self.theta      = theta
# 		self.num_layers = num_layers
# 		self.writer     = writer
# 		self.fstep      = 0
# 		self.wr_freq    = 10
# 		self.alpha		= alpha

# 		self.layers = torch.nn.ModuleList()
# 		for t in range(self.num_layers):
# 			self.layers.append( ResNet_step(fun=fun,h=h,theta=theta) ) #,writer=writer) )

# 	def forward(self, y):
# 		residuals = np.zeros((self.num_layers,),dtype=np.float)
# 		iters     = np.zeros((self.num_layers,),dtype=np.float)
# 		for t in range(self.num_layers):
# 			y, residuals[t], iters[t] = self.layers[t](y)

# 		if self.writer is not None and (self.fstep%(self.wr_freq))==0:
# 				self.writer.add_scalar('solvers/nsolve/max_residual',  np.amax(residuals), self.fstep)
# 				self.writer.add_scalar('solvers/nsolve/max_iters',     np.amax(iters),     self.fstep)
# 				self.writer.add_scalar('solvers/nsolve/mean_residual', np.mean(residuals), self.fstep)
# 				self.writer.add_scalar('solvers/nsolve/mean_iters',    np.mean(iters),     self.fstep)
# 		self.fstep += 1
# 		if self.alpha==0:
# 			return y
# 		else:
# 			return y + self.TV_weights(self.alpha)

# 	def init_from_coarse(self,coarse_model):
# 		for t in range(coarse_model.num_layers):
# 			for w_new, w_old in zip( self.layers[2*t].parameters(), coarse_model.layers[t].parameters() ):
# 				w_new.detach().copy_(w_old.detach())
# 		for t in range(coarse_model.num_layers-1):
# 			for w_new, w_old1, w_old2 in zip( self.layers[2*t+1].parameters(), coarse_model.layers[t].parameters(), coarse_model.layers[t+1].parameters()):
# 				torch.lerp(w_old1.detach(), w_old2.detach(), 0.5, out=w_new.detach())


# 	def TV_weights(self, alpha=0.1):
# 		R = 0
# 		for name, weight in self.layers[0].named_parameters():
# 			if 'weight' in name:
# 				old_weight = weight
# 		for t in range(1,self.num_layers):
# 			for name, weight in self.layers[t].named_parameters():
# 				if 'weight' in name:
# 					R += torch.norm( weight - old_weight, p=1, dim=None, keepdim=False )
# 					old_weight = weight
# 		return alpha * R

# 	def set_theta(self,theta):
# 		for t in range(self.num_layers):
# 			self.layers[t].set_theta(theta)




# def refine_model(coarse_model,keep_old=False):
# 	model = []
# 	if type(coarse_model)==Sequential:
# 		for m in coarse_model:
# 			if type(m)==ResNet:
# 				model.append( ResNet(fun=m.fun, num_layers=2*(m.num_layers-1)+1, h=0.5*m.h, theta=m.theta, writer=m.writer) )
# 				model[-1].init_from_coarse(m)
# 				if not keep_old:
# 					print("How to delete module?")
# 			else:
# 				model.append(m)
# 	else:
# 		exit("coarse_model must be Sequential")
# 	return Sequential(*model)




###############################################################################
###############################################################################



class Antisym(Linear):
	def forward(self, inputs):
		return linear(inputs, self.weight-self.weight.transpose(0,1), self.bias)


class AntisymConv2d(Conv2d):
	def forward(self, inputs):
		return conv2d(inputs, self.weight, self.bias, padding=self.padding) - conv_transpose2d(inputs, self.weight, self.bias, padding=self.padding)



###############################################################################
###############################################################################


class ResNet(Module):
	def __init__(self, params=None, width=10, num_layers=10, h=1, theta=0.0, fixed_point_iters=0, power_iters=0, basis='haar', init_domain='original', init_level=None, writer=None):
		super(ResNet,self).__init__()
		self.params     = params
		self.basis      = basis
		self.h          = h
		self.theta      = theta
		self.width      = width
		self.num_layers = num_layers
		self.writer 	= writer
		self.level      = init_level
		self.fstep      = 0
		self.activation = ReLU()
		self.fixed_point_iters = fixed_point_iters
		self.power_iters       = power_iters

		assert params is not None, "params cannot be None"

		if params['type']=='Conv2d':
			# initialize weights in the original domain
			W = torch.empty((num_layers,params['channels'],params['channels'],params['kernel_size'],params['kernel_size']), dtype=_dtype, device=_device)
			b = torch.empty((num_layers,params['channels']), dtype=_dtype, device=_device)
			torch.nn.init.xavier_uniform_(W,gain=0.1)
			torch.nn.init.xavier_uniform_(b,gain=0.1)

			# initialize weights in the transform domain
			if init_domain=='transform' or basis is None:
				self.W_c = torch.nn.Parameter(W, requires_grad=True)
				self.b_c = torch.nn.Parameter(b, requires_grad=True)


	def forward(self, y):
		self.fstep += 1

		#######################################################################
		# change coordinates back to original domain
		if self.basis is None:
			W = self.W_c
			b = self.b_c

		#######################################################################
		# spectral normalization
		if self.power_iters>0:
			with torch.no_grad():
				# estimate left and right sigular vectors
				for _ in range(self.power_iters):
					self.W_v = torch.nn.functional.normalize(torch.bmm(W.transpose(1,2), self.W_u), dim=1, eps=1.e-12)
					self.W_u = torch.nn.functional.normalize(torch.bmm(W,                self.W_v), dim=1, eps=1.e-12)
			# estimate spectral norm
			W_sigma = torch.bmm(self.W_u.transpose(1,2), torch.bmm(W, self.W_v))
			# normalize weight matrix
			W = W / W_sigma
			self.writer.add_scalar('solvers/mean_sigma', W_sigma.mean(), self.fstep)

		#######################################################################
		# time stepping

		if self.params['type']=='Conv2d':
			def rhs(t,x):
				return self.h * self.activation(conv2d(x,W[t]-W[t].flip([2,3]),b[t],padding=self.params['kernel_size']//2))
		if self.params['type']=='Dense':
			def rhs(t,x):
				return self.h * self.activation(linear(x,W[t]-W[t].t(),b[t]))

		self.cum_residual = max_residual = 0
		if self.theta>0 and self.fixed_point_iters>0:
			for t in range(self.num_layers):
				explicit = y + (1-self.theta) * rhs(t,y)
				# fixed point iterations
				for _ in range(self.fixed_point_iters):
					y = explicit + self.theta * rhs(t,y)
				if self.training:
					# choose active constraints
					residual = (y-explicit-self.theta*rhs(t,y)).pow(2).sum()
					if residual>=1.e-7:  self.cum_residual = self.cum_residual + residual
					if residual>max_residual: max_residual = residual
			if self.training:
				self.writer.add_scalar('solvers/cum_residual', self.cum_residual, self.fstep)
				self.writer.add_scalar('solvers/max_residual', max_residual,      self.fstep)
		else:
			for t in range(self.num_layers):
				y = y + rhs(t,y)
		return y

	def plot_weights(self):
		import matplotlib.pyplot as plt
		W0 = self.W_c.abs().detach().cpu().numpy().ravel()
		b0 = self.b_c.abs().detach().cpu().numpy().ravel()
		W = np.flip(np.sort(W0))
		b = np.flip(np.sort(b0))
		for i in range(W.size):
			self.writer.add_scalar('misc/decay/weight', W[i], i)
		for i in range(b.size):
			self.writer.add_scalar('misc/decay/bias', b[i], i)
		return W0, b0

	def regularizer(self, alpha=0.1, beta=1.0, norm='l1'):
		reg_value = 0
		if beta>0 and self.theta>0:
			reg_value = reg_value + beta * self.cum_residual
		return reg_value


