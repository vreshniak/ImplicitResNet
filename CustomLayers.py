import time
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






###############################################################################
###############################################################################


# def train(model,optimizer,num_epochs,dataloader,val_dataset,loss_fn,regularizer=None,scheduler=None,writer=None,epoch0=0):
def train(model,optimizer,num_epochs,dataset,val_dataset,batch_size,loss_fn,accuracy_fn=None,regularizer=None,scheduler=None,writer=None,write_hist=False,epoch0=0): #,device=torch.device('cpu')):
	# scheduler = ReduceLROnPlateau(optimizer, 'min')
	dataloader     = torch.utils.data.DataLoader(dataset,     batch_size=batch_size,       shuffle=True) if batch_size is not None else \
					 torch.utils.data.DataLoader(dataset,     batch_size=len(dataset),     shuffle=False)
	val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
	# optimizer = optimizer(model)
	device = model.parameters().__next__().device
	for epoch in range(num_epochs):
		# initial validation loss/accuracy
		if epoch==0:
			sec = 0
			for batch_ndx, sample in enumerate(val_dataloader):
				val_x, val_y = sample[0].to(device), sample[1].to(device)
				val_loss     = loss_fn( model.forward(val_x), val_y )
				if accuracy_fn is not None:
					val_acc  = accuracy_fn( model.forward(val_x), val_y )
					print("Epoch %d: val_loss %f, val_acc %f "%(epoch, val_loss, val_acc))
				else:
					print("Epoch %d: val_loss %f "%(epoch, val_loss))
			if writer is not None:
				writer.add_scalar('loss/validation', val_loss, epoch+epoch0)
				if accuracy_fn is not None:
					writer.add_scalar('accuracy/validation', val_acc, epoch+epoch0)
				if write_hist:
					for name, weight in model.state_dict().items():
						writer.add_histogram('parameters/'+name, weight, epoch+epoch0, bins='tensorflow')

		#######################################################
		# training loop
		start = time.time()
		for batch_ndx, sample in enumerate(dataloader):
			optimizer.zero_grad()

			x, y = sample[0].to(device), sample[1].to(device)
			loss = loss_fn( model.forward(x), y )
			if regularizer is not None:
				(loss + regularizer(model)).backward()
			else:
				loss.backward()
			if accuracy_fn is not None:
				acc = accuracy_fn( model.forward(x), y )

			optimizer.step()
		if scheduler is not None:
			scheduler.step()
		sec += (time.time() - start)
		#######################################################

		# print history
		if writer is not None:
			writer.add_scalar('loss/train', loss, epoch+epoch0)

		if (epoch%10==0 and epoch>0) or epoch==num_epochs-1:
			for batch_ndx, sample in enumerate(val_dataloader):
				val_x, val_y = sample[0].to(device), sample[1].to(device)
				val_loss     = loss_fn( model.forward(val_x), val_y )
				if accuracy_fn is not None:
					val_acc  = accuracy_fn( model.forward(val_x), val_y )
					print("Epoch %d: %4.2f sec, loss %5.2e, val_loss %5.2e, acc %4.2f, val_acc %4.2f "%(epoch, sec, loss, val_loss, acc, val_acc))
				else:
					print("Epoch %d: %4.2f sec, loss %5.2e, val_loss %5.2e"%(epoch, sec, loss, val_loss))
				sec = 0.0
			if writer is not None:
				writer.add_scalar('loss/validation', val_loss, epoch+epoch0)
				if accuracy_fn is not None:
					writer.add_scalar('accuracy/train',      acc,     epoch+epoch0)
					writer.add_scalar('accuracy/validation', val_acc, epoch+epoch0)
		if write_hist:
			if (epoch%100==0 and epoch>0) or epoch==num_epochs-1:
				if writer is not None:
					for name, weight in model.named_parameters():
						writer.add_histogram('parameters/'+name, weight, epoch+epoch0, bins='tensorflow')
						writer.add_histogram('gradients/'+name, weight.grad, epoch+epoch0, bins='tensorflow')


def initialize(model,weight_initializer=torch.nn.init.xavier_uniform_, bias_initializer=torch.nn.init.zeros_):
	for name, weight in model.named_parameters():
		if 'weight' in name:
			weight_initializer(weight)
		elif 'bias' in name:
			bias_initializer(weight)



# class customModule(Module):
# 	def initialize(self,weight_initializer=torch.nn.init.xavier_uniform_, bias_initializer=torch.nn.init.zeros_):
# 		initialize(self,weight_initializer,bias_initializer)
# 	def train(self,optimizer,num_epochs,dataloader,val_dataset,loss_fn,regularizer=None,writer=None,epoch0=0):
# 		train(self,optimizer,num_epochs,dataloader,val_dataset,loss_fn,regularizer,writer,epoch0)



# class customSequential(Sequential):
# 	def initialize(self,weight_initializer=torch.nn.init.xavier_uniform_, bias_initializer=torch.nn.init.zeros_):
# 		initialize(self,weight_initializer,bias_initializer)
# 	def train(self,optimizer,num_epochs,dataloader,val_dataset,loss_fn,regularizer=None,writer=None,epoch0=0):
# 		train(self,optimizer,num_epochs,dataloader,val_dataset,loss_fn,regularizer,writer,epoch0)



###############################################################################
###############################################################################




class TV_regularizer:
	def __init__(self,model,alpha):
		self.model = model
		self.alpha = alpha

	def __call__(self):
		R = None
		for name, weight in self.model.named_parameters():
			if 'weight' in name:
				if R is not None:
					R += torch.norm( weight - old_weight, p='fro', dim=None, keepdim=False )
				else:
					# R = torch.norm( weight - weight, p=1, dim=None, keepdim=False )
					R = 0
				old_weight = weight
		return self.alpha * R




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


class ResNet_step(Module):
	def __init__(self, fun=ReLU, h=1, theta=0.5, maxniter=10):
		super(ResNet_step,self).__init__()
		self.h        = h
		self.theta    = theta
		self.F        = fun()
		self.maxniter = maxniter

	def set_theta(self,theta):
		self.theta = theta

	class implicit_correction(Function):
		@staticmethod
		def forward(ctx,obj,y):
			ctx.save_for_backward(y)
			ctx.obj = obj
			return y

		@staticmethod
		def backward(ctx,dy):
			y,     = ctx.saved_tensors
			ndof   = y.flatten().size()[0]
			device = ctx.obj.parameters().__next__().device

			with torch.enable_grad():
				z = ctx.obj.StepFun(y)
			def mv(v):
				v0 = torch.from_numpy(v).view_as(y).to(device=device,dtype=_dtype)
				Av = v0 - ctx.obj.theta * torch.autograd.grad(z, y, grad_outputs=v0, retain_graph=True, only_inputs=True)[0]
				return Av.cpu().detach().numpy().ravel()
			A = sla.LinearOperator(dtype=np.float, shape=(ndof,ndof), matvec=mv)

			dy_cpu = dy.cpu().detach().numpy().ravel()
			dx, info = sla.lgmres( A, dy_cpu, x0=dy_cpu, maxiter=100, atol=1.e-5, M=None )
			return None, torch.from_numpy(dx).view_as(dy).to(device=device,dtype=_dtype)
			# return None, (dy + ctx.obj.theta * torch.autograd.grad(z, y, grad_outputs=dy, retain_graph=False, only_inputs=True)[0])

	def nsolve(self,y0,explicit):
		def functional(z):
			with torch.enable_grad():
				F = self.StepFun(z)
			residual = z - explicit - self.theta * F
			fun  = 0.5 * torch.sum( residual**2 )
			dfun = (residual - self.theta*torch.autograd.grad(F, z, grad_outputs=residual)[0])
			return fun, dfun

		alpha = 1.e-1
		y = y0
		y.requires_grad_(True)
		for i in range(10):
			f, df = functional(y)
			y = y - alpha * df
		f, df = functional(y)

		return y, f, 10

	def scipy_nsolve(self,y0,explicit):
		device = self.parameters().__next__().device
		def functional(z):
			z0 = torch.from_numpy(z).view_as(explicit).to(device=device,dtype=_dtype).requires_grad_(True)
			F  = self.StepFun(z0)
			residual = z0 - explicit - self.theta * F
			fun  = 0.5 * (residual**2).sum()
			dfun = residual - self.theta * torch.autograd.grad(F, z0, grad_outputs=residual)[0]
			return fun.cpu().detach().numpy(), dfun.cpu().detach().numpy().ravel()
		opt_res = opt.minimize(functional, y0.cpu().numpy(), method='CG', jac=True, tol=1.e-4, options={'maxiter': self.maxniter,'disp': False})
		return torch.from_numpy(opt_res.x).view_as(y0).to(device=device,dtype=_dtype), opt_res.fun, opt_res.nit

	def StepFun(self, x):
		return self.h * self.F(x)

	def forward(self, x):
		batch_size = x.size()[0]

		explicit = x + (1-self.theta) * self.StepFun(x)
		if self.theta>0:
			y, fmin, iters = self.scipy_nsolve(y0=explicit.detach(),explicit=explicit.detach())
			# y, fmin, iters = self.nsolve(y0=explicit.detach(),explicit=explicit.detach())

			# y    = torch.empty_like(x, dtype=_dtype)
			# y_np = y.detach().numpy()
			#
			# fmin  = np.zeros((batch_size,))
			# iters = np.zeros((batch_size,))
			# for batch in range(batch_size):
			# 	y_np[batch,...], fmin[batch] = self.nsolve(y0=x.detach()[batch],explicit=explicit.detach()[batch].unsqueeze(0))
			# with omp.Pool(processes=torch.get_num_threads()) as pool:
			# with omp.Pool() as pool:
			# 	for batch in range(batch_size):
			# 		y_np[batch,...], fmin = pool.apply_async( nsolve, args=(self,x.detach()[batch],explicit.detach()[batch],) ).get()

			return ResNet_step.implicit_correction.apply(self, explicit + self.theta*self.StepFun(y).detach() ), fmin, iters
			# return explicit + self.theta*self.StepFun(y).detach(), fmin, iters
		else:
			return explicit, 0, 0



class ResNet(Module):
	def __init__(self, fun=None, num_layers=10, h=1, theta=0.0, writer=None):
		super(ResNet,self).__init__()
		self.fun        = fun
		self.h          = h
		self.theta      = theta
		self.num_layers = num_layers
		self.writer     = writer
		self.fstep      = 0
		self.wr_freq    = 10

		self.layers = torch.nn.ModuleList()
		for t in range(self.num_layers):
			self.layers.append( ResNet_step(fun=fun,h=h,theta=theta) ) #,writer=writer) )

	def forward(self, y):
		residuals = np.zeros((self.num_layers,),dtype=np.float)
		iters     = np.zeros((self.num_layers,),dtype=np.float)
		for t in range(self.num_layers):
			y, residuals[t], iters[t] = self.layers[t](y)

		if self.writer is not None and (self.fstep%(self.wr_freq))==0:
				self.writer.add_scalar('solvers/nsolve/max_residual',  np.amax(residuals), self.fstep)
				self.writer.add_scalar('solvers/nsolve/max_iters',     np.amax(iters),     self.fstep)
				self.writer.add_scalar('solvers/nsolve/mean_residual', np.mean(residuals), self.fstep)
				self.writer.add_scalar('solvers/nsolve/mean_iters',    np.mean(iters),     self.fstep)
		self.fstep += 1
		return y

	def init_from_coarse(self,coarse_model):
		for t in range(coarse_model.num_layers):
			for w_new, w_old in zip( self.layers[2*t].parameters(), coarse_model.layers[t].parameters() ):
				w_new.detach().copy_(w_old.detach())
		for t in range(coarse_model.num_layers-1):
			for w_new, w_old1, w_old2 in zip( self.layers[2*t+1].parameters(), coarse_model.layers[t].parameters(), coarse_model.layers[t+1].parameters()):
				torch.lerp(w_old1.detach(), w_old2.detach(), 0.5, out=w_new.detach())


	def TV_weights(self, alpha=0.1):
		R = 0
		for name, weight in self.layers[0].named_parameters():
			if 'weight' in name:
				old_weight = weight
		for t in range(1,self.num_layers):
			for name, weight in self.layers[t].named_parameters():
				if 'weight' in name:
					R += torch.norm( weight - old_weight, p=1, dim=None, keepdim=False )
					old_weight = weight
		return alpha * R

	def set_theta(self,theta):
		for t in range(self.num_layers):
			self.layers[t].set_theta(theta)




def refine_model(coarse_model,keep_old=False):
	model = []
	if type(coarse_model)==Sequential:
		for m in coarse_model:
			if type(m)==ResNet:
				model.append( ResNet(fun=m.fun, num_layers=2*(m.num_layers-1)+1, h=0.5*m.h, theta=m.theta, writer=m.writer) )
				model[-1].init_from_coarse(m)
				if not keep_old:
					print("How to delete module?")
			else:
				model.append(m)
	else:
		exit("coarse_model must be Sequential")
	return Sequential(*model)




###############################################################################
###############################################################################



class Antisym(Linear):
	def forward(self, inputs):
		return linear(inputs, self.weight-self.weight.transpose(0,1), self.bias)


class AntisymConv2d(Conv2d):
	def forward(self, inputs):
		return conv2d(inputs, self.weight, self.bias, padding=self.padding) - conv_transpose2d(inputs, self.weight, self.bias, padding=self.padding)

class Flatten(Module):
	def forward(self, inputs):
		return inputs.view(inputs.size()[0],-1)