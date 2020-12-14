import time

import torch
from torch.utils.tensorboard import SummaryWriter


# class stat_writer:
# 	def __init__(self, writer, num_steps):
# 		self.writer    = writer
# 		self.num_steps = num_steps
# 		self.counter   = 0
# 		self.stat      = {}
# 		self.fstep = self.bstep = 0

# 	def __call__(stat):
# 		self.counter += 1
# 		for key in stat.keys():
# 			self.stat['key'] = self.stat.get(key,0) + stat[key] / (self.num_steps if 'residual' in key else 1)
# 		if 'forward' in key:
# 			self.fstep

# 		if self.counter%self.num_steps == 0:
# 			for key in self.stat.keys():
# 				writer.add_scalar(key, self.stat[key], self.fstep)
# 			self.counter = 0
# 			self.stat    = {}


###############################################################################
###############################################################################

class training_loop:
	def __init__(self,model,dataset,val_dataset,batch_size,loss_fn,accuracy_fn=None,optimizer=None,regularizer=None,scheduler=None,lr_schedule=None,writer=None,write_hist=False,history=False,checkpoint=None,init_epoch=0):
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
		# self.device = model.parameters().__next__().device
		self.loss_fn = loss_fn
		self.accuracy_fn = accuracy_fn
		self.regularizer = regularizer
		self.scheduler = scheduler
		self.lr_schedule = lr_schedule
		self.writer = writer
		self.write_hist = write_hist
		self.history = history
		self.curr_epoch = init_epoch
		self.checkpoint=checkpoint

	def __call__(self,num_epochs,optimizer=None,scheduler=None,lr_schedule=None,regularizer=None,checkpoint=None,writer=None):
		self.curr_epoch += num_epochs
		if optimizer is not None:
			self.optimizer   = optimizer
			self.scheduler   = scheduler
		if scheduler is not None:
			self.scheduler   = scheduler
			self.lr_schedule = None
		if lr_schedule is not None:
			self.scheduler   = None
			self.lr_schedule = lr_schedule
		if regularizer is not None: self.regularizer = regularizer
		if checkpoint is not None:  self.checkpoint  = checkpoint
		if writer is not None:      self.writer      = writer
		if self.history:
			return train(self.model,self.optimizer,num_epochs,self.dataset,self.val_dataset,self.batch_size,self.loss_fn,self.accuracy_fn,self.regularizer,self.scheduler,self.lr_schedule,self.writer,self.write_hist,self.curr_epoch-num_epochs,self.history,self.checkpoint)
		else:
			train(self.model,self.optimizer,num_epochs,self.dataset,self.val_dataset,self.batch_size,self.loss_fn,self.accuracy_fn,self.regularizer,self.scheduler,self.lr_schedule,self.writer,self.write_hist,self.curr_epoch-num_epochs,self.history,self.checkpoint)



def train(model,optimizer,num_epochs,dataset,val_dataset,batch_size,loss_fn,accuracy_fn=None,regularizer=None,scheduler=None,lr_schedule=None,writer=None,write_hist=False,epoch0=0,history=False,checkpoint=None):
	stat_freq = 1
	val_freq  = 10
	hist_freq = 10
	if history:
		hist_i = 0
		loss_history = -1*np.ones((3,np.arange(0,num_epochs+1,val_freq).size))

	# prepare data
	val_batch_size = batch_size
	batch_shuffle  = True
	if batch_size is None or batch_size<0:
		batch_size     = len(dataset)
		val_batch_size = len(val_dataset)
		batch_shuffle  = False
	dataloader     = torch.utils.data.DataLoader(dataset,     batch_size=batch_size,     shuffle=batch_shuffle)
	val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

	# convert model to dictionary of models
	if isinstance(model, list):
		model = { str(i): model[i] for i in range(len(model)) }
	elif not isinstance(model, dict):
		model = {'0': model}
	#TODO: if several models, then need several loss_fn and accuracy_fn

	device = iter(model.values()).__next__().parameters().__next__().device

	sec = 0
	for epoch in range(num_epochs):
		start = time.time()

		for MODEL in model.values():
			MODEL = MODEL.train()

		loss     = {}
		acc      = {}

		###########################################################
		# single epoch

		epoch_loss     = {}
		epoch_acc      = {}
		epoch_reg_loss = {}
		for batch_ndx, sample in enumerate(dataloader):
			optimizer.zero_grad()
			# loss_sum     = 0
			# reg_loss     = {}
			# reg_loss_sum = 0
			# acc_sum      = 0

			# # compute losses for each model
			# x, y = sample[0].to(device), sample[1].to(device)
			# for key, MODEL in model.items():
			# 	y_pred    = MODEL(x)
			# 	loss[key] = loss_fn( y_pred, y )
			# 	loss_sum  = loss_sum + loss[key]
			# 	if accuracy_fn is not None:
			# 		acc[key] = accuracy_fn( y_pred, y )
			# 		acc_sum  = acc_sum + acc[key]
			# 		epoch_acc[key]  = epoch_acc.get(key,0)  + acc[key].detach()
			# 	epoch_loss[key] = epoch_loss.get(key,0) + loss[key].detach()

			# # compute regularizers
			# if regularizer is not None:
			# 	reg_loss = regularizer()
			# 	# convert regularizers to dictionary
			# 	if isinstance(reg_loss, list):
			# 		reg_loss = { str(i): reg_loss[i] for i in range(len(reg_loss)) }
			# 	elif not isinstance(reg_loss, dict):
			# 		reg_loss = {'0': reg_loss}
			# 	# for reg in reg_loss.values():
			# 	# 	reg_loss_sum = reg_loss_sum + reg
			# for mod in model.values():
			# 	for name, MODULE in mod.named_modules():
			# 		for key, value in getattr(MODULE,'regularizer',{}).items():
			# 			reg_loss[key] = reg_loss.get(key,0) + value
			# for key, reg in reg_loss.items():
			# 	reg_loss_sum = reg_loss_sum + reg
			# 	epoch_reg_loss[key] = epoch_reg_loss.get(key,0) + reg.detach()


			def eval_loss_acc():
				loss_sum     = 0
				reg_loss     = {}
				reg_loss_sum = 0
				acc_sum      = 0
				# compute losses for each model
				# x, y = sample[0].to(device), sample[1].to(device)
				x = [ sample[i].to(device) for i in range(len(sample)-1) ]
				y = sample[len(sample)-1].to(device)
				for key, MODEL in model.items():
					# y_pred    = MODEL(x)
					y_pred    = MODEL(*x)
					if y.shape!=y_pred.shape and epoch==0:
						print("Warning: "+str(y.shape)+" not equal to "+str(y_pred.shape))
					# if not isinstance(loss_fn, torch.nn.CrossEntropyLoss):
					# 	assert y.shape==y_pred.shape, str(y.shape)+" not equal to "+str(y_pred.shape)
					loss[key] = loss_fn( y_pred, y )
					loss_sum  = loss_sum + loss[key]
					if accuracy_fn is not None:
						acc[key] = accuracy_fn( y_pred, y )
						acc_sum  = acc_sum + acc[key]
						epoch_acc[key]  = epoch_acc.get(key,0)  + acc[key].detach()
					epoch_loss[key] = epoch_loss.get(key,0) + loss[key].detach()

				# compute regularizers
				if regularizer is not None:
					reg_loss = regularizer()
					# convert regularizers to dictionary
					if isinstance(reg_loss, list):
						reg_loss = { str(i): reg_loss[i] for i in range(len(reg_loss)) }
					elif not isinstance(reg_loss, dict):
						reg_loss = {'0': reg_loss}
					# for reg in reg_loss.values():
					# 	reg_loss_sum = reg_loss_sum + reg
				for mod in model.values():
					for name, MODULE in mod.named_modules():
						for key, value in getattr(MODULE,'regularizer',{}).items():
							reg_loss[key] = reg_loss.get(key,0) + value
				for key, reg in reg_loss.items():
					reg_loss_sum = reg_loss_sum + reg
					epoch_reg_loss[key] = epoch_reg_loss.get(key,0) + reg.detach()
				return loss_sum, reg_loss_sum, acc_sum, reg_loss

			# propagate gradients
			if isinstance(optimizer,torch.optim.LBFGS):
				def closure():
					loss_sum, reg_loss_sum, _, _ = eval_loss_acc()
					loss_reg_loss = loss_sum + reg_loss_sum
					loss_reg_loss.backward()
					return loss_reg_loss
				optimizer.step(closure)
				loss_sum, reg_loss_sum, acc_sum, reg_loss = eval_loss_acc()
			else:
				loss_sum, reg_loss_sum, acc_sum, reg_loss = eval_loss_acc()
				(loss_sum+reg_loss_sum).backward()
				optimizer.step()
			# torch.nn.utils.clip_grad_value_(model.parameters(), 10)
			# torch.nn.utils.clip_grad_norm_(model.parameters(), 1000, 'inf')

		if checkpoint is not None:
			if (epoch+epoch0)%checkpoint['epochs']==0 and (epoch+epoch0)>0:
				for mod in model.values():
					torch.save(mod.state_dict(), Path(checkpoint['name']+'_chp_'+str(epoch+epoch0)))
				print("checkpoint at epoch "+str(epoch+epoch0))

		with torch.no_grad():
			###########################################################
			# lerning rate schedule

			if scheduler is not None and lr_schedule is None:
				if type(scheduler)==torch.optim.lr_scheduler.ReduceLROnPlateau:
					scheduler.step(loss_sum)
				else:
					scheduler.step()

			if lr_schedule is not None:
				for param_group in optimizer.param_groups:
					param_group['lr'] = lr_schedule[epoch]

			sec += (time.time() - start)

			###########################################################
			# write training stat

			if epoch%stat_freq==0 or epoch==num_epochs-1:
				if writer is not None:

					# hyperparameters
					for idx, param_group in enumerate(optimizer.param_groups):
						writer.add_scalar('hparams/lr_'+str(idx), param_group['lr'], epoch+epoch0)

					# model evaluation stat
					for mkey, MODEL in model.items():
						for name, MODULE in MODEL.named_modules():
							for key, value in getattr(MODULE,'statistics',{}).items():
								writer.add_scalar(key, value, epoch+epoch0)
								# writer.add_scalar(key.replace('/','/'+mkey+'_'+name+'_'), value, epoch+epoch0)
								# writer.add_scalar(mkey+'_'+key, value, epoch+epoch0)

					# loss and accuracy
					for key, LOSS in epoch_loss.items():
						writer.add_scalar('loss/'+key+'_train', LOSS/(batch_ndx+1), epoch+epoch0)
					# if regularizer is not None:
					for key, REG_LOSS in epoch_reg_loss.items():
						writer.add_scalar('loss/regularizer_'+str(key), REG_LOSS/(batch_ndx+1), epoch+epoch0)
					if accuracy_fn is not None:
						for key, ACC in epoch_acc.items():
							writer.add_scalar('accuracy/'+key+'_train', ACC/(batch_ndx+1), epoch+epoch0)

			#######################################################

			# validation error/accuracy
			for mod in model.values():
				mod = mod.eval()

			val_loss = {}
			val_acc  = {}
			if epoch%val_freq==0 or epoch==num_epochs-1:
				val_loss_sum = 0
				val_acc_sum  = 0
				epoch_val_loss = {}
				epoch_val_acc  = {}
				for batch_ndx, sample in enumerate(val_dataloader):
					# val_x, val_y = sample[0].to(device), sample[1].to(device)
					val_x = [ sample[i].to(device) for i in range(len(sample)-1) ]
					val_y = sample[len(sample)-1].to(device)
					for key, MODEL in model.items():
						# val_y_pred   = MODEL(val_x)
						val_y_pred   = MODEL(*val_x)
						val_loss     = loss_fn( val_y_pred, val_y )
						val_loss_sum = val_loss_sum + val_loss
						epoch_val_loss[key] = epoch_val_loss.get(key,0.0) + val_loss
						if accuracy_fn is not None:
							val_acc     = accuracy_fn( val_y_pred, val_y )
							val_acc_sum = val_acc_sum + val_acc
							epoch_val_acc[key] = epoch_val_acc.get(key,0.0) + val_acc
					# break
				if writer is not None:
					for key, VAL_LOSS in epoch_val_loss.items():
						writer.add_scalar('loss/'+key+'_validation', VAL_LOSS/(batch_ndx+1), epoch+epoch0)
						if accuracy_fn is not None:
							writer.add_scalar('accuracy/'+key+'_validation', epoch_val_acc[key]/(batch_ndx+1), epoch+epoch0)
				if accuracy_fn is not None:
					print("Epoch %d: %4.2f sec, loss %5.2e, val_loss %5.2e, acc %4.2f, val_acc %4.2f "%(epoch+epoch0, sec, loss_sum, val_loss_sum/(batch_ndx+1), acc_sum, val_acc_sum/(batch_ndx+1)))
				else:
					print("Epoch %d: %4.2f sec, loss %5.2e, val_loss %5.2e"%(epoch+epoch0, sec, loss_sum, val_loss_sum/(batch_ndx+1)))
				if history:
					loss_history[0,hist_i] = epoch+epoch0
					loss_history[1,hist_i] = loss
					loss_history[2,hist_i] = val_loss
					hist_i += 1
				sec = 0.0
			if write_hist:
				if epoch%hist_freq==0 or epoch==num_epochs-1:
					if writer is not None:
						if isinstance(model, dict):
							for key, value in model.items():
								model0 = value
								break
						else:
							model0 = model
						# for name, weight in model0.named_parameters():
						# 	if weight.grad is not None:
						# 		writer.add_scalar('mean_param_value/'+name, weight.abs().mean(),      epoch+epoch0)
						# 		writer.add_scalar('mean_param_grad/'+name,  weight.grad.abs().mean(), epoch+epoch0)
						# 		writer.add_histogram('parameters/'+name,    weight,      epoch+epoch0, bins='tensorflow')
						# 		writer.add_histogram('gradients/'+name,     weight.grad, epoch+epoch0, bins='tensorflow')
	if history:
		return loss_history




###############################################################################
###############################################################################



def directional_derivative(fun, input, direction, create_graph=True):
	input  = input.detach().requires_grad_(True)
	output = fun(input)

	# v = direction.detach().requires_grad_(True)
	v = torch.ones_like(output).requires_grad_(True)

	# normalize direction
	batch_dim = direction.size(0)
	dir_norm  = torch.clamp( direction.reshape((batch_dim,-1)).norm(dim=1, keepdim=True), min=1.e-16 ) # avoid division by zero
	direction = direction / dir_norm

	grad_x = torch.autograd.grad(
		outputs=output,
		inputs=input,
		grad_outputs=v,
		create_graph=True)[0]

	grad_v, = torch.autograd.grad(
		outputs=grad_x,
		inputs=v,
		grad_outputs=direction.detach(),
		create_graph=create_graph)  # need create_graph to find it's derivative

	return grad_v



class TraceJacobianReg(torch.nn.Module):
	def __init__(self, n=1):
		super().__init__()
		self.n = n

	def forward(self, fun, input, create_graph=True):
		'''
		Compute trace and Frobenius norm of the Jacobian averaged over the batch dimension
		'''
		with torch.enable_grad():
			batch_dim = input.size(0)

			input  = input.detach().requires_grad_(True)
			output = fun(input)

			div = jac = t = q = 0
			if hasattr(fun, 'trace'):
				div = fun.trace(input).sum() * self.n

			# Randomized version based on Hutchinson algorithm for trace estimation
			for _ in range(self.n):
				v = torch.randn_like(output)
				v_jac, = torch.autograd.grad(
					outputs=output,
					inputs=input,
					grad_outputs=v,
					create_graph=create_graph,  # need create_graph to find it's derivative
					only_inputs=True)
				vjacv = v*v_jac
				if not hasattr(fun, 'trace'):
					div = div + vjacv.sum()
				jac = jac + v_jac.pow(2).sum()
				# t = t + vjacv
				# q = q + v*v

			# remove diagonal elements
			# jac = (jac/self.n-(t/q).pow(2).sum())/batch_dim

		return div/self.n/batch_dim, jac/self.n/batch_dim
# class TraceJacobianReg(torch.nn.Module):
# 	def __init__(self, n=1, div=True, jac=False):
# 		super().__init__()
# 		self.n = n
# 		self.is_div = div
# 		self.is_jac = jac

# 	def forward(self, output, input, create_graph=True):
# 		'''
# 		Compute Frobenius norm of the Jacobian averaged over batch dimension
# 		'''
# 		batch_dim = input.size()[0]

# 		div = jac = 0
# 		if not self.is_div and not self.is_jac:
# 			return div, jac

# 		# Randomized version based on Hutchinson algorithm for trace estimation
# 		for _ in range(self.n):
# 			v = torch.randn_like(output)
# 			v_jac, = torch.autograd.grad(
# 				outputs=output,
# 				inputs=input,
# 				grad_outputs=v,
# 				create_graph=create_graph,  # need create_graph to find it's derivative
# 				only_inputs=True)
# 			if self.is_div: div = div + (v*v_jac).sum()
# 			if self.is_jac: jac = jac + v_jac.pow(2).sum()

# 		return div/self.n/batch_dim, jac/self.n/batch_dim



class JacDiagReg(torch.nn.Module):
	def __init__(self, n=1, value=None):
		super().__init__()
		self.n = n
		self.value = value

	def forward(self, fun, input, create_graph=True):
		'''
		Compute stochastic estimate of the Jacobian diagonal:
		Bekas, C., Kokiopoulou, E., Saad, Y.: An estimator for the diagonal of a matrix. Appl. Numer. Math. 57(11), 1214–1229 (2007)
		'''
		batch_dim = input.size(0)

		input  = input.detach().requires_grad_(True)
		output = fun(input)

		t = 0
		q = 0
		for _ in range(self.n):
			v = torch.randn_like(output)
			v_jac, = torch.autograd.grad(
				outputs=output,
				inputs=input,
				grad_outputs=v,
				create_graph=create_graph,  # need create_graph to find it's derivative
				only_inputs=True)
			t = t + v*v_jac
			q = q + v*v

		if self.value is not None:
			return ((t/q)-self.value).pow(2).sum() / batch_dim
		else:
			return t / q
			# return (t / q).reshape((batch_dim,-1))




def jacobian(output, input, create_graph=False):
	jacobian = []

	Id = torch.zeros(*output.shape).to(input.device)
	for i in range(input.numel()):
		Id.data.flatten()[i] = 1.0

		jac_i = torch.autograd.grad(
			outputs=output,
			inputs=input,
			grad_outputs=Id,
			create_graph=create_graph,  # need create_graph to find it's derivative
			only_inputs=True)[0]

		jacobian.append(jac_i)

		Id.data.flatten()[i] = 0.0
	return torch.stack(jacobian, dim=0).reshape(output.shape+input.shape)


# def divergence(output, input, create_graph=False):
# 	jac = jacobian(output, input, create_graph).reshape((output.shape[0],output.numel()//output.shape[0],input.shape[0],input.numel()//input.shape[0]))
# 	return torch.stack([ jac[i,:,i,:].diag().sum() for i in range(output.shape[0]) ], dim=0)



# class divreg(torch.nn.Module):
# 	def __init__(self, n=1):
# 		self.n = n
# 		super(divreg, self).__init__()

# 	def forward(self, output, input, create_graph=False):
# 		reg = 0
# 		for i in range(self.n):
# 			v = torch.randn_like(input)
# 			jac_v, = torch.autograd.grad(
# 				outputs=output,
# 				inputs=input,
# 				grad_outputs=v,
# 				create_graph=create_graph,  # need create_graph to find it's derivative
# 				only_inputs=True)
# 			reg = reg + (v*jac_v).sum()
# 		return reg / self.n / input.size()[0]




# def jacobian_diag(output, input, create_graph=False):
# 	jacobian = []

# 	assert output.numel()==input.numel(), "input and output must have the sane shape"

# 	# out = output.flatten()
# 	# inp = input.flatten()
# 	# print(out[0])
# 	# print(inp[0])

# 	# for i in range(input.numel()):
# 	for i, j in zip(output, input):
# 		print(i,j)
# 		exit()
# 		jac_i = torch.autograd.grad(
# 			outputs=out[i],
# 			inputs=inp[i],
# 			grad_outputs=None,
# 			create_graph=create_graph,  # need create_graph to find it's derivative
# 			only_inputs=True)[0]
# 		jacobian.append(jac_i)

# 	print(jac_i)
# 	exit()

# 	return torch.stack(jacobian, dim=0)