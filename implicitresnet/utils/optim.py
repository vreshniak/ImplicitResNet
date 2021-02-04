import time
import math

import torch
from torch.utils.tensorboard import SummaryWriter



###############################################################################
###############################################################################


def as_sum_and_dict(value):
	val   = 0
	items = {}
	if value is not None:
		if isinstance(value, list):
			for i, v in enumerate(value):
				val = val + v
				items[str(i)] = v.detach()
		elif isinstance(value, dict):
			for key, v in value.items():
				val = val + v
				items[key] = v.detach()
		else:
			val   = value
			items = {'0': value.detach()}
	return val, items


###############################################################################
###############################################################################


class TrainingLoop:
	def __init__(self, model, loss_fn, dataset, batch_size, optimizer,
		val_dataset=None, val_batch_size=-1, accuracy_fn=None, regularizer=None, scheduler=None, lr_schedule=None,
		checkpoint=None, writer=None, write_hist=False, init_epoch=0, val_freq=1, stat_freq=1, pin_memory=False):
		self.model       = model
		self.loss_fn     = loss_fn
		self.optimizer   = optimizer
		self.accuracy_fn = accuracy_fn
		self.regularizer = regularizer
		self.scheduler   = scheduler
		self.lr_schedule = lr_schedule
		self.checkpoint  = checkpoint
		self.writer      = writer
		self.write_hist  = write_hist
		self.curr_epoch  = init_epoch
		self.val_freq    = val_freq
		self.stat_freq   = stat_freq

		if checkpoint is not None:
			assert isinstance(checkpoint, dict), "checkpoint must be dictionary"
			assert ('each_nth' in checkpoint.keys()) and ('dir' in checkpoint.keys()), "checkpoint must have 'each_nth' and 'dir' keys"

		# prepare data
		batch_shuffle = True
		if batch_size is None or batch_size<0:
			batch_size    = len(dataset)
			batch_shuffle = False
		self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=batch_shuffle, pin_memory=pin_memory)
		if val_dataset is not None:
			if val_batch_size is None or val_batch_size<0:
				val_batch_size = len(val_dataset)
			self.val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, pin_memory=pin_memory)
		else:
			self.val_dataloader = None

		self.immutable_attributes = ['model', 'loss_fn', 'dataset', 'val_dataset', 'batch_size', 'accuracy_fn']


	def evaluate_loss_regularizers_accuracy(self, y_pred, y):
		# evaluate loss
		loss, loss_items = as_sum_and_dict(self.loss_fn(y_pred, y))

		# evaluate regularizers
		reg, reg_items = as_sum_and_dict(self.regularizer() if self.regularizer is not None else None)

		# evaluate regularizers which are model properties
		for name, module in self.model.named_modules():
			for key, value in getattr(module,'regularizer',{}).items():
				reg = reg + value
				reg_items[key] = reg_items.get(key,0) + value.detach()

		# evaluate accuracy
		_, acc_items = as_sum_and_dict(self.accuracy_fn(y_pred, y) if self.accuracy_fn is not None else None)

		return loss, reg, loss_items, reg_items, acc_items


	def __call__(self, epochs, **kwargs):
		# override default parameters
		for key, val in kwargs.items():
			assert key not in self.immutable_attributes, "train.%s parameter is immutable"%(key)
			setattr(self, key, val)

		# model device
		device = self.model.parameters().__next__().device

		sec = 0
		for epoch in range(1,epochs+1):
			self.curr_epoch += 1

			self.model.train()

			# train on training dataset
			epoch_loss_items = {}
			epoch_reg_items  = {}
			epoch_acc_items  = {}
			start = time.time()
			for batch_ndx, sample in enumerate(self.dataloader):
				self.optimizer.zero_grad()

				# model input and target
				x = [ sample[i].to(device) for i in range(len(sample)-1) ]
				y = sample[-1].to(device)

				# model prediction
				y_pred = self.model(*x)
				if y.shape!=y_pred.shape and epoch==1:
					print("Warning: target shape "+str(y.shape)+" not equal to model output shape "+str(y_pred.shape))

				# propagate gradients
				if isinstance(self.optimizer,torch.optim.LBFGS):
					loss_items = []
					reg_items  = []
					acc_items  = []
					def closure():
						loss, reg, loss_items[0], reg_items[0], acc_items[0] = self.evaluate_loss_regularizers_accuracy(y_pred, y)
						loss_reg = loss + reg
						loss_reg.backward()
						return loss_reg
					self.optimizer.step(closure)
					loss_items = loss_items[0]
					reg_items  = reg_items[0]
					acc_items  = acc_items[0]
				else:
					loss, reg, loss_items, reg_items, acc_items = self.evaluate_loss_regularizers_accuracy(y_pred, y)
					(loss + reg).backward()
					self.optimizer.step()

				# collect loss, regularizer and accuracy components
				for key, val in loss_items.items(): epoch_loss_items[key] = epoch_loss_items.get(key,0) + val
				for key, val in reg_items.items():  epoch_reg_items[key]  = epoch_reg_items.get(key,0)  + val
				for key, val in acc_items.items():  epoch_acc_items[key]  = epoch_acc_items.get(key,0)  + val

			for key in epoch_loss_items.keys(): epoch_loss_items[key] = epoch_loss_items[key] / (batch_ndx+1)
			for key in epoch_reg_items.keys():  epoch_reg_items[key]  = epoch_reg_items[key]  / (batch_ndx+1)
			for key in epoch_acc_items.keys():  epoch_acc_items[key]  = epoch_acc_items[key]  / (batch_ndx+1)
			sec += (time.time() - start)


			if self.checkpoint is not None:
				if self.curr_epoch%self.checkpoint['each_nth']==0:
					torch.save(self.model.state_dict(), Path(self.checkpoint['dir'],"_chp_%d"%(self.curr_epoch)))
					print("checkpoint at epoch %d"%(self.curr_epoch))


			with torch.no_grad():
				###########################################################
				# lerning rate schedule
				if self.scheduler is not None and self.lr_schedule is None:
					if type(self.scheduler)==torch.optim.lr_scheduler.ReduceLROnPlateau:
						self.scheduler.step(loss)
					else:
						self.scheduler.step()
				if self.lr_schedule is not None:
					for param_group in self.optimizer.param_groups:
						param_group['lr'] = self.lr_schedule[self.curr_epoch]

				###########################################################
				# write training stat
				if epoch%self.stat_freq==0 or epoch==epochs:
					if self.writer is not None:
						# hyperparameters
						for idx, param_group in enumerate(self.optimizer.param_groups):
							self.writer.add_scalar('hparams/lr_'+str(idx), param_group['lr'], self.curr_epoch)

						# model evaluation stat
						for name, module in self.model.named_modules():
							for key, value in getattr(module,'statistics',{}).items():
								self.writer.add_scalar(key, value, self.curr_epoch)

						# loss components, regularizers and accuracy components
						for key, value in epoch_loss_items.items(): self.writer.add_scalar('loss/'+key+'_train',     value, self.curr_epoch)
						for key, value in epoch_reg_items.items():  self.writer.add_scalar('loss/reg_'+key,          value, self.curr_epoch)
						for key, value in epoch_acc_items.items():  self.writer.add_scalar('accuracy/'+key+'_train', value, self.curr_epoch)

				#######################################################
				# validation error/accuracy
				self.model.eval()

				if epoch%self.val_freq==0 or epoch==epochs:
					# evaluate on validation dataset
					if self.val_dataloader is not None:
						epoch_val_loss_items = {}
						epoch_val_acc_items  = {}
						start = time.time()
						for batch_ndx, sample in enumerate(self.val_dataloader):
							# model input and target
							x = [ sample[i].to(device) for i in range(len(sample)-1) ]
							y = sample[-1].to(device)

							# model prediction
							y_pred = self.model(*x)

							# evaluate loss and accuracy
							val_loss, _, val_loss_items, _, val_acc_items = self.evaluate_loss_regularizers_accuracy(y_pred, y)

							# collect loss and accuracy components
							for key, val in val_loss_items.items(): epoch_val_loss_items[key] = epoch_val_loss_items.get(key,0) + val
							for key, val in val_acc_items.items():  epoch_val_acc_items[key]  = epoch_val_acc_items.get(key,0)  + val
						for key in val_loss_items.keys(): epoch_val_loss_items[key] = epoch_val_loss_items[key] / (batch_ndx+1)
						for key in val_acc_items.keys():  epoch_val_acc_items[key]  = epoch_val_acc_items[key]  / (batch_ndx+1)
						val_sec = time.time() - start

						# write validation loss and accuracy components
						if self.writer is not None:
							for key, value in epoch_val_loss_items.items(): self.writer.add_scalar('loss/'+key+'_validation',     value, self.curr_epoch)
							for key, value in epoch_val_acc_items.items():  self.writer.add_scalar('accuracy/'+key+'_validation', value, self.curr_epoch)

					# print training progress
					message = "Epoch %4d: %4.2f sec, loss"%(self.curr_epoch, sec)
					for val in epoch_loss_items.values():
						message = message + " %5.2e"%(val)
					if self.accuracy_fn is not None:
						message = message + ", acc"%(val)
						for val in epoch_acc_items.values():
							message = message + " %4.2f"%(val)
					if self.val_dataloader is not None:
						message = message + "  ||  %4.2f sec, val_loss"%(val_sec)
						for val in epoch_val_loss_items.values():
							message = message + " %5.2e"%(val)
						if self.accuracy_fn is not None:
							message = message + ", val_acc "%(val)
							for val in epoch_val_acc_items.values():
								message = message + "%4.2f "%(val)
					print(message)
					sec = 0

				if self.write_hist:
					if epoch%self.hist_freq==0 or epoch==epochs:
						for name, weight in self.model.named_parameters():
							if weight.grad is not None:
								writer.add_histogram('parameters/'+name,    weight,      self.curr_epoch, bins='tensorflow')
								writer.add_histogram('gradients/'+name,     weight.grad, self.curr_epoch, bins='tensorflow')
								# writer.add_scalar('mean_param_value/'+name, weight.abs().mean(),      self.curr_epoch)
								# writer.add_scalar('mean_param_grad/'+name,  weight.grad.abs().mean(), self.curr_epoch)



###############################################################################
###############################################################################



class EvenReductionLR(torch.optim.lr_scheduler.StepLR):
	def __init__(self, optimizer, lr_reduction, gamma, epochs, last_epoch=-1):
		super().__init__(optimizer, step_size=max(int(epochs*math.log(gamma)/math.log(lr_reduction)),1), gamma=gamma, last_epoch=last_epoch)