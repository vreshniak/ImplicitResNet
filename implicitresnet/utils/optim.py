import time
import math

from pathlib import Path

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
	def __init__(self, model, loss_fn, dataset, batch_size, optimizer, data_augmentation=None, tol=1.e-12,
		val_dataset=None, val_batch_size=-1, accuracy_fn=None, regularizer=None, scheduler=None, lr_schedule=None,
		checkpoints=None, writer=None, write_hist=False, init_epoch=0, val_freq=1, stat_freq=1, pin_memory=False):
		self.model       = model
		self.loss_fn     = loss_fn
		self.optimizer   = optimizer
		self.data_augm   = data_augmentation
		self.tol         = tol
		self.accuracy_fn = accuracy_fn
		self.regularizer = regularizer
		self.scheduler   = scheduler
		self.lr_schedule = lr_schedule
		self.checkpoints = checkpoints
		self.writer      = writer
		self.write_hist  = write_hist
		self.curr_epoch  = init_epoch
		self.val_freq    = val_freq
		self.stat_freq   = stat_freq

		if checkpoints is not None:
			assert isinstance(checkpoints, dict), "checkpoints must be dictionary"
			assert ('dir' in checkpoints.keys()), "checkpoints must have 'dir' keys"
			if not 'name' in checkpoints.keys(): checkpoints['name'] = 'chkp'
			Path(checkpoints['dir'],"best_loss").mkdir(parents=True, exist_ok=True)
			Path(checkpoints['dir'],"best_val_loss").mkdir(parents=True, exist_ok=True)
			Path(checkpoints['dir'],"best_accuracy").mkdir(parents=True, exist_ok=True)
			Path(checkpoints['dir'],"best_val_accuracy").mkdir(parents=True, exist_ok=True)

			self.best_loss = 1.e6
			self.best_val_loss = 1.e6
			self.best_accuracy = 0.0
			self.best_val_accuracy = 0.0

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
		epoch_loss = 1.0
		for epoch in range(1,epochs+1):
			if epoch_loss<=self.tol: break
			self.curr_epoch += 1

			self.model.train()

			# train on training dataset
			epoch_loss = 0
			epoch_loss_items = {}
			epoch_reg_items  = {}
			epoch_acc_items  = {}
			start = time.time()
			for batch_ndx, sample in enumerate(self.dataloader):
				# model input and target
				x = [ sample[i].to(device) for i in range(len(sample)-1) ]
				y = sample[-1].to(device)

				if self.data_augm is not None:
					aug_x = [ self.data_augm(xi,y)  for xi in x  ]
					x = [ torch.cat((xi, augi)) for xi, augi in zip(x,aug_x)  ]
					y = torch.cat((x[0].size(0)//y.size(0))*[y])

				self.optimizer.zero_grad()

				# model prediction
				y_pred = self.model(*x)
				if y.shape!=y_pred.shape and epoch==1 and batch_ndx==0:
					print("Warning: target shape "+str(y.shape)+" not equal to model output shape "+str(y_pred.shape))

				# propagate gradients
				if isinstance(self.optimizer,torch.optim.LBFGS):
					loss, loss_items, reg_items, acc_items = [0], [0], [0], [0]
					def closure():
						self.optimizer.zero_grad()
						y_pred = self.model(*x)
						loss[0], reg, loss_items[0], reg_items[0], acc_items[0] = self.evaluate_loss_regularizers_accuracy(y_pred, y)
						loss_reg = loss[0] + reg
						loss_reg.backward()
						return loss_reg
					self.optimizer.step(closure)
					loss, loss_items, reg_items, acc_items = loss[0], loss_items[0], reg_items[0], acc_items[0]
				else:
					loss, reg, loss_items, reg_items, acc_items = self.evaluate_loss_regularizers_accuracy(y_pred, y)
					(loss + reg).backward()
					self.optimizer.step()

				# collect loss, regularizer and accuracy components
				epoch_loss = epoch_loss + loss.detach()
				for key, val in loss_items.items(): epoch_loss_items[key] = epoch_loss_items.get(key,0) + val
				for key, val in reg_items.items():  epoch_reg_items[key]  = epoch_reg_items.get(key,0)  + val
				for key, val in acc_items.items():  epoch_acc_items[key]  = epoch_acc_items.get(key,0)  + val

			epoch_loss = epoch_loss / (batch_ndx+1)
			for key in epoch_loss_items.keys(): epoch_loss_items[key] = epoch_loss_items[key] / (batch_ndx+1)
			for key in epoch_reg_items.keys():  epoch_reg_items[key]  = epoch_reg_items[key]  / (batch_ndx+1)
			for key in epoch_acc_items.keys():  epoch_acc_items[key]  = epoch_acc_items[key]  / (batch_ndx+1)
			sec += (time.time() - start)


			with torch.no_grad():
				###########################################################
				# lerning rate schedule
				if self.scheduler is not None and self.lr_schedule is None:
					if type(self.scheduler)==torch.optim.lr_scheduler.ReduceLROnPlateau:
						self.scheduler.step(epoch_loss)
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
					message = "Epoch %4d: %5.2f sec, loss"%(self.curr_epoch, sec)
					for val in epoch_loss_items.values():
						message = message + " %5.2e"%(val)
					if self.accuracy_fn is not None:
						message = message + ", acc"%(val)
						for val in epoch_acc_items.values():
							message = message + " %6.2f"%(val*100)
					if self.val_dataloader is not None:
						message = message + "  ||  %5.2f sec, val_loss"%(val_sec)
						for val in epoch_val_loss_items.values():
							message = message + " %5.2e"%(val)
						if self.accuracy_fn is not None:
							message = message + ", val_acc "%(val)
							for val in epoch_val_acc_items.values():
								message = message + "%6.2f "%(val*100)
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


			if self.checkpoints is not None:
				if 'each_nth' in self.checkpoints.keys() and self.curr_epoch%self.checkpoints['each_nth']==0:
					Path(self.checkpoints['dir'],"epoch_%d"%(self.curr_epoch)).mkdir(parents=True, exist_ok=True)
					torch.save(self.model.state_dict(), Path(self.checkpoints['dir'],"epoch_%d"%(self.curr_epoch),self.checkpoints['name']))
					print("checkpoint at epoch %d"%(self.curr_epoch))
				for value in epoch_loss_items.values():
					if value<=self.best_loss:
						self.best_loss = value
						torch.save(self.model.state_dict(), Path(self.checkpoints['dir'],"best_loss",self.checkpoints['name']))
						print("checkpoint best loss")
				for value in epoch_val_loss_items.values():
					if value<=self.best_val_loss:
						self.best_val_loss = value
						torch.save(self.model.state_dict(), Path(self.checkpoints['dir'],"best_val_loss",self.checkpoints['name']))
						print("checkpoint best validation loss")
				if self.accuracy_fn is not None:
					# for value in epoch_acc_items.values():
					# 	if value>=self.best_accuracy:
					# 		self.best_accuracy = value
					# 		torch.save(self.model.state_dict(), Path(self.checkpoints['dir'],"best_accuracy",self.checkpoints['name']))
					# 		print("checkpoint best accuracy")
					for value in epoch_val_acc_items.values():
						if value>=self.best_val_accuracy:
							self.best_val_accuracy = value
							torch.save(self.model.state_dict(), Path(self.checkpoints['dir'],"best_val_accuracy",self.checkpoints['name']))
							print("checkpoint best validation accuracy")



###############################################################################
###############################################################################



class EvenReductionLR(torch.optim.lr_scheduler.StepLR):
	def __init__(self, optimizer, lr_reduction, gamma, epochs, last_epoch=-1):
		super().__init__(optimizer, step_size=max(int(epochs*math.log(gamma)/math.log(lr_reduction)),1), gamma=gamma, last_epoch=last_epoch)