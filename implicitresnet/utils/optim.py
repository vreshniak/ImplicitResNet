import time
import math

from pathlib import Path

from tqdm import tqdm, trange

import torch
from torch.utils.tensorboard import SummaryWriter



###############################################################################
###############################################################################


def as_sum_and_dict(value):
	'''
	* convert value to dictionary
	* return dict and its sum
	'''
	val   = torch.tensor(0)
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
	def __init__(self, model, loss_fn, dataset, batch_size, optimizer, data_augmentation=None, tol=None, tol_target=None,
		val_dataset=None, val_batch_size=None, accuracy_fn=None, regularizer=None, scheduler=None, lr_schedule=None,
		checkpoints=None, writer=None, init_epoch=0, min_epochs=0, val_freq=1, stat_freq=1, hist_freq=0, pin_memory=False,
		verbose=False, progress_bar=True, eval_init_epoch_loss=False):
		self.model       = model
		self.loss_fn     = loss_fn
		self.optimizer   = optimizer
		self.data_augm   = data_augmentation
		self.tol         = tol
		self.tol_target  = tol_target
		self.accuracy_fn = accuracy_fn
		self.regularizer = regularizer
		self.scheduler   = scheduler
		self.lr_schedule = lr_schedule
		self.checkpoints = checkpoints
		self.writer      = writer
		self.hist_freq   = hist_freq
		self.curr_epoch  = init_epoch
		self.min_epochs  = min_epochs
		self.val_freq    = val_freq
		self.stat_freq   = stat_freq

		self.progress_bar = progress_bar
		self.verbose      = verbose
		self.eval_init_epoch_loss = eval_init_epoch_loss

		self.loss_history = []
		self.acc_history = []
		self.val_loss_history = []
		self.val_acc_history = []


		if checkpoints is not None:
			assert isinstance(checkpoints, dict), "checkpoints must be dictionary"
			assert ('dir' in checkpoints.keys()), "checkpoints must have 'dir' key"
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
			if val_batch_size is None:
				val_batch_size = batch_size
			elif val_batch_size<0:
				val_batch_size = len(val_dataset)
			self.val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, pin_memory=pin_memory)
		else:
			self.val_dataloader = None

		self.immutable_attributes = ['model', 'loss_fn', 'dataset', 'val_dataset', 'batch_size', 'accuracy_fn']


	def evaluate_loss_regularizers_accuracy(self, y_pred, y):
		# evaluate loss
		loss, loss_items = as_sum_and_dict(self.loss_fn(y_pred, y) if self.loss_fn is not None else None)

		# evaluate regularizers
		reg, reg_items = as_sum_and_dict(self.regularizer() if self.regularizer is not None else None)

		# evaluate regularizers which are model properties
		for name, module in self.model.named_modules():
			if hasattr(module,'regularizer'):
				for key, value in module.regularizer.items():
					reg = reg + value
					reg_items[key] = reg_items.get(key,0) + value.detach()
				# regularizers should be evaluated only once and then freed
				module.regularizer = {}
			# for key, value in getattr(module,'regularizer',{}).items():
			# 	reg = reg + value
			# 	reg_items[key] = reg_items.get(key,0) + value.detach()

		# evaluate accuracy
		_, acc_items = as_sum_and_dict(self.accuracy_fn(y_pred, y) if self.accuracy_fn is not None else None)

		return loss, reg, loss_items, reg_items, acc_items


	def log_histograms(self, epoch, epochs):
		if self.hist_freq>0 and (epoch%self.hist_freq==0 or epoch==epochs):
			for name, weight in self.model.named_parameters():
				self.writer.add_histogram('parameters/'+name,    weight,      self.curr_epoch, bins='tensorflow')
				if weight.grad is not None:
					self.writer.add_histogram('gradients/'+name, weight.grad, self.curr_epoch, bins='tensorflow')
				# writer.add_scalar('mean_param_value/'+name, weight.abs().mean(),      self.curr_epoch)
				# writer.add_scalar('mean_param_grad/'+name,  weight.grad.abs().mean(), self.curr_epoch)


	def get_history(self):
		return self.loss_history, self.acc_history, self.val_loss_history, self.val_acc_history


	@torch.no_grad()
	def loss_over_dataset(self, dataloader):
		# model device
		device = self.model.parameters().__next__().device

		dataset_loss = 0
		dataset_loss_items, dataset_reg_items, dataset_acc_items = {}, {}, {}
		for batch_ndx, sample in enumerate(dataloader):
			# model input and target
			x = [ sample[i].to(device) for i in range(len(sample)-1) ]
			y = sample[-1].to(device)
			y_pred = self.model(*x)
			if self.verbose and batch_ndx==0 and y.shape!=y_pred.shape:
				print(f"Warning: target shape {y.shape} not equal to model output shape {y_pred.shape} for {self.loss_fn} loss")
			loss, reg, loss_items, reg_items, acc_items = self.evaluate_loss_regularizers_accuracy(y_pred, y)

			# accumulate values
			batch_weight = y.size(0) / len(dataloader.dataset)
			dataset_loss = dataset_loss + loss.detach() * batch_weight
			for key, val in loss_items.items(): dataset_loss_items[key] = dataset_loss_items.get(key,0) + val * batch_weight
			for key, val in reg_items.items():  dataset_reg_items[key]  = dataset_reg_items.get(key,0)  + val * batch_weight
			for key, val in acc_items.items():  dataset_acc_items[key]  = dataset_acc_items.get(key,0)  + val * batch_weight
		# # average values
		# dataset_loss = dataset_loss / (batch_ndx+1)
		# for key in dataset_loss_items.keys():
		# 	dataset_loss_items[key] = dataset_loss_items[key] / (batch_ndx+1)
		# return dataset_loss if self.tol_target is None else dataset_loss_items[self.tol_target]
		return dataset_loss, dataset_loss_items, dataset_reg_items, dataset_acc_items


	def __call__(self, epochs, **kwargs):
		# override default parameters
		for key, val in kwargs.items():
			assert key not in self.immutable_attributes, "train.%s parameter is immutable"%(key)
			setattr(self, key, val)

		# model device
		device = self.model.parameters().__next__().device

		###########################################################
		# initial logs
		if self.eval_init_epoch_loss:
			self.model.eval()
			epoch_loss, epoch_loss_items, epoch_reg_items, epoch_acc_items = self.loss_over_dataset(self.dataloader)

			self.log_histograms(self.curr_epoch, self.curr_epoch+epochs)
			if self.writer is not None:
				# model evaluation stat
				for name, module in self.model.named_modules():
					for key, value in getattr(module,'statistics',{}).items():
						self.writer.add_scalar(key, value, self.curr_epoch)

				# training loss, regularizers and accuracy
				self.writer.add_scalar(f'loss/train', epoch_loss, self.curr_epoch)
				if len(epoch_loss_items)>1:
					for key, value in epoch_loss_items.items(): self.writer.add_scalar(f'loss/{key}_train', value, self.curr_epoch)
				for key, value in epoch_reg_items.items():  self.writer.add_scalar(f'regularizers/{key}',   value, self.curr_epoch)
				for key, value in epoch_acc_items.items():  self.writer.add_scalar(f'accuracy/{key}_train', value, self.curr_epoch) if len(epoch_acc_items)>1 else self.writer.add_scalar(f'accuracy/train', value, self.curr_epoch)

				# validation loss and accuracy
				if self.val_dataloader is not None:
					epoch_val_loss, epoch_val_loss_items, _, epoch_val_acc_items = self.loss_over_dataset(self.val_dataloader)
					self.writer.add_scalar(f'loss/valid', epoch_val_loss, self.curr_epoch)
					if len(epoch_val_loss_items)>1:
						for key, value in epoch_val_loss_items.items(): self.writer.add_scalar('loss/'+key+'_valid', value, self.curr_epoch)
					for key, value in epoch_val_acc_items.items():  self.writer.add_scalar('accuracy/'+key+'_valid', value, self.curr_epoch) if len(epoch_val_acc_items)>1 else self.writer.add_scalar(f'accuracy/valid', value, self.curr_epoch)
		else:
			epoch_loss = None
		###########################################################

		# initial tolerance
		tol_loss  = epoch_loss if self.tol_target is None else epoch_loss_items[self.tol_target]
		converged = False

		sec = 0
		message = ""
		#for epoch in tqdm(range(1,epochs+1)):
		#for epoch in trange(1,epochs+1, disable=(not self.progress_bar), postfix={'s':message}):
		pbar = tqdm(range(1,epochs+1), disable=(not self.progress_bar), bar_format="{l_bar}{bar} [{elapsed}<{remaining}, {rate_fmt}]{postfix}")
		for epoch in pbar:
			self.curr_epoch += 1

			# termination condition
			if self.tol is not None and tol_loss<=self.tol and self.curr_epoch>self.min_epochs:
				converged = True
				break

			###########################################################
			# training
			self.model.train()

			# train on training dataset
			epoch_loss = 0
			epoch_loss_items, epoch_reg_items, epoch_acc_items = {}, {}, {}
			start = time.time()
			for batch_ndx, sample in enumerate(self.dataloader):
				# model input and target
				x = [ sample[i].to(device) for i in range(len(sample)-1) ]
				y = sample[-1].to(device)

				if self.data_augm is not None:
					aug_x = [ self.data_augm(xi,y) for xi in x  ]
					x = [ torch.cat((xi, augi)) for xi, augi in zip(x,aug_x)  ]
					y = torch.cat((x[0].size(0)//y.size(0))*[y])

				# propagate gradients, closure is always used for consistency
				# lists are used to modify variables from inside closure
				loss, loss_items, reg_items, acc_items = [0], [0], [0], [0]
				def closure():
					self.optimizer.zero_grad()
					y_pred = self.model(*x)
					loss[0], reg, loss_items[0], reg_items[0], acc_items[0] = self.evaluate_loss_regularizers_accuracy(y_pred, y)
					loss_reg = loss[0] + reg
					loss_reg.backward()
					return loss_reg
				# clip_params = [ param for name,param in self.model.named_parameters() if ('cnt_var' not in name or 'lip_var' not in name) ]
				# torch.nn.utils.clip_grad_value_(clip_params,5.0)
				# torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0,'inf')
				self.optimizer.step(closure)
				loss, loss_items, reg_items, acc_items = loss[0], loss_items[0], reg_items[0], acc_items[0]

				# collect loss, regularizer and accuracy components
				batch_weight = y.size(0) / len(self.dataloader.dataset)
				epoch_loss = epoch_loss + loss.detach() * batch_weight
				for key, val in loss_items.items(): epoch_loss_items[key] = epoch_loss_items.get(key,0) + val * batch_weight
				for key, val in reg_items.items():  epoch_reg_items[key]  = epoch_reg_items.get(key,0)  + val * batch_weight
				for key, val in acc_items.items():  epoch_acc_items[key]  = epoch_acc_items.get(key,0)  + val * batch_weight
			sec += (time.time() - start)
			# average (over the batches in the epoch) loss, regularizer and accuracy components
			# epoch_loss = epoch_loss / (batch_ndx+1)
			# for key in epoch_loss_items.keys(): epoch_loss_items[key] = epoch_loss_items[key] / (batch_ndx+1)
			# for key in epoch_reg_items.keys():  epoch_reg_items[key]  = epoch_reg_items[key]  / (batch_ndx+1)
			# for key in epoch_acc_items.keys():  epoch_acc_items[key]  = epoch_acc_items[key]  / (batch_ndx+1)

			# convergence criterion
			tol_loss = epoch_loss if self.tol_target is None else epoch_loss_items[self.tol_target]
			# if loss/old_loss>1.2:
			# 	print(epoch)
			# old_loss = loss.detach()

			self.loss_history.append(epoch_loss.item())
			if self.accuracy_fn is not None:
				self.acc_history.append(next(iter(epoch_acc_items.values())).item())


			with torch.no_grad():
				#######################################################
				# validation error/accuracy
				self.model.eval()
				if epoch%self.val_freq==0 or epoch==epochs:
					# evaluate on validation dataset
					if self.val_dataloader is not None:
						epoch_val_loss = 0
						epoch_val_loss_items, epoch_val_acc_items = {}, {}
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
							batch_weight = y.size(0) / len(self.val_dataloader.dataset)
							epoch_val_loss = epoch_val_loss + val_loss.detach() * batch_weight
							for key, val in val_loss_items.items(): epoch_val_loss_items[key] = epoch_val_loss_items.get(key,0) + val * batch_weight
							for key, val in val_acc_items.items():  epoch_val_acc_items[key]  = epoch_val_acc_items.get(key,0)  + val * batch_weight
						val_sec = time.time() - start
						# epoch_val_loss = epoch_val_loss / (batch_ndx+1)
						# for key in val_loss_items.keys(): epoch_val_loss_items[key] = epoch_val_loss_items[key] / (batch_ndx+1)
						# for key in val_acc_items.keys():  epoch_val_acc_items[key]  = epoch_val_acc_items[key]  / (batch_ndx+1)

						self.val_loss_history.append(epoch_val_loss.item())
						if self.accuracy_fn is not None:
							self.val_acc_history.append(next(iter(epoch_val_acc_items.values())).item())

				###########################################################
				# log stat, loss, regularizers, and accuracy
				if self.writer is not None:
					if epoch%self.stat_freq==0 or epoch==epochs:
						# hyperparameters
						for idx, param_group in enumerate(self.optimizer.param_groups):
							self.writer.add_scalar('hparams/lr_'+str(idx), param_group['lr'], self.curr_epoch)

						# model evaluation stat
						for name, module in self.model.named_modules():
							for key, value in getattr(module,'statistics',{}).items():
								self.writer.add_scalar(key, value, self.curr_epoch)

						# training loss, regularizers and accuracy
						self.writer.add_scalar(f'loss/train', epoch_loss, self.curr_epoch)
						if len(epoch_loss_items)>1:
							for key, value in epoch_loss_items.items(): self.writer.add_scalar(f'loss/{key}_train', value, self.curr_epoch)
						for key, value in epoch_reg_items.items():  self.writer.add_scalar(f'regularizers/{key}',   value, self.curr_epoch)
						for key, value in epoch_acc_items.items():  self.writer.add_scalar(f'accuracy/{key}_train', value, self.curr_epoch) if len(epoch_acc_items)>1 else self.writer.add_scalar(f'accuracy/train', value, self.curr_epoch)

					# validation loss and accuracy
					if self.val_dataloader is not None:
						if epoch%self.val_freq==0 or epoch==epochs:
							self.writer.add_scalar(f'loss/valid', epoch_val_loss, self.curr_epoch)
							if len(epoch_val_loss_items)>1:
								for key, value in epoch_val_loss_items.items(): self.writer.add_scalar('loss/'+key+'_valid', value, self.curr_epoch)
							for key, value in epoch_val_acc_items.items():  self.writer.add_scalar('accuracy/'+key+'_valid', value, self.curr_epoch) if len(epoch_val_acc_items)>1 else self.writer.add_scalar(f'accuracy/valid', value, self.curr_epoch)

				###########################################################
				# print training progress
				if epoch%self.val_freq==0:
					message = "Epoch %4d: %5.2f sec, loss"%(self.curr_epoch, sec)
					# for val in epoch_loss_items.values():
					# 	message = message + " %5.2e"%(val)
					message = message + " %5.2e"%(epoch_loss)
					if self.accuracy_fn is not None:
						message = message + ", acc"#%(val)
						for val in epoch_acc_items.values():
							message = message + " %6.2f"%(val*100)
					if self.val_dataloader is not None:
						message = message + "  ||  %5.2f sec, val_loss"%(val_sec)
						# for val in epoch_val_loss_items.values():
						# 	message = message + " %5.2e"%(val)
						message = message + " %5.2e"%(epoch_val_loss)
						if self.accuracy_fn is not None:
							message = message + ", val_acc "#%(val)
							for val in epoch_val_acc_items.values():
								message = message + "%6.2f "%(val*100)
					sec = 0
					if self.verbose and not self.progress_bar:
						print(message)
					# pbar.set_postfix(_='\n'+message)
					# pbar.set_postfix({0:'\n'+message})
					pbar.postfix = message

				#if epoch>1740:
				self.log_histograms(self.curr_epoch, self.curr_epoch+epochs)

				###########################################################
				# lerning rate schedule
				if self.scheduler is not None and self.lr_schedule is None:
					if isinstance(self.scheduler,torch.optim.lr_scheduler.ReduceLROnPlateau):
						self.scheduler.step(epoch_val_loss)
					else:
						self.scheduler.step()
				if self.lr_schedule is not None:
					for param_group in self.optimizer.param_groups:
						param_group['lr'] = self.lr_schedule[self.curr_epoch]

				###########################################################
				# checkpoint model
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

		return converged

###############################################################################
###############################################################################



class EvenReductionLR(torch.optim.lr_scheduler.StepLR):
	def __init__(self, optimizer, lr_reduction, gamma, epochs, last_epoch=-1):
		super().__init__(optimizer, step_size=max(int(epochs*math.log(gamma)/math.log(lr_reduction)),1), gamma=gamma, last_epoch=last_epoch)