import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from datasets import get_dataset
from dotmap import DotMap
import time

from models import *

class Trainer:
	def __init__(self, config_file_data):
		self.cfg = config_file_data

	def create_model(self):
		if self.cfg.model.type == 'FFNModel':
			model = FFNModel(self.cfg.model.input_dim, self.cfg.model.hidden_dim_1, \
				self.cfg.model.hidden_dim_2, self.cfg.model.output_dim)
		elif self.cfg.model.type == 'FFNModel_Softmax':
			model = FFNModel_Softmax(self.cfg.model.input_dim, self.cfg.model.hidden_dim_1, \
				self.cfg.model.hidden_dim_2, self.cfg.model.output_dim)
		elif self.cfg.model.type == 'FFNModel_Sigmoid':
			model = FFNModel_Sigmoid(self.cfg.model.input_dim, self.cfg.model.hidden_dim_1, \
				self.cfg.model.hidden_dim_2, self.cfg.model.output_dim)
		else:
			print('Unknown model in Trainer class')
			exit(1)
		return model

	def loss_function(self):
		if self.cfg.loss.type == 'BCELoss':
			criterion = nn.BCELoss()
			criterion_nored = nn.BCELoss(reduction = 'none')
		elif self.cfg.loss.type == 'CrossEntropyLoss':
			criterion = nn.CrossEntropyLoss()
			criterion_nored = nn.CrossEntropyLoss(reduction = 'none')
		elif self.cfg.loss.type == 'MeanSquaredLoss':
			criterion = nn.MSELoss()
			criterion_nored = nn.MSELoss(reduction = 'none')
		else:
			print('Unknown loss in Trainer class')
			exit(1)
		return criterion, criterion_nored

	def optimizer_with_scheduler(self, model):
		if self.cfg.optimizer.type == 'sgd':
			optimizer = optim.SGD(self.model.parameters(), lr=self.cfg.optimizer.lr,
								  momentum=self.cfg.optimizer.momentum,
								  weight_decay=self.cfg.optimizer.weight_decay,
								  nesterov=self.cfg.optimizer.nesterov)
		elif self.cfg.optimizer.type == 'adam':
			optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.optimizer.lr)
		elif self.cfg.optimizer.type == 'rmsprop':
			optimizer = optim.RMSprop(self.model.parameters(), lr=self.cfg.optimizer.lr)

		if self.cfg.scheduler.type == 'cosine_annealing':
			scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.scheduler.T_max)
		else:
			scheduler = None
		return optimizer, scheduler

	def parse_words(self, data, outputs):
		words_and_scores = []
		for i in range(len(data[0])):
			words_and_scores.append([data[0][i], data[1][i], outputs[i]])
		return words_and_scores

	def evaluate(self, dataloader_type, sigmoid_threshold = 0.5, test_index = 0, have_targets = True):
		if dataloader_type == 'trn':
			dataloader = self.trn_dataloader
		elif dataloader_type == 'val':
			dataloader = self.val_dataloader
		elif dataloader_type == 'test':
			dataloader = self.test_dataloaders[test_index]
		else:
			print('Unknown dataloader type in Trainer')

		total_loss = 0
		total_data = 0
		correctly_classified = 0
		words_and_scores = []
		with torch.no_grad():
			for _, (data, inputs, targets) in enumerate(dataloader):
				inputs, targets = inputs.to(self.cfg.train_args.device), \
								  targets.to(self.cfg.train_args.device, non_blocking=True)
				outputs = self.model(inputs.float())
				
				if sigmoid_threshold >= 0:
					words_and_scores.extend(self.parse_words(data, np.array(outputs.view(-1))))
				else:
					words_and_scores.extend(self.parse_words(data, np.array(((outputs.max(1))[1]).view(-1))))
				
				if have_targets:
					if self.cfg.loss.type == 'BCELoss':
						loss = self.criterion(outputs.view(-1), targets.float())
					elif self.cfg.loss.type == 'CrossEntropyLoss':
						loss = self.criterion(outputs, targets)
					total_loss += loss.item()

				total_data += inputs.size(0)

				if have_targets:
					if sigmoid_threshold >= 0:
						predicted = ((outputs > sigmoid_threshold).long()).view(-1)
					else:
						_, predicted = outputs.max(1)
					correctly_classified += predicted.eq(targets).sum().item()
		accuracy = 0
		if total_data != 0:
			accuracy = correctly_classified/total_data
		return total_loss, correctly_classified, total_data, accuracy, words_and_scores

	def train(self):
		trn_set, val_set, test_sets, num_cls = get_dataset(self.cfg.dataset)

		self.trn_dataloader = DataLoader(trn_set, batch_size = self.cfg.dataloader.trn_bs, shuffle = self.cfg.dataloder.shuffle)
		if val_set is not None:
			self.val_dataloader = DataLoader(val_set, batch_size = self.cfg.dataloader.val_bs, shuffle = self.cfg.dataloder.shuffle)
		if test_sets is not None:
			self.test_dataloaders = []
			for test_set in test_sets:
				self.test_dataloaders.append(DataLoader(test_set, batch_size = self.cfg.dataloader.test_bs, \
					shuffle = self.cfg.dataloder.shuffle))

		if 'have_val_targets' not in self.cfg.dataset:
			self.cfg.dataset.have_val_targets = True
		if 'have_test_targets' not in self.cfg.dataset:
			self.cfg.dataset.have_test_targets = True

		trn_losses = []
		val_losses = []
		test_losses = []
		timing = []
		trn_acces = []
		val_acces = []
		test_acces = []
		subtrn_losses = []
		for i in range(len(test_sets)):
			test_losses.append([])
			test_acces.append([])
		round_acc, round_loss = 4, 4

		self.model = self.create_model()
		self.criterion, self.criterion_nored = self.loss_function()
		optimizer, scheduler = self.optimizer_with_scheduler(self.model)

		for epoch in range(self.cfg.train_args.num_epochs):

			subtrn_loss = 0
			self.model.train()
			start_time = time.time()
			for _, (_, inputs, targets) in enumerate(self.trn_dataloader):
				inputs = inputs.to(self.cfg.train_args.device)
				targets = targets.to(self.cfg.train_args.device, non_blocking=True)
				optimizer.zero_grad()
				outputs = self.model(inputs.float())
				# print('here', outputs[0], inputs.isnan().any(), torch.isnan(inputs.view(-1)).sum().item())
				if self.cfg.loss.type == 'BCELoss':
					loss = self.criterion(outputs.view(-1), targets.float())
				elif self.cfg.loss.type == 'CrossEntropyLoss':
					loss = self.criterion(outputs, targets)
				subtrn_loss += loss.item()
				loss.backward()
				optimizer.step()
			epoch_time = time.time() - start_time
			if not scheduler == None:
				scheduler.step()
			timing.append(epoch_time)
			subtrn_losses.append(subtrn_loss)

			print_str = 'Epoch ' + str(epoch) + ': '

			print_args = self.cfg.train_args.print_args
			if (epoch % self.cfg.train_args.print_every == 0) or (epoch == self.cfg.train_args.num_epochs - 1):
				self.model.eval()

				trn_str, val_str, test_str = '', '', ''
				if 'sigmoid_threshold' not in self.cfg.dataset:
					self.cfg.dataset.sigmoid_threshold = -1

				if ('trn_loss' in print_args) or ('trn_acc' in print_args):
					trn_loss, trn_correct, trn_total, trn_acc, _ = self.evaluate('trn', sigmoid_threshold = self.cfg.dataset.sigmoid_threshold)
					if 'trn_acc' in print_args:
						trn_acces.append(trn_acc)
						trn_str += 'tn_a: '+ str(round(trn_acc, round_acc)) + ' | '
					if 'trn_loss' in print_args:
						trn_losses.append(trn_loss)
						trn_str += 'tn_l: '+str(round(trn_loss, round_loss))

				if val_set is not None and (('val_loss' in print_args) or ('val_acc' in print_args)):
					val_loss, val_correct, val_total, val_acc, _ = self.evaluate('val', sigmoid_threshold = self.cfg.dataset.sigmoid_threshold, 
																				have_targets = self.cfg.dataset.have_val_targets)
					if 'val_acc' in print_args:
						val_acces.append(val_acc)
						val_str += 'v_a: '+str(round(val_acc, round_acc)) + ' | '
					if 'val_loss' in print_args:
						val_losses.append(val_loss)
						val_str += 'v_l: '+str(round(val_loss, round_loss))

				if test_sets is not None and (('test_loss' in print_args) or ('test_acc' in print_args)):
					for test_dataloader_index in range(len(self.test_dataloaders)):
						test_loss, test_correct, test_total, test_acc, _ = self.evaluate('test', sigmoid_threshold = self.cfg.dataset.sigmoid_threshold, 
																							test_index = test_dataloader_index,
																							have_targets = self.cfg.dataset.have_test_targets)
						if 'test_acc' in print_args:
							test_acces[test_dataloader_index].append(test_acc)
							test_str += 't_a'+str(test_dataloader_index)+': '+str(round(test_acc, round_acc)) + ' | '
						if 'test_loss' in print_args:
							test_losses[test_dataloader_index].append(test_loss)
							test_str += 't_l'+str(test_dataloader_index)+': '+str(round(test_loss, round_loss)) + ' | '

				print_str += trn_str + ' || ' + val_str + ' || ' + test_str
				print(print_str)

		max_acces = {'trn': max(trn_acces)}
		if len(val_acces) > 0:
			max_acces['val'] = max(val_acces)
		for i in range(len(test_acces)):
			if len(test_acces[i]) > 0:
				max_acces['test'+str(i)] = max(test_acces[i])
		return max_acces




