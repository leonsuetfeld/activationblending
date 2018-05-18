"""
Collection of classes and functions relating to the task. Should not contain
any classes or functions that are independent of the task, or depend on the
network. Move such classes or functions to the _aux or corresonding network_
file.

@authors: Leon Sütfeld, Flemming Brieger
"""

import os
import os.path
import numpy as np
import pickle
import time
import tensorflow as tf
from tensorflow.python.client import timeline
import scipy
import scipy.misc
import scipy.ndimage
import scipy.stats as st
import random
from random import sample
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threading import Thread
import json
import csv
import sys
import argparse
import subprocess
from sklearn.utils import shuffle
import deepnet_aux_cifar as aux
import psutil

# ##############################################################################
# ### TASK SETTINGS ############################################################
# ##############################################################################

class TaskSettings(object):

	def __init__(self, args):

		assert args['path_relative'] is not None, 'path_relative must be specified.'
		assert args['experiment_name'] is not None, 'experiment_name must be specified.'
		self.path_relative = args['path_relative']
		self.mode = args['mode']
		self.experiment_name = args['experiment_name']
		self.task_name = 'cifar10'

		if self.mode != 'analysis':
			assert args['run'] is not None, 'run must be specified.'
			assert args['spec_name'] is not None, 'spec_name must be specified.'
			assert args['minibatch_size'] is not None, 'minibatch_size must be specified.'
			assert args['dropout_keep_probs_inference'] is not None, 'dropout_keep_probs_inference must be specified for training.'
			self.spec_name = args['spec_name']
			self.run = args['run']
			self.minibatch_size = args['minibatch_size']
			self.dropout_keep_probs_inference = args['dropout_keep_probs_inference']
			if self.mode in ['train','training','']:
				assert args['n_minibatches'] is not None, 'n_minibatches (runtime) must be specified for training.'
				assert args['epochs_between_checkpoints'] is not None, 'epochs_between_checkpoints must be specified for training.'
				assert args['safe_af_ws_n'] is not None, 'safe_af_ws_n must be specified for training.'
				assert args['safe_all_ws_n'] is not None, 'safe_all_ws_n must be specified for training.'
				assert args['create_val_set'] is not None, 'create_val_set must be specified for training.'
				assert args['val_set_fraction'] is not None, 'val_set_fraction must be specified for training.'
				assert args['dropout_keep_probs'] is not None, 'dropout_keep_probs must be specified for training.'
				assert args['lr'] is not None, 'lr must be specified for training.'
				assert args['lr_schedule_type'] is not None, 'lr_schedule_type must be specified for training.'
				assert args['lr_decay'] is not None, 'lr_decay must be specified for training.'
				assert args['lr_lin_min'] is not None, 'lr_lin_min must be specified for training.'
				assert args['lr_lin_steps'] is not None, 'lr_lin_steps must be specified for training.'
				assert args['lr_step_ep'] is not None, 'lr_step_ep must be specified for training.'
				assert args['lr_step_multi'] is not None, 'lr_step_multi must be specified for training.'
				assert args['preprocessing'] is not None, 'preprocessing must be specified for training.'
				assert args['walltime'] is not None, 'walltime must be specified for training.'
				assert args['create_checkpoints'] is not None, 'create_checkpoints must be specified for training.'
				assert args['save_af_weights_at_test_mb'] is not None, 'save_af_weights_at_test_mb must be specified for training.'
				assert args['save_all_weights_at_test_mb'] is not None, 'save_all_weights_at_test_mb must be specified for training.'
				assert args['create_lc_on_the_fly'] is not None, 'create_lc_on_the_fly must be specified for training.'
				self.pre_processing = args['preprocessing']
				self.n_minibatches = args['n_minibatches']
				self.create_val_set = args['create_val_set']
				self.val_set_fraction = args['val_set_fraction']
				self.val_to_val_mbs = 20
				self.walltime = args['walltime']
				self.restore_model = True
				self.dropout_keep_probs = args['dropout_keep_probs']
				self.lr = args['lr']
				self.lr_schedule_type = args['lr_schedule_type'] 				# (constant, linear, step, decay)
				self.lr_decay = args['lr_decay']								# (e.g., 1e-6)
				self.lr_lin_min = args['lr_lin_min']							# (e.g., 4*1e-5)
				self.lr_lin_steps = args['lr_lin_steps']						# (e.g., 60000)
				self.lr_step_ep = args['lr_step_ep']
				self.lr_step_multi = args['lr_step_multi']
				if args['blend_trainable'] or args['swish_beta_trainable']:
					self.af_weights_exist = True
				else:
					self.af_weights_exist = False
				################################################################
				# FILE WRITING OPTIONS #########################################
				################################################################
				self.create_checkpoints = args['create_checkpoints']
				self.epochs_between_checkpoints = args['epochs_between_checkpoints']
				self.checkpoints = self.get_checkpoints(50000, args['val_set_fraction'], args['minibatch_size'], args['n_minibatches'], args['epochs_between_checkpoints'])
				self.save_af_weights_at_minibatch = np.around(np.linspace(1, args['n_minibatches'], num=args['safe_af_ws_n'], endpoint=True)).tolist()
				self.save_all_weights_at_minibatch = np.around(np.linspace(1, args['n_minibatches'], num=args['safe_all_ws_n'], endpoint=True)).tolist()
				self.save_af_weights_at_test_mb = args['save_af_weights_at_test_mb']
				self.save_all_weights_at_test_mb = args['save_all_weights_at_test_mb']
				self.create_lc_on_the_fly = args['create_lc_on_the_fly']
				self.tracer_minibatches = [0,50,100]
				self.run_tracer = False					# recommended: False 	-- use only for efficiency optimization
				self.write_summary = False				# recommended: False   	-- use only for debugging
				self.keep_train_val_datasets = False	# recommended: False   	-- only necessary when continuing training after a break
		self.print_overview()

	def get_checkpoints(self, dataset_size, val_set_fraction, mb_size, training_duration_in_mbs, epochs_between_checkpoints):
		trainingset_size = int(np.floor(dataset_size*(1-val_set_fraction)))
		mbs_per_epoch = trainingset_size // mb_size
		mbs_per_checkpoint = mbs_per_epoch * epochs_between_checkpoints
		checkpoints = [0]
		while checkpoints[-1] < training_duration_in_mbs:
			checkpoints.append(checkpoints[-1]+mbs_per_checkpoint)
		checkpoints[-1] = training_duration_in_mbs
		return checkpoints

	def even_splits(self, n, k):
		return np.around(np.linspace(0, args['n_minibatches'], num=args['safe_af_ws_n'], endpoint=True))

	def print_overview(self):
		print('')
		print('###########################################')
		print('### TASK SETTINGS OVERVIEW ################')
		print('###########################################')
		print('')
		print(' - experiment name: "%s"' %(self.experiment_name))
		print(' - task name: "%s"' %(self.task_name))
		if self.mode != 'analysis':
			print(' - spec name: "%s"' %(self.spec_name))
			print(' - run: %i' %(self.run))
			if self.mode in ['train','training','']:
				print(' - pre-processing: %s' %(self.pre_processing))
				print(' - # of minibatches per run: %i' %(self.n_minibatches))
				print(' - minibatch size: %i' %(self.minibatch_size))

class Paths(object):

	def __init__(self, TaskSettings):
		self.relative = TaskSettings.path_relative # path from scheduler
		# data locations
		self.train_batches = self.relative+'1_data_cifar10/train_batches/' 					# original data
		self.test_batches = self.relative+'1_data_cifar10/test_batches/'					# original data
		self.train_set = self.relative+'1_data_cifar10/train_set/'
		self.test_set = self.relative+'1_data_cifar10/test_set/'
		self.train_set_ztrans = self.relative+'1_data_cifar10/train_set_ztrans/'
		self.test_set_ztrans = self.relative+'1_data_cifar10/test_set_ztrans/'
		self.train_set_gcn_zca = self.relative+'1_data_cifar10/train_set_gcn_zca/'
		self.test_set_gcn_zca = self.relative+'1_data_cifar10/test_set_gcn_zca/'
		self.sample_images = self.relative+'1_data_cifar10/samples/'
		# save paths (experiment level)
		self.experiment = self.relative+'3_output_cifar/'+str(TaskSettings.experiment_name)+'/'
		self.af_weight_dicts = self.experiment+'0_af_weights/' 					# corresponds to TaskSettings.save_af_weights
		self.all_weight_dicts = self.experiment+'0_all_weights/' 			 	# corresponds to TaskSettings.save_weights
		self.analysis = self.experiment+'0_analysis/'							# path for analysis files, not used during training
		self.chrome_tls = self.experiment+'0_chrome_timelines/' 				# corresponds to TaskSettings.run_tracer
		# save paths (spec / run level)
		if TaskSettings.mode != 'analysis':
			self.experiment_spec = self.experiment+str(TaskSettings.spec_name)+'/'
			self.experiment_spec_run = self.experiment_spec+'run_'+str(TaskSettings.run)+'/'
			# sub-paths (run level)
			self.info_files = self.experiment_spec_run
			self.recorder_files = self.experiment_spec_run
			self.incomplete_run_info = self.experiment_spec_run
			self.run_learning_curves = self.experiment_spec_run
			self.run_datasets = self.experiment_spec_run+'datasets/'				# corresponds to TaskSettings.keep_train_val_datasets
			self.models = self.experiment_spec_run+'models/'						# corresponds to TaskSettings.save_models
			self.summaries = self.experiment_spec_run+'0_summaries/'							# corresponds to TaskSettings.keep_train_val_datasets

# ##############################################################################
# ### DATA HANDLER #############################################################
# ##############################################################################

class TrainingHandler(object):

	def __init__(self, TaskSettings, Paths, args):

		self.TaskSettings = TaskSettings
		self.Paths = Paths
		self.val_mb_counter = 0
		self.train_mb_counter = 0
		self.load_dataset(args)
		self.split_training_validation()
		self.shuffle_training_data_idcs()
		self.n_train_minibatches_per_epoch = int(np.floor((self.n_training_samples / TaskSettings.minibatch_size)))
		self.n_val_minibatches = int(self.n_validation_samples / TaskSettings.minibatch_size)

		self.print_overview()

	def reset_val(self):
		self.val_mb_counter = 0

	def load_dataset(self, args):
		if args['preprocessing'] in ['none', 'tf_ztrans']:
			path_train_set = self.Paths.train_set+'cifar10_trainset.pkl'
			data_dict = pickle.load(open( path_train_set, 'rb'), encoding='bytes')
			self.dataset_images = data_dict['images']
			self.dataset_labels = data_dict['labels']
		if args['preprocessing'] == 'ztrans':
			path_train_set = self.Paths.train_set_ztrans+'cifar10_trainset.pkl'
			data_dict = pickle.load(open( path_train_set, 'rb'), encoding='bytes')
			self.dataset_images = data_dict['images']
			self.dataset_labels = data_dict['labels']
		elif args['preprocessing'] == 'gcn_zca':
			path_train_set = self.Paths.train_set_gcn_zca+'cifar10_trainset.pkl'
			data_dict = pickle.load(open( path_train_set, 'rb'), encoding='bytes')
			self.dataset_images = data_dict['images']
			self.dataset_labels = data_dict['labels']
		else:
			print('[ERROR] requested preprocessing type unknown (%s)' %(args['preprocessing']))

	def split_training_validation(self):
	# only call this function once per run, as it will randomly split the dataset into training set and validation set
		self.dataset_images, self.dataset_labels = shuffle(self.dataset_images, self.dataset_labels)
		self.n_total_samples = int(len(self.dataset_labels))
		self.n_training_samples = self.n_total_samples
		self.n_validation_samples = 0
		self.validation_images = []
		self.validation_labels = []
		self.training_images = self.dataset_images[:]
		self.training_labels = self.dataset_labels[:]
		if self.TaskSettings.create_val_set:
			# determine train/ val split: make n_validation_samples multiple of minibatch_size
			self.n_validation_samples = np.round(self.n_total_samples*self.TaskSettings.val_set_fraction)
			offset = self.n_validation_samples%self.TaskSettings.minibatch_size
			self.n_validation_samples -= offset
			if offset > 0.5*self.TaskSettings.minibatch_size: # add one minibatch to n_validation_samples if that gets it closer to n_total_samples*val_set_fraction
				self.n_validation_samples += self.TaskSettings.minibatch_size
			self.n_training_samples = int(self.n_total_samples - self.n_validation_samples)
			self.n_validation_samples = int(self.n_validation_samples)
			# split dataset
			self.validation_images = self.dataset_images[:self.n_validation_samples]
			self.validation_labels = self.dataset_labels[:self.n_validation_samples]
			self.training_images = self.dataset_images[self.n_validation_samples:]
			self.training_labels = self.dataset_labels[self.n_validation_samples:]

	def save_run_datasets(self, print_messages=False):
	# saves the current spec_name/runs datasets to a file to allow for an uncontaminated resume after restart of run, should only be called once at the beginning of a session
		datasets_dict = { 't_img': self.training_images,
						  't_lab': self.training_labels,
						  'v_img': self.validation_images,
						  'v_lab': self.validation_labels }
		if not os.path.exists(self.Paths.run_datasets):
			os.makedirs(self.Paths.run_datasets)
		file_path = self.Paths.run_datasets+'datasets_'+str(self.TaskSettings.spec_name)+'_run_'+str(self.TaskSettings.run)+'.pkl'
		if os.path.exists(file_path):
			if print_messages:
				print('[MESSAGE] no datasets saved for current spec/run as file already existed: %s'%(file_path))
				print('[MESSAGE] instead restoring datasets from saved file.')
			self.restore_run_datasets()
		else:
			pickle.dump(datasets_dict, open(file_path,'wb'), protocol=3)
			if print_messages:
				print('[MESSAGE] file saved: %s (datasets for current spec/run)'%(file_path))

	def restore_run_datasets(self, print_messages=False):
	# loads the current spec_name/runs datasets from a file to make sure the validation set is uncontaminated after restart of run
		file_path = self.Paths.run_datasets+'datasets_'+str(self.TaskSettings.spec_name)+'_run_'+str(self.TaskSettings.run)+'.pkl'
		if os.path.exists(file_path):
			datasets_dict = pickle.load(open( file_path, 'rb'), encoding='bytes')
			self.training_images = datasets_dict['t_img']
			self.training_labels = datasets_dict['t_lab']
			self.validation_images = datasets_dict['v_img']
			self.validation_labels = datasets_dict['v_lab']
			if print_messages:
				print('[MESSAGE] spec/run dataset restored from file: %s' %(file_path))
		else:
			raise IOError('\n[ERROR] Couldn\'t restore datasets, stopping run to avoid contamination of validation set.\n')

	def delete_run_datasets(self):
	# deletes the run datasets at the end of a completed run to save storage space
		file_path = self.Paths.run_datasets+'datasets_'+str(self.TaskSettings.spec_name)+'_run_'+str(self.TaskSettings.run)+'.pkl'
		if os.path.exists(file_path) and not self.TaskSettings.keep_train_val_datasets:
			os.remove(file_path)
			print('[MESSAGE] spec/run dataset deleted to save disk space')
		else:
			print('[MESSAGE] call to delete spec/run dataset had no effect: no dataset file found or TaskSettings.keep_train_val_datasets==True')

	def shuffle_training_data_idcs(self):
		self.training_data_idcs = shuffle(list(range(self.n_training_samples)))

	def get_train_minibatch(self):
		return self.create_next_train_minibatch()

	def create_random_train_minibatch(self): # no longer in use
		mb_idcs = random.sample(range(0,self.n_training_samples),self.TaskSettings.minibatch_size)
		random_mb_images = [self.training_images[i] for i in mb_idcs]
		random_mb_labels = [self.training_labels[i] for i in mb_idcs]
		self.train_mb_counter += 1
		return random_mb_images, random_mb_labels

	def create_next_train_minibatch(self):
		if self.train_mb_counter % self.n_train_minibatches_per_epoch == 0:
			self.shuffle_training_data_idcs()
		start_idx = int(self.TaskSettings.minibatch_size*(self.train_mb_counter % self.n_train_minibatches_per_epoch))
		end_idx = int(self.TaskSettings.minibatch_size*((self.train_mb_counter % self.n_train_minibatches_per_epoch)+1))
		mb_idcs = self.training_data_idcs[start_idx:end_idx]
		next_mb_images = [self.training_images[i] for i in mb_idcs]
		next_mb_labels = [self.training_labels[i] for i in mb_idcs]
		self.train_mb_counter += 1
		return next_mb_images, next_mb_labels

	def create_random_val_minibatch(self):
		minibatch_idcs = random.sample(range(0,self.n_validation_samples),self.TaskSettings.minibatch_size)
		random_mb_images = self.validation_images[minibatch_idcs]
		random_mb_labels = self.validation_labels[minibatch_idcs]
		self.val_mb_counter += 1
		return random_mb_images, random_mb_labels

	def create_next_val_minibatch(self):
		start_idx = int(self.TaskSettings.minibatch_size*self.val_mb_counter)
		end_idx = int(self.TaskSettings.minibatch_size*(self.val_mb_counter+1))
		mb_idcs = list(range(start_idx,end_idx))
		next_mb_images = [self.validation_images[i] for i in mb_idcs]
		next_mb_labels = [self.validation_labels[i] for i in mb_idcs]
		self.val_mb_counter += 1
		return next_mb_images, next_mb_labels

	def print_overview(self):
		print('')
		print('###########################################')
		print('### TRAINING & VALIDATION SET OVERVIEW ####')
		print('###########################################')
		print('')
		print(' - desired split: %.4f (t) / %.4f (v)' %(1.0-self.TaskSettings.val_set_fraction, self.TaskSettings.val_set_fraction))
		print(' - actual split: %.4f (t) / %.4f (v)' %(self.n_training_samples/self.n_total_samples, self.n_validation_samples/self.n_total_samples))
		print(' - # total samples: ' + str(self.n_total_samples))
		print(' - # training samples: ' + str(self.n_training_samples))
		print(' - # validation samples: ' + str(self.n_validation_samples))
		print(' - # validation minibatches: ' + str(self.n_val_minibatches))
		print(' - # minibatches per epoch: ' + str(self.n_train_minibatches_per_epoch))

class TestHandler(object):

	def __init__(self, TaskSettings, Paths, args):
		self.TaskSettings = TaskSettings
		self.Paths = Paths
		self.load_test_data(args)
		self.n_test_samples = int(len(self.test_images))
		self.n_test_minibatches = int(np.floor(self.n_test_samples/TaskSettings.minibatch_size))
		self.test_mb_counter = 0
		self.print_overview()

	def load_test_data(self, args):
		if args['preprocessing'] in ['none', 'tf_ztrans']:
			path_test_set = self.Paths.test_set+'cifar10_testset.pkl'
			data_dict = pickle.load(open( path_test_set, 'rb'), encoding='bytes')
			self.test_images = data_dict['images']
			self.test_labels = data_dict['labels']
		elif args['preprocessing'] == 'ztrans':
			path_test_set = self.Paths.test_set_ztrans+'cifar10_testset.pkl'
			data_dict = pickle.load(open( path_test_set, 'rb'), encoding='bytes')
			self.test_images = data_dict['images']
			self.test_labels = data_dict['labels']
		elif args['preprocessing'] == 'gcn_zca':
			path_test_set = self.Paths.test_set_gcn_zca+'cifar10_testset.pkl'
			data_dict = pickle.load(open( path_test_set, 'rb'), encoding='bytes')
			self.test_images = data_dict['images']
			self.test_labels = data_dict['labels']
		else:
			print('[ERROR] requested preprocessing type unknown (%s)' %(args['preprocessing']))

	def create_next_test_minibatch(self):
		start_idx = int(self.TaskSettings.minibatch_size*self.test_mb_counter)
		end_idx = int(self.TaskSettings.minibatch_size*(self.test_mb_counter+1))
		mb_idcs = list(range(start_idx,end_idx))
		next_mb_images = [self.test_images[i] for i in mb_idcs]
		next_mb_labels = [self.test_labels[i] for i in mb_idcs]
		self.test_mb_counter += 1
		return next_mb_images, next_mb_labels

	def reset_test(self):
		self.test_mb_counter = 0

	def print_overview(self):
		print('')
		print('###########################################')
		print('############ TEST SET OVERVIEW ############')
		print('###########################################')
		print('')
		print(' - # test samples: %i (using %i)'%(self.n_test_samples, self.n_test_samples - (self.n_test_samples % self.TaskSettings.minibatch_size)))
		print(' - # test minibatches: '+str(self.n_test_minibatches))

# ##############################################################################
# ### SUPORT CLASSES ###########################################################
# ##############################################################################

class Recorder(object):

	def __init__(self, TaskSettings, TrainingHandler, Paths):
		self.TaskSettings = TaskSettings
		self.Paths = Paths
		# train
		self.train_loss_hist = []
		self.train_top1_hist = []
		self.train_mb_n_hist = []
		# val
		self.val_loss_hist = []
		self.val_top1_hist = []
		self.val_apc_hist = []
		self.val_af_weights_hist = []
		self.val_mb_n_hist = []
		# test
		self.test_loss_hist = []
		self.test_top1_hist = []
		self.test_apc_hist = []
		self.test_mb_n_hist = []
		# split & time keeping
		self.checkpoints = self.TaskSettings.checkpoints
		self.completed_ckpt_list = []
		self.completed_ckpt_mbs = []
		self.completed_ckpt_epochs = []
		# counter
		self.mb_count_total = 0 # current mb
		self.ep_count_total = 0	# current ep
		self.mbs_per_epoch = TrainingHandler.n_train_minibatches_per_epoch
		self.training_completed = False
		self.test_completed = False

	def feed_train_performance(self, loss, top1):
		self.train_loss_hist.append(loss)
		self.train_top1_hist.append(top1)
		self.train_mb_n_hist.append(self.mb_count_total)

	def feed_val_performance(self, loss, top1, apc, af_weights_dict={}):
		self.val_loss_hist.append(loss)
		self.val_top1_hist.append(top1)
		self.val_apc_hist.append(apc)
		self.val_af_weights_hist.append(af_weights_dict)
		self.val_mb_n_hist.append(self.mb_count_total)

	def feed_test_performance(self, loss, top1, apc, mb=-1):
		self.test_loss_hist.append(loss)
		self.test_top1_hist.append(top1)
		self.test_apc_hist.append(apc)
		if mb == -1:
			mb = self.mb_count_total
		self.test_mb_n_hist.append(mb)
		self.test_completed = True

	def mb_plus_one(self):
		self.mb_count_total += 1
		self.ep_count_total = 1 + (self.mb_count_total-1) // self.mbs_per_epoch

	def mark_checkpoint(self):
		if len(self.completed_ckpt_mbs) > 0:
			assert self.completed_ckpt_mbs[-1] != self.mb_count_total, '[ERROR] tried to mark checkpoint twice in same minibatch.'
		assert self.mb_count_total in self.checkpoints, '[ERROR] tried to mark checkpoint outside of defined checkpoints.'
		if len(self.completed_ckpt_list) == 0:
			self.completed_ckpt_list.append(1)
		else:
			self.completed_ckpt_list.append(self.completed_ckpt_list[-1]+1)
		self.completed_ckpt_mbs.append(self.mb_count_total)
		self.completed_ckpt_epochs.append(self.ep_count_total)

	def mark_end_of_session(self): # to iterrupt runs
		if self.mb_count_total == self.TaskSettings.n_minibatches:
			self.training_completed = True
		else:
			with open(self.Paths.incomplete_run_info+'run_incomplete_mb_'+str(self.mb_count_total)+'.txt', "w+") as text_file:
				print('run stopped incomplete. currently stopped after minibatch '+str(self.mb_count_total), file=text_file)
			print('[MESSAGE] incomplete run file written.')

	def get_running_average(self, measure='t-loss', window_length=50):
		assert measure in ['t-loss','t-acc','v-loss','v-acc'], 'requested performance measure unknown.'
		if measure == 't-loss':
			p_measure = self.train_loss_hist
		if measure == 't-acc':
			p_measure = self.train_top1_hist
		if measure == 'v-loss':
			p_measure = self.val_loss_hist
		if measure == 'v-acc':
			p_measure = self.val_top1_hist
		if window_length > len(p_measure):
			window_length = len(p_measure)
		return np.mean(np.array(p_measure)[-window_length:])

	def save_as_dict(self, print_messages=True):
		# create dict
		recorder_dict = {}
		# names
		recorder_dict['task_settings'] = self.TaskSettings
		recorder_dict['paths'] = self.Paths
		recorder_dict['experiment_name'] = self.TaskSettings.experiment_name
		recorder_dict['spec_name'] = self.TaskSettings.spec_name
		recorder_dict['run'] = self.TaskSettings.run
		# training performance
		recorder_dict['train_loss_hist'] = self.train_loss_hist
		recorder_dict['train_top1_hist'] = self.train_top1_hist
		recorder_dict['train_mb_n_hist'] = self.train_mb_n_hist
		# validation performance
		recorder_dict['val_loss_hist'] = self.val_loss_hist
		recorder_dict['val_top1_hist'] = self.val_top1_hist
		recorder_dict['val_apc_hist'] = self.val_apc_hist
		recorder_dict['val_af_weights_hist'] = self.val_af_weights_hist
		recorder_dict['val_mb_n_hist'] = self.val_mb_n_hist
		# test performance
		recorder_dict['test_loss'] = self.test_loss_hist
		recorder_dict['test_top1'] = self.test_top1_hist
		recorder_dict['test_apc'] = self.test_apc_hist
		recorder_dict['test_mb_n_hist'] = self.test_mb_n_hist
		# splits
		recorder_dict['self.checkpoints'] = self.checkpoints
		recorder_dict['completed_ckpt_list'] = self.completed_ckpt_list
		recorder_dict['completed_ckpt_mbs'] = self.completed_ckpt_mbs
		recorder_dict['completed_ckpt_epochs'] = self.completed_ckpt_epochs
		# counter
		recorder_dict['mb_count_total'] = self.mb_count_total
		recorder_dict['ep_count_total'] = self.ep_count_total
		recorder_dict['mbs_per_epoch'] = self.mbs_per_epoch
		recorder_dict['training_completed'] = self.training_completed
		recorder_dict['test_completed'] = self.test_completed
		# save dict
		if not os.path.exists(self.Paths.recorder_files):
			os.makedirs(self.Paths.recorder_files)
		savepath = self.Paths.recorder_files
		filename = 'record_'+str(self.TaskSettings.spec_name)+'_run_'+str(self.TaskSettings.run)+'.pkl'
		pickle.dump(recorder_dict, open(savepath+filename,'wb'), protocol=3)
		if print_messages:
			print('================================================================================================================================================================================================================')
			print('[MESSAGE] recorder dict saved: %s'%(savepath+filename))

	def restore_from_dict(self, Timer):
		# restore dict
		restore_dict_filename = self.Paths.recorder_files+'record_'+str(self.TaskSettings.spec_name)+'_run_'+str(self.TaskSettings.run)+'.pkl'
		if os.path.exists(restore_dict_filename):
			recorder_dict = pickle.load( open( restore_dict_filename, 'rb' ) )
			# names
			if not self.TaskSettings.experiment_name == recorder_dict['experiment_name']:
				print('[WARNING] experiment name in restored performance file and current TaskSettings don\'t match up (restored: %s).'%(recorder_dict['experiment_name']))
			if not self.TaskSettings.spec_name == recorder_dict['spec_name']:
				print('[WARNING] spec name in restored performance file and current TaskSettings don\'t match up (restored: %s).'%(recorder_dict['spec_name']))
			if not self.TaskSettings.run == recorder_dict['run']:
				print('[WARNING] run in restored performance file and current TaskSettings don\'t match up (restored: %s).'%(recorder_dict['run']))
			self.TaskSettings = recorder_dict['task_settings']
			self.Paths = recorder_dict['paths']
			# training performance
			self.train_loss_hist = recorder_dict['train_loss_hist']
			self.train_top1_hist = recorder_dict['train_top1_hist']
			self.train_mb_n_hist = recorder_dict['train_mb_n_hist']
			# validation performance
			self.val_loss_hist = recorder_dict['val_loss_hist']
			self.val_top1_hist = recorder_dict['val_top1_hist']
			self.val_apc_hist = recorder_dict['val_apc_hist'] # accuracy per class
			self.val_af_weights_hist = recorder_dict['val_af_weights_hist']
			self.val_mb_n_hist = recorder_dict['val_mb_n_hist']
			# test performance
			self.test_loss_hist = recorder_dict['test_loss']
			self.test_top1_hist = recorder_dict['test_top1']
			self.test_apc_hist = recorder_dict['test_apc']
			self.test_mb_n_hist = recorder_dict['test_mb_n_hist']
			# splits
			self.checkpoints = recorder_dict['self.checkpoints']
			self.completed_ckpt_list = recorder_dict['completed_ckpt_list']
			self.completed_ckpt_mbs = recorder_dict['completed_ckpt_mbs']
			self.completed_ckpt_epochs = recorder_dict['completed_ckpt_epochs']
			# counter
			self.mb_count_total = recorder_dict['mb_count_total']
			self.ep_count_total = recorder_dict['ep_count_total']
			self.mbs_per_epoch = recorder_dict['mbs_per_epoch']
			self.training_completed = recorder_dict['training_completed']
			self.test_completed = recorder_dict['test_completed']
			# set Timer
			Timer.set_ep_count_at_session_start(self.ep_count_total)
			# return
			print('================================================================================================================================================================================================================')
			print('[MESSAGE] performance recorder restored from file: %s' %(restore_dict_filename))
			return True
		return False

class SessionTimer(object):

	def __init__(self, Paths):
		self.session_start_time = 0
		self.session_end_time = 0
		self.session_duration = 0
		self.mb_list = []
		self.mb_duration_list = []
		self.mb_list_val = []
		self.val_duration_list = []
		self.laptime_start = 0
		self.laptime_store = []
		self.session_shut_down = False
		self.last_checkpoint_time = time.time()
		self.ep_count_at_session_start = 0
		# delete any incomplete run files
		if os.path.exists(Paths.incomplete_run_info):
			incomplete_run_files = [f for f in os.listdir(Paths.incomplete_run_info) if ('run_incomplete' in f)]
			for fname in incomplete_run_files:
				os.remove(Paths.incomplete_run_info+fname)

	def set_ep_count_at_session_start(self, ep_count):
		self.ep_count_at_session_start = ep_count

	def feed_mb_duration(self, mb, mb_duration):
		self.mb_list.append(mb)
		self.mb_duration_list.append(mb_duration)

	def feed_val_duration(self, val_duration):
		self.val_duration_list.append(val_duration)

	def set_session_start_time(self):
		self.session_start_time = time.time()

	def set_session_end_time(self):
		now = time.time()
		self.session_end_time = now
		self.session_duration = now-self.session_start_time

	def get_mean_mb_duration(self, window_length=500):
		if window_length > len(self.mb_duration_list):
			window_length = len(self.mb_duration_list)
		return np.mean(np.array(self.mb_duration_list)) if (window_length == -1) else np.mean(np.array(self.mb_duration_list)[-window_length:])

	def get_mean_val_duration(self, window_length=25):
		if window_length > len(self.val_duration_list):
			window_length = len(self.val_duration_list)
		return np.mean(np.array(self.val_duration_list)) if (window_length == -1) else np.mean(np.array(self.val_duration_list)[-window_length:])

	def laptime(self):
		# keeps 'laptimes', i.e. times between calls of this function. can be used to report printout-to-printout times
		now = time.time()
		latest_laptime = now-self.laptime_start
		self.laptime_start = now
		self.laptime_store.append(latest_laptime)
		return latest_laptime

	def set_checkpoint_time(self):
		self.last_checkpoint_time = time.time()

	def end_session(self):
		self.session_shut_down = True

# ##############################################################################
# ### MAIN FUNCTIONS ###########################################################
# ##############################################################################

def train(TaskSettings, Paths, Network, TrainingHandler, TestHandler, Timer, Rec, args):

	# CREATE RUN FOLDER
	if not os.path.exists(Paths.experiment_spec_run):
		os.makedirs(Paths.experiment_spec_run)
	print('')

	# SESSION CONFIG AND START
	Timer.set_session_start_time()
	Timer.laptime()
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config = config) as sess:
	# tf.set_random_seed(1)

		# INITIALIZATION OF VARIABLES/ GRAPH, SAVER, SUMMARY WRITER
		saver = tf.train.Saver(max_to_keep=100)# , write_version=tf.train.SaverDef.V1)
		sess.run(tf.global_variables_initializer()) # initialize all variables (must be done after the graph is constructed and the session is started)
		merged_summary_op = tf.summary.merge_all()
		if TaskSettings.write_summary:
			summary_writer = tf.summary.FileWriter(Paths.summaries, sess.graph)
			summary_writer.add_graph(sess.graph)
		n_minibatches_remaining = TaskSettings.n_minibatches

		# RESTORE
		if TaskSettings.restore_model:
			model_restored = False
			# create model weights folder if it does not exist yet
			if not os.path.exists(Paths.models):
				os.makedirs(Paths.models)
			# make list of files in model folder
			files_in_models_folder = [f for f in os.listdir(Paths.models) if (os.path.isfile(os.path.join(Paths.models, f)) and f.startswith('model'))]
			files_in_models_folder = sorted(files_in_models_folder, key=str.lower)
			# find model save file with the highest number (of minibatches) in weight folder
			highest_mb_in_filelist = -1
			for fnum in range(len(files_in_models_folder)):
				if files_in_models_folder[fnum].split('.')[-1].startswith('data'):
					if int(files_in_models_folder[fnum].split('-')[1].split('.')[0]) > highest_mb_in_filelist:
						highest_mb_in_filelist = int(files_in_models_folder[fnum].split('-')[1].split('.')[0])
						restore_meta_filename = files_in_models_folder[fnum].split('.')[0]+'.meta'
						restore_data_filename = files_in_models_folder[fnum].split('.')[0]
			if highest_mb_in_filelist > -1: # if a saved model file was found
				# restore weights, counter, and performance history (recorder)
				n_minibatches_remaining = TaskSettings.n_minibatches - highest_mb_in_filelist
				TrainingHandler.restore_run_datasets()
				Rec.restore_from_dict(Timer)
				saver.restore(sess, Paths.models+restore_data_filename)
				model_restored = True
				# print notification
				print('================================================================================================================================================================================================================')
				print('[MESSAGE] model restored from file "' + restore_data_filename + '"')

		# CHECK IF RUN FOLDER EXISTS, ABORT RUN IF IT DOES
		if not TaskSettings.restore_model or not model_restored:
			print('================================================================================================================================================================================================================')
			if os.path.exists(Paths.experiment_spec_run):
				run_info_files = [f for f in os.listdir(Paths.experiment_spec_run) if 'run_info' in f]
				if len(run_info_files) > 0:
					raise IOError('\n[ERROR] Training aborted to prevent accidental overwriting. A run info file for "%s", run %i already exists, but no model was found to restore from or restore_model was set to False. Delete run folder manually to start fresh.'%(TaskSettings.spec_name, TaskSettings.run))
				else:
					print('[MESSAGE] no pre-existing folder found for experiment "%s", spec "%s", run %i. starting fresh run.' %(TaskSettings.experiment_name, TaskSettings.spec_name, TaskSettings.run))
			else:
				print('[MESSAGE] no pre-existing folder found for experiment "%s", spec "%s", run %i. starting fresh run.' %(TaskSettings.experiment_name, TaskSettings.spec_name, TaskSettings.run))

		print('================================================================================================================================================================================================================')
		print('=== TRAINING STARTED ===========================================================================================================================================================================================')
		print('================================================================================================================================================================================================================')

		# save overview of the run as a txt file
		aux.args_to_txt(args, Paths, training_complete_info=str(Rec.training_completed), test_complete_info=str(Rec.test_completed))

		if TaskSettings.create_val_set and Rec.mb_count_total == 0:
			validate(TaskSettings, sess, Network, TrainingHandler, Timer, Rec)

		while n_minibatches_remaining > 0 and Timer.session_shut_down == False:

			# MB START
			time_mb_start = time.time()
			Rec.mb_plus_one()
			imageBatch, labelBatch = TrainingHandler.get_train_minibatch()

			# SESSION RUN
			current_lr = aux.lr_scheduler(TaskSettings, Rec.mb_count_total)
			input_dict = {Network.X: imageBatch, Network.Y: labelBatch, Network.lr: current_lr, Network.dropout_keep_prob: TaskSettings.dropout_keep_probs}
			run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
			run_metadata = tf.RunMetadata()
			# different options w.r.t. summaries and tracer
			if TaskSettings.write_summary:
				if TaskSettings.run_tracer and Rec.mb_count_total in TaskSettings.tracer_minibatches:
					# _, _, loss, top1, summary = sess.run([Network.update, Network.normalize_bw, Network.loss, Network.top1, merged_summary_op], feed_dict = input_dict, options=run_options, run_metadata=run_metadata)
					_, loss, top1, summary = sess.run([Network.update, Network.loss, Network.top1, merged_summary_op], feed_dict = input_dict, options=run_options, run_metadata=run_metadata)
					_ = sess.run([Network.normalize_bw])
				else:
					# _, _, loss, top1, summary = sess.run([Network.update, Network.normalize_bw, Network.loss, Network.top1, merged_summary_op], feed_dict = input_dict)
					_, loss, top1, summary = sess.run([Network.update, Network.loss, Network.top1, merged_summary_op], feed_dict = input_dict)
					_ = sess.run([Network.normalize_bw])
			else:
				if TaskSettings.run_tracer and Rec.mb_count_total in TaskSettings.tracer_minibatches:
					# _, _, loss, top1 = sess.run([Network.update, Network.normalize_bw, Network.loss, Network.top1], feed_dict = input_dict, options=run_options, run_metadata=run_metadata)
					_, loss, top1 = sess.run([Network.update, Network.loss, Network.top1], feed_dict = input_dict, options=run_options, run_metadata=run_metadata)
					_ = sess.run([Network.normalize_bw])
				else:
					# _, _, loss, top1 = sess.run([Network.update, Network.normalize_bw, Network.loss, Network.top1], feed_dict = input_dict)
					_, loss, top1 = sess.run([Network.update, Network.loss, Network.top1], feed_dict = input_dict)
					_ = sess.run([Network.normalize_bw])

			# WRITE SUMMARY AND TRACER FILE
			if TaskSettings.write_summary:
				summary_writer.add_summary(summary,Network.global_step.eval(session=sess))
			if TaskSettings.run_tracer and Rec.mb_count_total in TaskSettings.tracer_minibatches:
				tl = timeline.Timeline(run_metadata.step_stats)
				ctf = tl.generate_chrome_trace_format()
				if not os.path.exists(Paths.chrome_tls):
					os.makedirs(Paths.chrome_tls)
				with open(Paths.chrome_tls+'timeline_mb'+str(Rec.mb_count_total)+'.json', 'w') as f:
					f.write(ctf)

			# TIME KEEPING
			mb_duration = time.time()-time_mb_start
			Timer.feed_mb_duration(Rec.mb_count_total, mb_duration)
			t_remaining_minibatches = ((TaskSettings.n_minibatches-Rec.mb_count_total)*Timer.get_mean_mb_duration(window_length=100))/60.
			t_remaining = t_remaining_minibatches
			if TaskSettings.create_val_set:
				t_remaining_validations = ((TaskSettings.n_minibatches-Rec.mb_count_total)*Timer.get_mean_val_duration(window_length=100))/60./TaskSettings.val_to_val_mbs
				t_remaining += t_remaining_validations

			# FEED TRAINING PERFORMANCE TO RECORDER
			Rec.feed_train_performance(loss, top1)

			# RUN VALIDATION AND PRINT
			if TaskSettings.create_val_set:
				if Rec.mb_count_total % TaskSettings.val_to_val_mbs == 0 or Rec.mb_count_total == TaskSettings.n_minibatches:
					validate(TaskSettings, sess, Network, TrainingHandler, Timer, Rec)
					t_total = (time.time()-Timer.session_start_time)/60.
					pid = os.getpid()
					py = psutil.Process(pid)
					memoryUse = py.memory_info()[0]/2.**30
					print('['+str(TaskSettings.spec_name)+'/'+str(TaskSettings.run).zfill(2)+'] mb '+str(Rec.mb_count_total).zfill(len(str(TaskSettings.n_minibatches)))+'/'+str(TaskSettings.n_minibatches)+
						  ' | lr %0.5f' %(current_lr) +
						  # ' | l(t) %05.3f [%05.3f]' %(Rec.train_loss_hist[-1], Rec.get_running_average(measure='t-loss', window_length=100)) +
						  ' | acc(t) %05.3f [%05.3f]' %(Rec.train_top1_hist[-1], Rec.get_running_average(measure='t-acc', window_length=100)) +
						  # ' | l(v) %05.3f [%05.3f]' %(Rec.val_loss_hist[-1], Rec.get_running_average(measure='v-loss', window_length=10)) +
						  ' | acc(v) %05.3f [%05.3f]' %(Rec.val_top1_hist[-1], Rec.get_running_average(measure='v-acc', window_length=10)) +
						  ' | t_mb %05.3f s' %(Timer.mb_duration_list[-1]) +
						  ' | t_v %05.3f s' %(Timer.val_duration_list[-1]) +
						  ' | t_tot %05.2f m' %(t_total) +
						  ' | t_rem %05.2f m' %(t_remaining) +
						  ' | wt_rem %05.2f m' %(TaskSettings.walltime-t_total) +
						  ' | mem %02.2f GB' %(memoryUse) )
			else:
				if (Rec.mb_count_total)%TaskSettings.val_to_val_mbs == 0 or Rec.mb_count_total == TaskSettings.n_minibatches:
					t_total = (time.time()-Timer.session_start_time)/60.
					pid = os.getpid()
					py = psutil.Process(pid)
					memoryUse = py.memory_info()[0]/2.**30
					print('['+str(TaskSettings.spec_name)+'/'+str(TaskSettings.run).zfill(2)+'] mb '+str(Rec.mb_count_total).zfill(len(str(TaskSettings.n_minibatches)))+'/'+str(TaskSettings.n_minibatches)+
  						  ' | lr %0.5f' %(current_lr) +
						  # ' | l(t) %05.3f [%05.3f]' %(Rec.train_loss_hist[-1], Rec.get_running_average(measure='t-loss', window_length=100)) +
						  ' | acc(t) %05.3f [%05.3f]' %(Rec.train_top1_hist[-1], Rec.get_running_average(measure='t-acc', window_length=100)) +
						  ' | t_mb %05.3f s' %(Timer.mb_duration_list[-1]) +
						  ' | t_tot %05.2f m' %(t_total) +
						  ' | t_rem %05.2f m' %(t_remaining) +
						  ' | wt_rem %05.2f m' %(TaskSettings.walltime-t_total) +
						  ' | mem %02.2f GB' %(memoryUse) )

			# SAVE (AF) WEIGHTS INTERMITTENTLY IF REQUESTED
			if Rec.mb_count_total in TaskSettings.save_af_weights_at_minibatch:
				Network.save_af_weights(sess, Rec.mb_count_total)
			if Rec.mb_count_total in TaskSettings.save_all_weights_at_minibatch:
				Network.save_all_weights(sess, Rec.mb_count_total)

			# SAVE MODEL & DATASETS AT CHECKPOINTS
			if Rec.mb_count_total in TaskSettings.checkpoints or Rec.mb_count_total == TaskSettings.n_minibatches:
				save_model_checkpoint(TaskSettings, TrainingHandler, Paths, Network, sess, saver, Rec, recorder=True, tf_model=True, dataset=True, print_messsages=True)
				if TaskSettings.create_lc_on_the_fly:
					aux.visualize_performance(TaskSettings, Paths)

			# minibatch finished
			n_minibatches_remaining -= 1
			assert n_minibatches_remaining + Rec.mb_count_total == TaskSettings.n_minibatches, '[BUG IN CODE] minibatch counting not aligned.'

			# CHECK IF SESSION (=SPLIT) CAN BE COMPLETED BEFORE WALLTIME
			end_session_now = False
			if Rec.mb_count_total in TaskSettings.checkpoints:
				if TaskSettings.create_checkpoints and TaskSettings.walltime > 0:
					t_total_this_session = (time.time()-Timer.session_start_time)/60.
					epochs_in_session_so_far = Rec.ep_count_total-Timer.ep_count_at_session_start
					checkpoints_in_session_so_far = epochs_in_session_so_far // TaskSettings.epochs_between_checkpoints
					Timer.set_checkpoint_time()
					t_since_last_checkpoint = (time.time() - Timer.last_checkpoint_time)/60.
					t_remaining_until_walltime = TaskSettings.walltime - t_total_this_session
					t_est_for_next_checkpoint = t_since_last_checkpoint
					if checkpoints_in_session_so_far > 0:
						t_per_checkpoint_mean = t_total_this_session / checkpoints_in_session_so_far
						t_est_for_next_checkpoint = np.maximum(t_per_checkpoint_mean, t_since_last_checkpoint)
					if t_est_for_next_checkpoint*2.0+1 > t_remaining_until_walltime:
						end_session_now = True

			# CHECK IF TRAINING IS COMPLETED
			if Rec.mb_count_total == TaskSettings.n_minibatches or end_session_now:
				Rec.mark_end_of_session() # created incomplete_run file or sets Rec.training_completed to True
				Rec.save_as_dict()
				aux.args_to_txt(args, Paths, training_complete_info=str(Rec.training_completed), test_complete_info=str(Rec.test_completed))
				Timer.set_session_end_time()
				Timer.end_session()
				if TaskSettings.write_summary:
					summary_writer.close()

	# AFTER TRAINING COMPLETION: SAVE MODEL WEIGHTS AND PERFORMANCE DICT
	if Rec.training_completed and not Rec.test_completed:
		print('================================================================================================================================================================================================================')
		print('=== TRAINING FINISHED -- TOTAL TIME: %04.2f MIN ================================================================================================================================================================' %(Timer.session_duration/60.))
		print('================================================================================================================================================================================================================')

		# RUN EVALUATION PROCEDURE:
		# (1) smooth validation learning curve & evaluate smoothed curve at model checkpoints, extract minibatch number for checkpoint with highest (smoothed) performance
		# (2) perform test with model from this minibatch number & save results
		# (3) delete saved datasets and all models but the tested one
		if len(Rec.val_top1_hist) > 1:
			early_stopping_mb = early_stopping_minibatch(Rec.val_top1_hist, Rec.val_mb_n_hist, Rec.checkpoints, Paths)
			test_top1, test_loss = test_saved_model(TaskSettings, Paths, Network, TestHandler, Rec, model_mb=early_stopping_mb)
			delete_all_models_but_one(Paths, early_stopping_mb)
		else:
			print('[WARNING] no validation performance record available, performing test with model after %i minibatches (full run).' %(TaskSettings.n_minibatches))
			test_top1, test_loss = test_saved_model(TaskSettings, Paths, Network, TestHandler, Rec, model_mb=TaskSettings.n_minibatches)
			delete_all_models_but_one(Paths, TaskSettings.n_minibatches)
		Rec.save_as_dict()
		aux.args_to_txt(args, Paths, training_complete_info=str(Rec.training_completed), test_complete_info=str(Rec.test_completed))
		TrainingHandler.delete_run_datasets()

		print('================================================================================================================================================================================================================')
		print('=== TEST FINISHED -- ACCURACY: %.3f ============================================================================================================================================================================' %(test_top1))
		print('================================================================================================================================================================================================================')

	print('')

def delete_all_models_but_one(Paths, keep_model_mb):
	# list all files in dir and delete files that don't contain keep_model_mb
	files_in_dir = [f for f in os.listdir(Paths.models) if ('model' in f or 'checkpoint' in f)]
	kill_list = []
	for filename in files_in_dir:
		if not 'model' in filename:
			os.remove(Paths.models+filename)
		else:
			if not str(keep_model_mb) in filename.split('.')[0]:
				os.remove(Paths.models+filename)

def early_stopping_minibatch(val_data, val_mb, checkpoints, Paths, save_plot=True):
	# catch cases in which the smoothing wouldnt really work
	if val_mb[-1] < 1000:
		print('[MESSAGE] no early stopping evaluation performed due to very short run. returning last minibatch (%i).' %(val_mb[-1]))
		return val_mb[-1]
	# calculate size of smoothing window & smoothe data
	val_steps_total = len(val_data)
	mb_per_val = val_mb[-1] // val_steps_total
	smoothing_window_mb = np.minimum(2000, np.maximum(val_mb[-1]//10, 500))
	smoothing_window = smoothing_window_mb // mb_per_val
	smooth_val_data = smooth(val_data, smoothing_window, 3)
	# get intercepts of checkpoints and smooth_val_data
	available_val_data = []
	for i in range(len(checkpoints)):
		val_data_idx_closest_to_save_spot = np.argmin(np.abs(np.array(val_mb)-checkpoints[i]))
		available_val_data.append(smooth_val_data[val_data_idx_closest_to_save_spot])
	# get max of available val data
	max_available_val_data = np.amax(available_val_data)
	minibatch_max_available_val_data = checkpoints[np.argmax(available_val_data)]
	# plot
	if save_plot:
		plt.figure(figsize=(12,8))
		plt.plot(np.array([0, val_mb[-1]]), np.array([np.max(smooth_val_data), np.max(smooth_val_data)]), linestyle='--', linewidth=1, color='black', alpha=0.8)
		for i in range(len(checkpoints)):
			if i == 0:
				plt.plot([checkpoints[i],checkpoints[i]],[0,1], color='blue', linewidth=0.5, label='model save points')
			else:
				plt.plot([checkpoints[i],checkpoints[i]],[0,1], color='blue', linewidth=0.5)
		plt.plot(val_mb, val_data, linewidth=0.5, color='green', alpha=0.5, label='raw validation data')
		plt.plot(val_mb, smooth_val_data, linewidth=2, color='red', label='smoothed validation data')
		plt.plot(checkpoints, available_val_data, linewidth=0.3, color='black', marker='x', markersize=4, alpha=1.0, label='estimated performance at save points')
		plt.plot([minibatch_max_available_val_data], [max_available_val_data], marker='*', markersize=8, color='orange', alpha=1.0, label='max(estimated performance at save point)')
		plt.grid(False)
		plt.ylim([max_available_val_data*0.9,max_available_val_data*1.05])
		plt.legend(loc='lower right', prop={'size': 11})
		plt.tight_layout()
		plt.savefig(Paths.experiment_spec_run+'early_stopping_analysis.png', dpi=300)
	early_stopping_mb = minibatch_max_available_val_data
	return early_stopping_mb

def smooth(y, smoothing_window, times):
	for t in range(times):
		smooth = []
		for i in range(len(y)):
			window_start = np.clip(i-(smoothing_window//2), 0, i)
			window_end = np.clip(i+(smoothing_window//2), 0, len(y))
			smooth.append(np.mean(y[window_start:window_end]))
		y = smooth
	return smooth

def save_model_checkpoint(TaskSettings, TrainingHandler, Paths, Network, sess, saver, Rec, all_weights_dict=False, af_weights_dict=False, recorder=False, tf_model=False, dataset=False, delete_previous=False, print_messsages=False):
	# all weights
	if all_weights_dict:
		Network.save_all_weights(sess, Rec.mb_count_total)
	# af weights
	if af_weights_dict:
		Network.save_af_weights(sess, Rec.mb_count_total)
	# rec
	if recorder:
		Rec.save_as_dict()
	# model
	if tf_model:
		if not os.path.exists(Paths.models):
			os.makedirs(Paths.models)
		saver.save(sess, Paths.models+'model', global_step=Rec.mb_count_total, write_meta_graph=True) # MEMORY LEAK HAPPENING HERE. ANY IDEAS?
	# dataset
	if dataset:
		TrainingHandler.save_run_datasets()
	# delete previous
	if delete_previous:
		delete_previous_savefiles(TaskSettings, Paths, Rec, ['all_weights','af_weights','models'])
	# print message
	if print_messsages:
		print('================================================================================================================================================================================================================')
		print('[MESSAGE] model saved: %s'%(Paths.models+'model'))

def delete_previous_savefiles(TaskSettings, Paths, Rec, which_files, print_messsages=False):
	# filenames must be manually defined to match the saved filenames
	current_mb = Rec.mb_count_total
	current_run = TaskSettings.run
	del_list = []
	# all weights
	if 'all_weights' in which_files:
		directory = Paths.all_weight_dicts
		if os.path.isdir(directory):
			files_in_dir = [f for f in os.listdir(directory) if 'all_weights_' in f]
			for f in files_in_dir:
				file_mb = int(f.split('.')[0].split('_')[-1])
				if file_mb < current_mb:
					del_list.append(directory+f)
	# af weights
	if 'af_weights' in which_files:
		directory = Paths.af_weight_dicts
		if os.path.isdir(directory):
			files_in_dir = [f for f in os.listdir(directory) if 'af_weights_' in f]
			for f in files_in_dir:
				file_mb = int(f.split('.')[0].split('_')[-1])
				file_run = int(f.split('run_')[1].split('_')[0])
				if file_mb < current_mb and file_run == current_run:
					del_list.append(directory+f)
	# model
	if 'models' in which_files:
		directory = Paths.models
		if os.path.isdir(directory):
			files_in_dir = [f for f in os.listdir(directory) if 'model' in f]
			for f in files_in_dir:
				file_mb = int(f.split('.')[0].split('-')[-1])
				if file_mb < current_mb:
					del_list.append(directory+f)
	# delete
	for del_file in del_list:
		os.remove(del_file)
		if print_messsages:
			print('[MESSAGE] file deleted: %s'%(del_file))

def validate(TaskSettings, sess, Network, TrainingHandler, Timer, Rec, print_val_apc=False):
	# VALIDATION START
	time_val_start = time.time()
	TrainingHandler.reset_val()
	# MINIBATCH HANDLING
	loss_store = []
	top1_store = []
	val_confusion_matrix = np.zeros((10,10))
	val_count_vector = np.zeros((10,1))
	while TrainingHandler.val_mb_counter < TrainingHandler.n_val_minibatches:
		# LOAD VARIABLES & RUN SESSION
		val_imageBatch, val_labelBatch = TrainingHandler.create_next_val_minibatch()
		loss, top1, logits = sess.run([Network.loss, Network.top1, Network.logits], feed_dict = { Network.X: val_imageBatch, Network.Y: val_labelBatch, Network.lr: 0., Network.dropout_keep_prob: TaskSettings.dropout_keep_probs_inference})
		# STORE PERFORMANCE
		loss_store.append(loss)
		top1_store.append(top1)
		max_logits = np.argmax(logits, axis=1)
		for entry in range(len(val_labelBatch)):
			val_confusion_matrix[int(val_labelBatch[entry]), max_logits[entry]] += 1.
			val_count_vector[int(val_labelBatch[entry])] += 1.
	# GET MEAN PERFORMANCE OVER VALIDATION MINIBATCHES
	val_loss = np.mean(loss_store)
	val_top1 = np.mean(top1_store)
	val_apc = np.zeros((10,))
	for i in range(10):
		val_apc[i] = np.array(val_confusion_matrix[i,i]/val_count_vector[i])[0]
	if print_val_apc:
		print('[MESSAGE] accuracy per class (v): {1: %.3f |' %val_apc[0] + ' 2: %.3f |' %val_apc[1] + ' 3: %.3f |' %val_apc[2] + ' 4: %.3f |' %val_apc[3] + ' 5: %.3f |' %val_apc[4] +
												' 6: %.3f |' %val_apc[5] + ' 7: %.3f |' %val_apc[6] + ' 8: %.3f |' %val_apc[7] + ' 9: %.3f |' %val_apc[8] + ' 10: %.3f}' %val_apc[9])
	# GET AF WEIGHTS
	af_weights_dict = Network.get_af_weights_dict(sess)
	# STORE RESULTS
	Rec.feed_val_performance(val_loss, val_top1, val_apc, af_weights_dict)
	Timer.feed_val_duration(time.time()-time_val_start)

def test_in_session(TaskSettings, sess, Network, TestHandler, Rec, print_test_apc=False):
	# TEST START
	TestHandler.reset_test()
	# MINIBATCH HANDLING
	loss_store = []
	top1_store = []
	test_confusion_matrix = np.zeros((10,10))
	test_count_vector = np.zeros((10,1))
	while TestHandler.test_mb_counter < TestHandler.n_test_minibatches:
		# LOAD VARIABLES & RUN SESSION
		test_imageBatch, test_labelBatch = TestHandler.create_next_test_minibatch()
		loss, top1, logits = sess.run([Network.loss, Network.top1, Network.logits], feed_dict = { Network.X: test_imageBatch, Network.Y: test_labelBatch, Network.lr: 0., Network.dropout_keep_prob: TaskSettings.dropout_keep_probs_inference})
		# STORE PERFORMANCE
		loss_store.append(loss)
		top1_store.append(top1)
		max_logits = np.argmax(logits, axis=1)
		for entry in range(len(test_labelBatch)):
			test_confusion_matrix[int(test_labelBatch[entry]), max_logits[entry]] += 1.
			test_count_vector[int(test_labelBatch[entry])] += 1.
	# GET MEAN PERFORMANCE OVER VALIDATION MINIBATCHES
	test_loss = np.mean(loss_store)
	test_top1 = np.mean(top1_store)
	test_apc = np.zeros((10,))
	for i in range(10):
		test_apc[i] = np.array(test_confusion_matrix[i,i]/test_count_vector[i])[0]
	if print_test_apc:
		print('[MESSAGE] accuracy per class (test): {1: %.3f |' %test_apc[0] + ' 2: %.3f |' %test_apc[1] + ' 3: %.3f |' %test_apc[2] + ' 4: %.3f |' %test_apc[3] + ' 5: %.3f |' %test_apc[4] +
												   ' 6: %.3f |' %test_apc[5] + ' 7: %.3f |' %test_apc[6] + ' 8: %.3f |' %test_apc[7] + ' 9: %.3f |' %test_apc[8] + ' 10: %.3f}' %test_apc[9])
	# STORE RESULTS
	Rec.feed_test_performance(test_loss, test_top1, test_apc)
	# RETURN
	return test_top1, test_loss

def test_saved_model(TaskSettings, Paths, Network, TestHandler, Rec, model_mb=-1, print_results=False, print_messages=False):
	# SESSION CONFIG AND START
	time_test_start = time.time()
	TestHandler.reset_test()
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config = config) as sess:

		print('debug, testing saved model %i' %(model_mb))

		# INITIALIZATION OF VARIABLES/ GRAPH, SAVER, SUMMARY WRITER, COUNTERS
		saver = tf.train.Saver()# , write_version=tf.train.SaverDef.V1)
		sess.run(tf.global_variables_initializer()) # initialize all variables (must be done after the graph is constructed and the session is started)

		# RESTORE
		# make list of files in model weights folder
		if not os.path.exists(Paths.models):
			os.makedirs(Paths.models)
		files_in_models_folder = [f for f in os.listdir(Paths.models) if (os.path.isfile(os.path.join(Paths.models, f)) and f.startswith('model'))]
		files_in_models_folder = sorted(files_in_models_folder, key=str.lower)
		restore_data_filename = 'none'
		# find model save file with the highest number (of minibatches) in weight folder
		if model_mb == -1:
			highest_mb_in_filelist = -1
			for fname in files_in_models_folder:
				if fname.split('.')[-1].startswith('data') and int(fname.split('-')[1].split('.')[0]) > highest_mb_in_filelist:
					highest_mb_in_filelist = int(fname.split('-')[1].split('.')[0])
					restore_data_filename = fname.split('.')[0]
		# or if a certain model_mb was requested, find that one
		else:
			for fname in files_in_models_folder:
				if fname.split('.')[-1].startswith('data') and int(fname.split('-')[1].split('.')[0]) == model_mb:
					restore_data_filename = fname.split('.')[0]
		# if no model meeting the requirements was found
		if restore_data_filename == 'none':
			raise IOError('\n[ERROR] Test aborted. Couldn\'t find a model of the requested name to test (%s / %s / run %i).\n'%(TaskSettings.experiment_name,TaskSettings.spec_name,TaskSettings.run))
		# restore weights, counter, and performance history (recorder)
		saver.restore(sess, Paths.models+restore_data_filename)
		# print notification
		print('================================================================================================================================================================================================================')
		print('[MESSAGE] model restored from file "' + restore_data_filename + '"')

		# MAIN
		if print_messages:
			print('================================================================================================================================================================================================================')
			print('=== TEST STARTED ===============================================================================================================================================================================================')
			print('================================================================================================================================================================================================================')
		# create stores for multi-batch processing
		loss_store = []
		top1_store = []
		test_confusion_matrix = np.zeros((10,10))
		test_count_vector = np.zeros((10,1))
		while TestHandler.test_mb_counter < TestHandler.n_test_minibatches:
			# LOAD DATA & RUN SESSION
			test_imageBatch, test_labelBatch = TestHandler.create_next_test_minibatch()
			loss, top1, logits = sess.run([Network.loss, Network.top1, Network.logits], feed_dict = { Network.X: test_imageBatch, Network.Y: test_labelBatch, Network.dropout_keep_prob: TaskSettings.dropout_keep_probs_inference}) # why was this set to 0.5?
			# STORE PERFORMANCE
			loss_store.append(loss)
			top1_store.append(top1)
			# ACCURACY PER CLASS CALCULATIONS
			max_logits = np.argmax(logits, axis=1)
			for entry in range(len(test_labelBatch)):
				test_confusion_matrix[int(test_labelBatch[entry]), max_logits[entry]] += 1.
				test_count_vector[int(test_labelBatch[entry])] += 1.
		# GET MEAN PERFORMANCE OVER MINIBATCHES
		test_loss = np.mean(loss_store)
		test_top1 = np.mean(top1_store)
		test_apc = np.zeros((10,))
		for i in range(10):
			test_apc[i] = np.array(test_confusion_matrix[i,i] / test_count_vector[i])[0]
		# STORE RESULTS AND PRINT TEST OVERVIEW
		if print_results:
			print('================================================================================================================================================================================================================')
			print('['+str(TaskSettings.spec_name)+'] test' +
				  ' | l(t): %.3f' %test_loss +
				  ' | acc(t): %.3f' %test_top1 +
				  ' | apc(t): { 1: %.3f |' %test_apc[0] + ' 2: %.3f |' %test_apc[1] + ' 3: %.3f |' %test_apc[2] + ' 4: %.3f |' %test_apc[3] + ' 5: %.3f |' %test_apc[4] +
							  ' 6: %.3f |' %test_apc[5] + ' 7: %.3f |' %test_apc[6] + ' 8: %.3f |' %test_apc[7] + ' 9: %.3f |' %test_apc[8] + ' 10: %.3f }' %test_apc[9] +
				  ' | t_test %05.2f' %(time.time()-time_test_start))
		if print_messages:
			print('================================================================================================================================================================================================================')
			print('=== TEST FINISHED ==============================================================================================================================================================================================')
			print('================================================================================================================================================================================================================')

		# SAVE (AF) WEIGHTS IF REQUESTED
		if TaskSettings.save_af_weights_at_test_mb:
			Network.save_af_weights(sess, model_mb) # to do: add option to mark correct mb
		if TaskSettings.save_all_weights_at_test_mb:
			Network.save_all_weights(sess, model_mb)

	# STORE RESULTS
	Rec.feed_test_performance(test_loss, test_top1, test_apc, mb=model_mb)
	# RETURN
	return test_top1, test_loss
