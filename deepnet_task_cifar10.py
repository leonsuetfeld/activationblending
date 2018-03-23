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

# use this line to extract avtivations per layer in order to look at transnet outputs
# activations = sess.run(squeezeNet.trans_state, feed_dict = {net.X: [img], net.Y: [1], net.SGD_lr: 1., net.dropout_keep_prob: 1.})
# activations = activations[0]

# ##############################################################################
# ### TASK SETTINGS ############################################################
# ##############################################################################

class TaskSettings(object):

	def __init__(self, args):

		assert args['experiment_name'] is not None, 'experiment_name must be specified.'
		self.mode = args['mode'][0]
		self.experiment_name = args['experiment_name'][0]

		if self.mode != 'analysis':
			assert args['run'] is not None, 'run must be specified.'
			assert args['spec_name'] is not None, 'spec_name must be specified.'
			assert args['minibatch_size'] is not None, 'minibatch_size must be specified.'
			self.spec_name = args['spec_name'][0]
			self.run = args['run'][0]
			self.minibatch_size = args['minibatch_size'][0]
			if self.mode in ['train','training','']:
				assert args['training_schedule'] is not None, 'training_schedule must be specified.'
				assert args['n_minibatches'] is not None, 'n_minibatches (runtime) must be specified for training.'
				assert args['create_val_set'] is not None, 'create_val_set must be specified for training.'
				assert args['val_set_fraction'] is not None, 'val_set_fraction must be specified for training.'
				self.n_minibatches = args['n_minibatches'][0]
				self.training_schedule = args['training_schedule'][0]
				self.create_val_set = args['create_val_set'][0] # True
				self.val_set_fraction = args['val_set_fraction'][0] # 0.05
				self.val_to_val_mbs = 20
				self.restore_model = True
				if args['blend_trainable'][0] or args['swish_beta_trainable'][0]:
					self.af_weights_exist = True
				else:
					self.af_weights_exist = False
				################################################################
				# FILE WRITING OPTIONS ['never', 'once' (if run==1), 'always'] #
				################################################################
				self.tracer_minibatches = [0,50,100]
				self.run_tracer = False					# recommended: False 	-- use only for efficiency optimization
				self.write_summary = False				# recommended: False   	-- use only for debugging
				self.keep_train_val_datasets = False	# recommended: False   	-- only necessary when continuing training after a break
				self.save_args_to_txt = 'always'		# recommended: 'always' -- creates small overview of run in a txt file, also prevents accidental overwrites
				self.save_weights = 'never' 			# recommended: 'never'  -- we currently have no use for these files
				self.save_af_weights = 'always' 		# recommended: 'always' -- af weights can be used as initial parameters for future runs, very small files
				self.save_models = 'once'				# recommended: 'once'   -- to save disk space save models only for run==1 as reference
				self.save_performance_dict = 'always' 	# recommended: 'always' -- VERY IMPORTANT! these are the main log / result files
				# warning message
				if self.save_performance_dict != 'always':
					print('[WARNING] Performance / result files will not be saved.')
		self.print_overview()

	def print_overview(self):
		print('')
		print('###########################################')
		print('### TASK SETTINGS OVERVIEW ################')
		print('###########################################')
		print('')
		print(' - experiment name: "%s"' %(self.experiment_name))
		if self.mode != 'analysis':
			print(' - spec name: "%s"' %(self.spec_name))
			print(' - run: %i' %(self.run))
			if self.mode in ['train','training','']:
				print(' - # of minibatches per run: %i' %(self.n_minibatches))
				print(' - minibatch size: %i' %(self.minibatch_size))
				print(' - training schedule: %s' %(self.training_schedule))

class Paths(object):

	def __init__(self, TaskSettings):
		# data location
		self.train_batches = './1_data_cifar10/train_batches/'
		self.test_batches = './1_data_cifar10/test_batches/'
		self.sample_images = './1_data_cifar10/samples/'
		# save paths (experiment level)
		self.exp_folder = './2_output_cifar10/'+str(TaskSettings.experiment_name)+'/'
		self.af_weights = self.exp_folder+'_af_weights/' 												# corresponds to TaskSettings.save_af_weights
		self.analysis = self.exp_folder+'_analysis/'													# path for analysis files, not used during training
		self.performance_sub = 'performance/'
		if TaskSettings.mode != 'analysis':
			# save paths (spec level)
			self.exp_spec_folder = self.exp_folder+str(TaskSettings.spec_name)+'/'
			self.performance = self.exp_spec_folder+self.performance_sub 								# corresponds to TaskSettings.save_performance_dict
			# save paths (run level)
			self.datasets = self.exp_spec_folder+'datasets/run_'+str(TaskSettings.run)+'/'				# corresponds to TaskSettings.keep_train_val_datasets
			self.models = self.exp_spec_folder+'models/run_'+str(TaskSettings.run)+'/'					# corresponds to TaskSettings.save_models
			self.summaries = self.exp_spec_folder+'summaries/run_'+str(TaskSettings.run)+'/' 		 	# corresponds to TaskSettings.write_summary
			self.chrome_tls = self.exp_spec_folder+'chrome_timelines/run_'+str(TaskSettings.run)+'/' 	# corresponds to TaskSettings.run_tracer
			self.weights = self.exp_spec_folder+'weights/run_'+str(TaskSettings.run)+'/'			 	# corresponds to TaskSettings.save_weights

# ##############################################################################
# ### DATA HANDLER #############################################################
# ##############################################################################

class TrainingHandler(object):

	def __init__(self, TaskSettings, Paths):

		self.TaskSettings = TaskSettings
		self.Paths = Paths
		self.val_mb_counter = 0
		self.train_mb_counter = 0
		self.load_dataset()
		self.shuffle_training_data_idcs()
		self.n_train_minibatches_per_epoch = int(np.floor((self.n_training_samples / TaskSettings.minibatch_size)))
		self.n_val_minibatches = int(self.n_validation_samples / TaskSettings.minibatch_size)

		self.print_overview()

	def reset_val(self):
		self.val_mb_counter = 0

	def load_dataset(self):
	# only call this function once per run, as it will randomly split the, dataset into training set and validation set
		self.dataset_images = []
		self.dataset_labels = []
		for batch in range(5):
			with open(self.Paths.train_batches+'data_batch_' + str(batch+1), 'rb') as file:
				data_dict = pickle.load(file, encoding='bytes')
				images = data_dict[b'data']
				self.dataset_images.extend(images)
				labels = data_dict[b'labels']
				self.dataset_labels.extend(labels)
		self.dataset_images, self.dataset_labels = shuffle(self.dataset_images, self.dataset_labels)
		self.n_total_samples = int(len(self.dataset_labels))
		self.n_training_samples = self.n_total_samples
		self.n_validation_samples = 0
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
		if not os.path.exists(self.Paths.datasets):
			os.makedirs(self.Paths.datasets)
		file_path = self.Paths.datasets+'datasets_'+str(self.TaskSettings.spec_name)+'_run_'+str(self.TaskSettings.run)+'.pkl'
		if os.path.exists(file_path):
			if print_messages:
				print('[MESSAGE] no datasets saved for current spec/run as file already existed: %s'%(file_path))
				print('[MESSAGE] instead restoring datasets from saved file.')
			self.restore_run_datasets()
		else:
			pickle.dump(datasets_dict, open(file_path,'wb'), protocol=3)
			if print_messages:
				print('[MESSAGE] file saved: %s (datasets for current spec/run)'%(file_path))

	def restore_run_datasets(self):
	# loads the current spec_name/runs datasets from a file to make sure the validation set is uncontaminated after restart of run
		file_path = self.Paths.datasets+'datasets_'+str(self.TaskSettings.spec_name)+'_run_'+str(self.TaskSettings.run)+'.pkl'
		if os.path.exists(file_path):
			datasets_dict = pickle.load(open( file_path, 'rb'), encoding='bytes')
			self.training_images = datasets_dict['t_img']
			self.training_labels = datasets_dict['t_lab']
			self.validation_images = datasets_dict['v_img']
			self.validation_labels = datasets_dict['v_lab']
			print('[MESSAGE] spec/run dataset restored from file: %s' %(file_path))
		else:
			raise IOError('\n[ERROR] Couldn\'t restore datasets, stopping run to avoid contamination of validation set.\n')

	def delete_run_datasets(self):
	# deletes the run datasets at the end of a completed run to save storage space
		file_path = self.Paths.datasets+'datasets_'+str(self.TaskSettings.spec_name)+'_run_'+str(self.TaskSettings.run)+'.pkl'
		if os.path.exists(file_path) and not self.TaskSettings.keep_train_val_datasets:
			os.remove(file_path)
			print('[MESSAGE] spec/run dataset deleted to save disk space')
		else:
			print('[MESSAGE] call to delete spec/run dataset had no effect: no dataset file found or TaskSettings.keep_train_val_datasets==True')

	def shuffle_training_data_idcs(self):
		self.training_data_idcs = shuffle(list(range(self.n_training_samples)))

	def get_train_minibatch(self):
		if self.TaskSettings.training_schedule == 'random':
			return self.create_random_train_minibatch()
		elif self.TaskSettings.training_schedule == 'epochs':
			return self.create_next_train_minibatch()

	def create_random_train_minibatch(self):
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
		if self.TaskSettings.training_schedule == 'epochs':
			print(' - # minibatches per epoch: ' + str(self.n_train_minibatches_per_epoch))

class ValidationHandler(object):

	def __init__(self, TaskSettings, Paths):
		self.info = 'Validation set is handled by TrainingHandler, since the validation set is a random fraction of the training set.'

class TestHandler(object):

	def __init__(self, TaskSettings, Paths):
		self.TaskSettings = TaskSettings
		self.Paths = Paths
		self.load_test_data()
		self.n_test_samples = int(len(self.test_images))
		self.n_test_minibatches = int(np.floor(self.n_test_samples/TaskSettings.minibatch_size))
		self.test_mb_counter = 0

		self.print_overview()

	def load_test_data(self):
		self.test_images = []
		self.test_labels = []
		for batch in range(5):
			with open(self.Paths.test_batches+'test_batch', 'rb') as file:
				test_dict = pickle.load(file, encoding='bytes')
				images = test_dict[b'data']
				self.test_images.extend(images)
				labels = test_dict[b'labels']
				self.test_labels.extend(labels)

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

class PerformanceRecorder(object):

	def __init__(self, TaskSettings, Paths):
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
		# top score
		self.is_last_val_top_score = False
		# time keeping
		self.previous_runtime = 0.
		self.init_time = time.time()

	def feed_train_performance(self, mb_n, loss, top1):
		self.train_loss_hist.append(loss)
		self.train_top1_hist.append(top1)
		self.train_mb_n_hist.append(mb_n)

	def feed_val_performance(self, mb_n, loss, top1, apc, af_weights_dict={}):
		self.val_loss_hist.append(loss)
		self.val_top1_hist.append(top1)
		self.val_apc_hist.append(apc)
		self.val_af_weights_hist.append(af_weights_dict)
		self.val_mb_n_hist.append(mb_n)
		# top score evaluation
		self.is_last_val_top_score = False
		if top1 == np.max(np.array(self.val_top1_hist)):
			self.is_last_val_top_score = True

	def feed_test_performance(self, mb_n, loss, top1, apc):
		self.test_loss_hist.append(loss)
		self.test_top1_hist.append(top1)
		self.test_apc_hist.append(apc)
		self.test_mb_n_hist.append(mb_n)

	def top_score(self, current_mb):
		return self.is_last_val_top_score

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

	def restore_from_dict(self):
		# restore dict
		restore_dict_filename = self.Paths.performance+'perf_dict_'+str(self.TaskSettings.spec_name)+'_run_'+str(self.TaskSettings.run)+'.pkl'
		if os.path.exists(restore_dict_filename):
			performance_dict = pickle.load( open( restore_dict_filename, 'rb' ) )
			# names
			if not self.TaskSettings.experiment_name == performance_dict['experiment_name']:
				print('[WARNING] experiment name in restored performance file and current TaskSettings don\'t match up (restored: %s).'%(performance_dict['experiment_name']))
			if not self.TaskSettings.spec_name == performance_dict['spec_name']:
				print('[WARNING] spec name in restored performance file and current TaskSettings don\'t match up (restored: %s).'%(performance_dict['spec_name']))
			if not self.TaskSettings.run == performance_dict['run']:
				print('[WARNING] run in restored performance file and current TaskSettings don\'t match up (restored: %s).'%(performance_dict['run']))
			# training performance
			self.train_loss_hist = performance_dict['train_loss_hist']
			self.train_top1_hist = performance_dict['train_top1_hist']
			self.train_mb_n_hist = performance_dict['train_mb_n_hist']
			# validation performance
			self.val_loss_hist = performance_dict['val_loss_hist']
			self.val_top1_hist = performance_dict['val_top1_hist']
			self.val_apc_hist = performance_dict['val_apc_hist']
			self.val_af_weights_hist = performance_dict['val_af_weights_hist']
			self.val_mb_n_hist = performance_dict['val_mb_n_hist']
			# test performance
			self.test_loss_hist = performance_dict['test_loss']
			self.test_top1_hist = performance_dict['test_top1']
			self.test_apc_hist = performance_dict['test_apc']
			self.test_mb_n_hist = performance_dict['test_mb_n_hist']
			# time
			self.previous_runtime = performance_dict['runtime']
			# return
			print('[MESSAGE] performance recorder restored from file: %s' %(restore_dict_filename))
			return True
		return False

	def save_as_dict(self, print_messages=False):
		# create dict
		performance_dict = {}
		# names
		performance_dict['experiment_name'] = self.TaskSettings.experiment_name
		performance_dict['spec_name'] = self.TaskSettings.spec_name
		performance_dict['run'] = self.TaskSettings.run
		# training performance
		performance_dict['train_loss_hist'] = self.train_loss_hist
		performance_dict['train_top1_hist'] = self.train_top1_hist
		performance_dict['train_mb_n_hist'] = self.train_mb_n_hist
		# validation performance
		performance_dict['val_loss_hist'] = self.val_loss_hist
		performance_dict['val_top1_hist'] = self.val_top1_hist
		performance_dict['val_apc_hist'] = self.val_apc_hist
		performance_dict['val_af_weights_hist'] = self.val_af_weights_hist
		performance_dict['val_mb_n_hist'] = self.val_mb_n_hist
		# test performance
		performance_dict['test_loss'] = self.test_loss_hist
		performance_dict['test_top1'] = self.test_top1_hist
		performance_dict['test_apc'] = self.test_apc_hist
		performance_dict['test_mb_n_hist'] = self.test_mb_n_hist
		# time
		performance_dict['runtime'] = (time.time()-self.init_time)+self.previous_runtime
		# save dict
		if not os.path.exists(self.Paths.performance):
			os.makedirs(self.Paths.performance)
		savepath = self.Paths.performance
		filename = 'perf_dict_'+str(self.TaskSettings.spec_name)+'_run_'+str(self.TaskSettings.run)+'.pkl'
		pickle.dump(performance_dict, open(savepath+filename,'wb'), protocol=3)
		if print_messages:
			print('[MESSAGE] file saved: %s (performance dict)'%(savepath+filename))

class Counter(object):
	def __init__(self):
		self.mb_count_total = 0
	def mb_plus_one(self):
		self.mb_count_total += 1
	def reset(self):
		self.mb_count_total = 0
	def set_counters(self, mb):
		self.mb_count_total = mb

class SessionTimer(object):
	def __init__(self):
		self.session_start_time = 0
		self.session_end_time = 0
		self.session_duration = 0
		self.mb_list = []
		self.mb_duration_list = []
		self.mb_list_val = []
		self.val_duration_list = []
		self.laptime_start = 0
		self.laptime_store = []
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

# ##############################################################################
# ### MAIN FUNCTIONS ###########################################################
# ##############################################################################

def train(TaskSettings, Paths, Network, training_handler, validation_handler, test_handler, counter, timer, rec, args, plot_learning_curves=False):

	assert TaskSettings.training_schedule in ['random', 'epochs'], 'requested training schedule unknown.'
	print('')

	# SESSION CONFIG AND START
	timer.set_session_start_time()
	timer.laptime()
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config = config) as sess:
	# tf.set_random_seed(1)

		# INITIALIZATION OF VARIABLES/ GRAPH, SAVER, SUMMARY WRITER, COUNTERS
		saver = tf.train.Saver()# , write_version=tf.train.SaverDef.V1)
		sess.run(tf.global_variables_initializer()) # initialize all variables (must be done after the graph is constructed and the session is started)
		merged_summary_op = tf.summary.merge_all()
		if TaskSettings.write_summary:
			summary_writer = tf.summary.FileWriter(Paths.summaries, sess.graph)
		n_minibatches_remaining = TaskSettings.n_minibatches

		# RESTORE
		if TaskSettings.restore_model:
			# create model weights folder if it does not exist yet
			if not os.path.exists(Paths.models):
				os.makedirs(Paths.models)
			# make list of files in model weights folder
			files_in_weights_folder = [f for f in os.listdir(Paths.models) if (os.path.isfile(os.path.join(Paths.models, f)) and f.startswith('model'))]
			files_in_weights_folder = sorted(files_in_weights_folder, key=str.lower)
			# find model save file with the highest number (of minibatches) in weight folder
			highest_mb_in_filelist = -1
			for fnum in range(len(files_in_weights_folder)):
				if files_in_weights_folder[fnum].split('.')[-1].startswith('data'):
					if int(files_in_weights_folder[fnum].split('-')[1].split('.')[0]) > highest_mb_in_filelist:
						highest_mb_in_filelist = int(files_in_weights_folder[fnum].split('-')[1].split('.')[0])
						restore_meta_filename = files_in_weights_folder[fnum].split('.')[0]+'.meta'
						restore_data_filename = files_in_weights_folder[fnum].split('.')[0]
			if highest_mb_in_filelist > -1: # if a saved model file was found
				# restore weights, counter, and performance history (recorder)
				n_minibatches_remaining = TaskSettings.n_minibatches - highest_mb_in_filelist
				if n_minibatches_remaining > 0:
					counter.set_counters(highest_mb_in_filelist)
					training_handler.restore_run_datasets()
					rec.restore_from_dict()
					saver.restore(sess, Paths.models+restore_data_filename)
					# print notification
					print('=================================================================================================================================================================================================')
					print('[MESSAGE] model restored from file "' + restore_data_filename + '"')
			# abort training if fully trained model was found
			if n_minibatches_remaining <= 0:
				raise IOError('\n[ERROR] Training aborted. Fully trained model (%i minibatches) for this spec_name / run already exists.'%(highest_mb_in_filelist))

		# IF NOT RESTORE
		else:
			if os.paths.exists(Paths.exp_spec_folder):
				files_in_exp_spec_folder = [f for f in os.listdir(Paths.exp_spec_folder) if (os.path.isfile(os.path.join(Paths.exp_spec_folder, f)) and f.startswith('settings'))]
				for settings_file in files_in_exp_spec_folder:
					run = int(settings_file.split('.')[0].split('run_')[1])
					if run == TaskSettings.run:
						# abort if spec / run has been created before to prevent accidental overwrites
						raise IOError('\n[ERROR] Training aborted. A settings file for "%s", run %i already exists.'%(TaskSettings.spec_name, TaskSettings.run))

		print('=================================================================================================================================================================================================')
		print('=== TRAINING STARTED ============================================================================================================================================================================')
		print('=================================================================================================================================================================================================')

		# save overview of the run as a txt file
		if (TaskSettings.save_args_to_txt == 'always') or (TaskSettings.save_args_to_txt == 'once' and TaskSettings.run == 1):
			args_to_txt(args, Paths)

		validate(sess, Network, training_handler, counter, timer, rec)
		for mb in range(n_minibatches_remaining):

			# MB START
			time_mb_start = time.time()
			counter.mb_plus_one()
			imageBatch, labelBatch = training_handler.get_train_minibatch()

			# SESSION RUN
			input_dict = {Network.X: imageBatch, Network.Y: labelBatch, Network.SGD_lr: lr_linear_decay(counter.mb_count_total), Network.dropout_keep_prob: .5}
			run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
			run_metadata = tf.RunMetadata()

			# different options w.r.t. summaries and tracer
			if TaskSettings.write_summary:
				if TaskSettings.run_tracer and mb in TaskSettings.tracer_minibatches:
					_, loss, top1, summary = sess.run([Network.update, Network.loss, Network.top1, merged_summary_op], feed_dict = input_dict, options=run_options, run_metadata=run_metadata)
				else:
					_, loss, top1, summary = sess.run([Network.update, Network.loss, Network.top1, merged_summary_op], feed_dict = input_dict)
			else:
				if TaskSettings.run_tracer and mb in TaskSettings.tracer_minibatches:
					_, loss, top1 = sess.run([Network.update, Network.loss, Network.top1], feed_dict = input_dict, options=run_options, run_metadata=run_metadata)
				else:
					_, loss, top1 = sess.run([Network.update, Network.loss, Network.top1], feed_dict = input_dict)

			# WRITE SUMMARY AND TRACER FILE
			if TaskSettings.write_summary:
				summary_writer.add_summary(summary,Network.global_step.eval(session=sess))
			if TaskSettings.run_tracer and mb in TaskSettings.tracer_minibatches:
				tl = timeline.Timeline(run_metadata.step_stats)
				ctf = tl.generate_chrome_trace_format()
				if not os.path.exists(Paths.chrome_tls):
					os.makedirs(Paths.chrome_tls)
				with open(Paths.chrome_tls+'timeline_mb'+str(counter.mb_count_total)+'.json', 'w') as f:
					f.write(ctf)

			# STORE RESULTS AND PRINT MINIBATCH OVERVIEW
			mb_duration = time.time()-time_mb_start
			timer.feed_mb_duration(mb, mb_duration)
			t_mb_remaining = ((TaskSettings.n_minibatches-mb)*timer.get_mean_mb_duration(window_length=100))/60.
			t_val_remaining = ((TaskSettings.n_minibatches-mb)*timer.get_mean_val_duration(window_length=100))/60./TaskSettings.val_to_val_mbs
			t_remaining = t_mb_remaining + t_val_remaining
			rec.feed_train_performance(counter.mb_count_total, loss, top1)

			# RUN VALIDATION AND PRINT
			if (mb+1)%TaskSettings.val_to_val_mbs == 0 or mb+1 == TaskSettings.n_minibatches:
				validate(sess, Network, training_handler, counter, timer, rec)
				print('['+str(TaskSettings.spec_name)+', run '+str(TaskSettings.run).zfill(2)+'] mb '+str(counter.mb_count_total).zfill(len(str(TaskSettings.n_minibatches)))+'/'+str(TaskSettings.n_minibatches)+
					  ' | l(t) %05.3f [%05.3f]' %(rec.train_loss_hist[-1], rec.get_running_average(measure='t-loss', window_length=50)) +
					  ' | acc(t) %05.3f [%05.3f]' %(rec.train_top1_hist[-1], rec.get_running_average(measure='t-acc', window_length=50)) +
						  ' | l(v) %05.3f [%05.3f]' %(rec.val_loss_hist[-1], rec.get_running_average(measure='v-loss', window_length=3)) +
						  ' | acc(v) %05.3f [%05.3f]' %(rec.val_top1_hist[-1], rec.get_running_average(measure='v-acc', window_length=3)) +
					  ' | t_mb %05.3f s' %(timer.mb_duration_list[-1]) +
					  ' | t_v %05.3f s' %(timer.val_duration_list[-1]) +
					  ' | t_tot %05.2f min' %((time.time()-timer.session_start_time)/60.) +
					  ' | t_rem %05.2f min' %(t_remaining))
				if rec.top_score(counter.mb_count_total):
					save_model(TaskSettings, Paths, Network, sess, saver, counter, rec, delete_previous=True)

		# AFTER TRAINING COMPLETION: SAVE MODEL WEIGHTS AND PERFORMANCE DICT
		timer.set_session_end_time()
		print('=================================================================================================================================================================================================')
		print('=== TRAINING FINISHED -- MAX VALIDATION ACCURACY: %.3f -- TOTAL TIME: %04.2f MIN ================================================================================================================' %(np.max(rec.val_top1_hist), (timer.session_duration/60.)))
		print('=================================================================================================================================================================================================')

		# TEST AFTER TRAINING IS COMPLETE
		test_acc, test_loss = test(sess, Network, test_handler, counter, rec)
		print('=== TESTING FINISHED -- ACCURACY: %.3f =========================================================================================================================================' %(test_acc)
		print('=================================================================================================================================================================================================')

		# SAVING STUFF
		if counter.mb_count_total == TaskSettings.n_minibatches:
			save_model(TaskSettings, Paths, Network, sess, saver, counter, rec, delete_previous=False)
			print('=================================================================================================================================================================================================')
			if plot_learning_curves:
				visualize_performance(TaskSettings, Paths)
	print('')

def save_model(TaskSettings, Paths, Network, sess, saver, counter, rec, delete_previous=False, print_messsages=False):
	# all weights
	if (TaskSettings.save_weights == 'always') or (TaskSettings.save_weights == 'once' and TaskSettings.run == 1):
		Network.save_all_weights(sess, counter)
	# af weights
	if (TaskSettings.af_weights_exist and TaskSettings.save_af_weights == 'always') or (TaskSettings.af_weights_exist and TaskSettings.save_af_weights == 'once' and TaskSettings.run == 1):
		Network.save_af_weights(sess, counter)
	# rec
	if (TaskSettings.save_performance_dict == 'always') or (TaskSettings.save_performance_dict == 'once' and TaskSettings.run == 1):
		rec.save_as_dict()
	# model
	if not os.path.exists(Paths.models):
		os.makedirs(Paths.models)
	if (TaskSettings.save_models == 'always') or (TaskSettings.save_models == 'once' and TaskSettings.run == 1):
		saver.save(sess,Paths.models+'model', global_step=counter.mb_count_total, write_meta_graph=True)
	if print_messsages:
		print('[MESSAGE] file saved: %s (model)'%(Paths.models+'model'))
	if delete_previous:
		delete_previous_savefiles(TaskSettings, Paths, counter, ['all_weights','af_weights','models'])

def delete_previous_savefiles(TaskSettings, Paths, counter, which_files, print_messsages=False):
	# filenames must be manually defined to match the saved filenames
	current_mb = counter.mb_count_total
	current_run = TaskSettings.run
	del_list = []
	# all weights
	if 'all_weights' in which_files:
		directory = Paths.weights
		if os.path.isdir(directory):
			files_in_dir = [f for f in os.listdir(directory) if 'all_weights_' in f]
			for f in files_in_dir:
				file_mb = int(f.split('.')[0].split('_')[-1])
				if file_mb < current_mb:
					del_list.append(directory+f)
	# af weights
	if 'af_weights' in which_files:
		directory = Paths.af_weights
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

def validate(sess, Network, training_handler, counter, timer, rec, print_val_apc=False):
	# VALIDATION START
	time_val_start = time.time()
	training_handler.reset_val()
	# MINIBATCH HANDLING
	loss_store = []
	top1_store = []
	val_confusion_matrix = np.zeros((10,10))
	val_count_vector = np.zeros((10,1))
	while training_handler.val_mb_counter < training_handler.n_val_minibatches:
		# LOAD VARIABLES & RUN SESSION
		val_imageBatch, val_labelBatch = training_handler.create_next_val_minibatch()
		loss, top1, logits = sess.run([Network.loss, Network.top1, Network.logits], feed_dict = { Network.X: val_imageBatch, Network.Y: val_labelBatch, Network.SGD_lr: 0., Network.dropout_keep_prob: 1.0})
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
	rec.feed_val_performance(counter.mb_count_total, val_loss, val_top1, val_apc, af_weights_dict)
	timer.feed_val_duration(time.time()-time_val_start)

def test(sess, Network, test_handler, counter, rec, print_test_apc=False):
	# TEST START
	test_handler.reset_test()
	# MINIBATCH HANDLING
	loss_store = []
	top1_store = []
	test_confusion_matrix = np.zeros((10,10))
	test_count_vector = np.zeros((10,1))
	while test_handler.test_mb_counter < test_handler.n_test_minibatches:
		# LOAD VARIABLES & RUN SESSION
		test_imageBatch, test_labelBatch = test_handler.create_next_test_minibatch()
		loss, top1, logits = sess.run([Network.loss, Network.top1, Network.logits], feed_dict = { Network.X: test_imageBatch, Network.Y: test_labelBatch, Network.SGD_lr: 0., Network.dropout_keep_prob: 1.0})
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
	rec.feed_test_performance(counter.mb_count_total, test_loss, test_top1, test_apc)
	# RETURN
	return test_top1, test_loss

def test_saved_model(TaskSettings, Paths, Network, test_handler, print_results=False, print_messages=True):
	# SESSION CONFIG AND START
	time_test_start = time.time()
	test_handler.reset_test()
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config = config) as sess:
		# tf.set_random_seed(1)
		# INITIALIZATION OF VARIABLES/ GRAPH, SAVER, SUMMARY WRITER, COUNTERS
		saver = tf.train.Saver()# , write_version=tf.train.SaverDef.V1)
		sess.run(tf.global_variables_initializer()) # initialize all variables (must be done after the graph is constructed and the session is started)

		# RESTORE
		# create model weights folder if it does not exist yet
		if not os.path.exists(Paths.models):
			os.makedirs(Paths.models)
		# make list of files in model weights folder
		files_in_weights_folder = [f for f in os.listdir(Paths.models) if (os.path.isfile(os.path.join(Paths.models, f)) and f.startswith('model'))]
		files_in_weights_folder = sorted(files_in_weights_folder, key=str.lower)
		# find model save file with the highest number (of minibatches) in weight folder
		highest_mb_in_filelist = -1
		for fnum in range(len(files_in_weights_folder)):
			if files_in_weights_folder[fnum].split('.')[-1].startswith('data'):
				if int(files_in_weights_folder[fnum].split('-')[1].split('.')[0]) > highest_mb_in_filelist:
					highest_mb_in_filelist = int(files_in_weights_folder[fnum].split('-')[1].split('.')[0])
					restore_meta_filename = files_in_weights_folder[fnum].split('.')[0]+'.meta'
					restore_data_filename = files_in_weights_folder[fnum].split('.')[0]
		if highest_mb_in_filelist > -1: # if a saved model file was found
			# restore weights, counter, and performance history (recorder)
			saver.restore(sess, Paths.models+restore_data_filename)
			# print notification
			print('=================================================================================================================================================================================================')
			print('[MESSAGE] model restored from file "' + restore_data_filename + '"')
		else:
			raise IOError('\n[ERROR] Test aborted. Couldn\'t find a model of the requested name to test (%s / %s / run %i).\n'%(TaskSettings.experiment_name,TaskSettings.spec_name,TaskSettings.run))

		# MAIN
		print('=================================================================================================================================================================================================')
		print('=== TEST STARTED ================================================================================================================================================================================')
		print('=================================================================================================================================================================================================')
		# create stores for multi-batch processing
		loss_store = []
		top1_store = []
		test_confusion_matrix = np.zeros((10,10))
		test_count_vector = np.zeros((10,1))
		while test_handler.test_mb_counter < test_handler.n_test_minibatches:
			# LOAD DATA & RUN SESSION
			test_imageBatch, test_labelBatch = test_handler.create_next_test_minibatch()
			loss, top1, logits = sess.run([Network.loss, Network.top1, Network.logits], feed_dict = { Network.X: test_imageBatch, Network.Y: test_labelBatch, Network.dropout_keep_prob: 1.0}) # why was this set to 0.5?
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
			print('=================================================================================================================================================================================================')
			print('['+str(TaskSettings.spec_name)+'] test' +
				  ' | l(t): %.3f' %test_loss +
				  ' | acc(t): %.3f' %test_top1 +
				  ' | apc(t): { 1: %.3f |' %test_apc[0] + ' 2: %.3f |' %test_apc[1] + ' 3: %.3f |' %test_apc[2] + ' 4: %.3f |' %test_apc[3] + ' 5: %.3f |' %test_apc[4] +
							  ' 6: %.3f |' %test_apc[5] + ' 7: %.3f |' %test_apc[6] + ' 8: %.3f |' %test_apc[7] + ' 9: %.3f |' %test_apc[8] + ' 10: %.3f }' %test_apc[9] +
				  ' | t_test %05.2f' %(time.time()-time_test_start))
			print('=================================================================================================================================================================================================')
		# create dict
		test_performance_dict = { 'experiment_name': TaskSettings.experiment_name,
								  'spec_name': TaskSettings.spec_name,
								  'run': TaskSettings.run,
								  'test_loss': test_loss,
								  'test_acc': test_top1,
								  'test_apc': test_apc }
		# save dict
		savepath = Paths.performance
		filename = 'test_performance_'+TaskSettings.spec_name+'_run_'+str(TaskSettings.run)+'.pkl'
		pickle.dump(test_performance_dict, open(savepath+filename,'wb'), protocol=3)
		if print_messages:
			print('[MESSAGE] file saved: %s (test performance dict)'%(savepath+filename))
		print('=================================================================================================================================================================================================')
		print('=== TEST FINISHED ===============================================================================================================================================================================')
		print('=================================================================================================================================================================================================')

# ##############################################################################
# ### SUPPORT FUNCTIONS ########################################################
# ##############################################################################

# ============================ VISUALIZE PERFORMANCE ===========================

def visualize_performance(TaskSettings, Paths):
	# load
	filename = Paths.performance+'perf_dict_'+str(TaskSettings.spec_name)+'_run_'+str(TaskSettings.run)+'.pkl'
	performance_dict = pickle.load( open( filename, "rb" ) )
	n_mb_total = int(np.max(performance_dict['train_mb_n_hist']))
	# plot
	fig = plt.figure(figsize=(10,10))
	ax = fig.add_subplot(1,1,1)
	ax.plot(np.array([0,n_mb_total]), np.array([1.0,1.0]), linewidth=3., color='black', label='100%', alpha=0.5)
	ax.plot(np.array(performance_dict['train_mb_n_hist']), performance_dict['train_top1_hist'], linewidth=1., color='green', label='accuracy train', alpha=0.3) # linewidth=2.,
	ax.plot(np.array([0,n_mb_total]), np.array([np.max(performance_dict['val_top1_hist']),np.max(performance_dict['val_top1_hist'])]), linewidth=1.0, color='blue', label='max val acc (%.3f)'%(np.max(performance_dict['val_top1_hist'])), alpha=0.5)
	ax.plot(np.array(performance_dict['val_mb_n_hist']), performance_dict['val_top1_hist'], linewidth=2., color='blue', label='accuracy val', alpha=0.5)
	ax.plot(np.array([0,n_mb_total]), np.array([0.1,0.1]), linewidth=3., color='red', label='chance level', alpha=0.5)
	ax.set_ylim(0.,1.01)
	ax.set_xlim(0.,float(n_mb_total))
	ax.set_xticks(np.arange(0, n_mb_total, 1000))
	ax.set_xticks(np.arange(0, n_mb_total, 200), minor=True)
	ax.set_yticks(np.arange(0., 1.1, .1))
	ax.set_yticks(np.arange(0., 1.1, .02), minor=True)
	ax.grid(which='major', alpha=0.6)
	ax.grid(which='minor', alpha=0.1)
	ax.set_aspect(float(n_mb_total))
	ax.set_title('accuracy over epochs ('+str(TaskSettings.spec_name)+', run_'+str(TaskSettings.run)+')')
	ax.legend(loc=4)
	plt.tight_layout()
	# save
	savepath = Paths.performance
	filename = 'performance_plots_'+str(TaskSettings.spec_name)+'_run_'+str(TaskSettings.run)+'.png'
	if not os.path.exists(savepath):
		os.makedirs(savepath)
	plt.savefig(savepath+filename, dpi = 120, transparent=False, bbox_inches='tight')
	print('=================================================================================================================================================================================================')
	print('[MESSAGE] file saved: %s (performance figure)' %(savepath+filename))
	print('=================================================================================================================================================================================================')
#
# def analysis(TaskSettings, Paths):
# 	experiment_name = Paths.exp_folder.split('/')[-2]
# 	# get spec spec names from their respective folder names
# 	spec_list = [f for f in os.listdir(Paths.exp_folder) if (os.path.isdir(os.path.join(Paths.exp_folder,f)) and not f.startswith('_'))]
# 	# put the pieces together and call spec_analysis() for each spec of the experiment
# 	spec_name_list = []
# 	n_runs_list = []
# 	mb_list = []
# 	best_run_list = []
# 	mean_run_list = []
# 	worst_run_list = []
# 	std_run_list = []
# 	mean_run_max_list = []
# 	var_run_max_list = []
# 	print('')
# 	print('=================================================================================================================================================================================================')
# 	for spec_name in spec_list:
# 		path = Paths.exp_folder+spec_name+'/'+Paths.performance_sub
# 		mb, best_run, worst_run, mean_run, std_run, n_runs, mean_run_max, var_run_max = spec_analysis(TaskSettings, Paths, spec_name=spec_name, perf_files_path=path, make_plot=True)
# 		spec_name_list.append(spec_name)
# 		n_runs_list.append(n_runs)
# 		mb_list.append(mb)
# 		best_run_list.append(best_run)
# 		mean_run_list.append(mean_run)
# 		worst_run_list.append(worst_run)
# 		std_run_list.append(std_run)
# 		mean_run_max_list.append(mean_run_max)
# 		var_run_max_list.append(var_run_max)
# 	print('=================================================================================================================================================================================================')
# 	# PLOTTING
# 	n_mb_total = int(np.max(mb_list[0]))
# 	fig = plt.figure(figsize=(10,10))
# 	ax = fig.add_subplot(1,1,1)
# 	cmap = matplotlib.cm.get_cmap('nipy_spectral')#('Dark2') ('nipy_spectral')
# 	color_list = ['black','blue','green','red']
# 	# layer 1
# 	for spec in range(len(spec_list)):
# 		# ax.plot(mb_list[spec], best_run_list[spec], linewidth=1.5, color=cmap(spec/len(spec_list)), alpha=0.15)
# 		ax.plot(np.array([0,300]), np.array([np.max(mean_run_list[spec]),np.max(mean_run_list[spec])]), linewidth=1.5, linestyle='-', color=cmap(spec/len(spec_list)), alpha=0.8)
# 	# layer 2
# 	for spec in range(len(spec_list)):
# 		ax.plot(mb_list[spec], mean_run_list[spec], linewidth=2.0, color=cmap(spec/len(spec_list)), label='[%i / m %.4f / v %.6f] %s (%.2f / %.2f)' %(n_runs_list[spec], 100*mean_run_max_list[spec], 100*var_run_max_list[spec], spec_list[spec], 100*np.max(mean_run_list[spec]), 100*np.max(best_run_list[spec])), alpha=0.8)
# 	# settings
# 	ax.set_ylim(0.1,1.)
# 	ax.set_xlim(0.,float(n_mb_total))
# 	ax.set_xticks(np.arange(0, float(n_mb_total)+.1, 1000))
# 	ax.set_xticks(np.arange(0, float(n_mb_total)+.1, 500), minor=True)
# 	ax.set_yticks(np.arange(0.1, 1.01, .1))
# 	ax.set_yticks(np.arange(0.1, 1.01, .05), minor=True)
# 	ax.grid(which='major', linewidth=0.5, linestyle='dotted', color='black', alpha=0.3)
# 	ax.grid(which='minor', linewidth=0.5, linestyle='dotted', color='black', alpha=0.3)
# 	ax.set_aspect(float(n_mb_total)+.1/0.901)
# 	ax.set_title('performance analysis (validation accuracy): %s' %(experiment_name))
# 	ax.legend(loc=4)
# 	plt.tight_layout()
# 	# save
# 	savepath = Paths.analysis
# 	filename = 'PA_'+str(experiment_name)+'.png'
# 	if not os.path.exists(savepath):
# 		os.makedirs(savepath)
# 	plt.savefig(savepath+filename, dpi = 120, transparent=False, bbox_inches='tight')
# 	print('[MESSAGE] file saved: %s (performance analysis plot for experiment "%s")' %(savepath+filename, experiment_name))
#
# def spec_analysis(TaskSettings, Paths, spec_name=None, perf_files_path=None, axis_2='af_weights', make_plot=False):
# 	assert (axis_2 in ['af_weights', 'loss']) or axis_2 is None, 'axis_2 needs to be defined as None, \'af_weights\', or \'loss\'.'
# 	# define save path
# 	analysis_savepath = Paths.analysis
# 	# define spec name
# 	if spec_name is None:
# 		spec_name = TaskSettings.spec_name
# 	# get list of all performance files (pickle dicts) within a spec
# 	if perf_files_path == None:
# 		perf_files_path = Paths.performance
# 	perf_files_list = [f for f in os.listdir(perf_files_path) if (os.path.isfile(os.path.join(perf_files_path, f)) and ('.pkl' in f) and not ('test_performance_' in f))]
# 	# extract from dicts
# 	afw_hist_store = []
# 	train_mb_n_hist_store = []
# 	train_top1_hist_store = []
# 	train_loss_hist_store = []
# 	val_mb_n_hist_store = []
# 	val_top1_hist_store = []
# 	val_loss_hist_store = []
# 	run_number_store = []
# 	# make a list of all completed runs
# 	n_val_samples_list = []
# 	for run_file in perf_files_list:
# 		p_dict = pickle.load( open( perf_files_path+run_file, "rb" ) )
# 		run_number = int(run_file.split('run_')[1].split('.')[0])
# 		n_val_samples = len(p_dict['val_mb_n_hist'])
# 		n_val_samples_list.append(n_val_samples)
# 	run_length = np.max(n_val_samples_list)
# 	complete_runs = []
# 	for run_file in perf_files_list:
# 		p_dict = pickle.load( open( perf_files_path+run_file, "rb" ) )
# 		run_number = int(run_file.split('run_')[1].split('.')[0])
# 		if len(p_dict['val_mb_n_hist']) == run_length:
# 			complete_runs.append(run_file)
# 	# extract data from files
# 	for run_file in complete_runs:
# 		p_dict = pickle.load( open( perf_files_path+run_file, "rb" ) )
# 		run_number = int(run_file.split('run_')[1].split('.')[0])
# 		n_val_samples = len(p_dict['val_mb_n_hist'])
# 		if np.mean(p_dict['val_top1_hist']) < .95 and np.mean(p_dict['val_top1_hist']) > .3:
# 			# if len(p_dict['val_af_weights_hist']) > 0:
# 				# build numpy array from af_weights_hist
# 				# af_weights_hist = p_dict['val_af_weights_hist']
# 				# afw_keys = []
# 				# for key in af_weights_hist[0]:
# 				#    afw_keys.append(key)
# 				# afw_hist = np.zeros((len(af_weights_hist), len(afw_keys), len(af_weights_hist[0][afw_keys[0]])))
# 				# for mb in range(len(af_weights_hist)):
# 				# 	for w in range(len(afw_keys)):
# 				# 		weights = np.array(af_weights_hist[mb][afw_keys[w]][:])
# 				# 		print(weights.shape, afw_hist.shape, afw_keys[w])
# 				# 		afw_hist[mb,w,:] = weights
# 				# afw_hist_store.append(afw_hist)
# 			train_mb_n_hist_store.append(np.array(p_dict['train_mb_n_hist']))
# 			train_top1_hist_store.append(np.array(p_dict['train_top1_hist']))
# 			train_loss_hist_store.append(np.array(p_dict['train_loss_hist']))
# 			val_mb_n_hist_store.append(np.array(p_dict['val_mb_n_hist']))
# 			val_top1_hist_store.append(np.array(p_dict['val_top1_hist']))
# 			val_loss_hist_store.append(np.array(p_dict['val_loss_hist']))
# 			run_number_store.append(run_number)
# 		else:
# 			print('[WARNING] bad run detected and excluded from analysis: %s, run %i' %(spec_name, run_number))
# 	if len(run_number_store) > 0:
#
# 		# if len(afw_hist_store) > 0:
# 		# 	val_afw_hist = np.array(afw_hist_store)
# 		# 	mean_afwh = np.mean(val_afw_hist, axis=0)
# 		# 	print(mean_afwh.shape,'hihihi') # correct so far
# 		# else:
# 		# 	mean_afwh = np.zeros(n_val_samples,1)
#
# 		train_mb_n = np.array(train_mb_n_hist_store)
# 		train_top1 = np.array(train_top1_hist_store)
# 		train_loss = np.array(train_loss_hist_store)
# 		val_mb_n = np.array(val_mb_n_hist_store)
# 		val_top1 = np.array(val_top1_hist_store)
# 		val_loss = np.array(val_loss_hist_store)
# 		# get relevant numbers
# 		n_runs = len(run_number_store)
# 		t_mb = train_mb_n[0,:]
# 		v_mb = val_mb_n[0,:]
# 		t_himax_run, t_lomax_run, _, _, t_mean_run, t_std_run, t_run_max_list = get_statistics(train_top1) # write these to dict => pkl
# 		v_himax_run, v_lomax_run, _, _, v_mean_run, v_std_run, v_run_max_list = get_statistics(val_top1) # write these to dict => pkl
# 		_, _, t_loss_himin_run, t_loss_lomin_run, t_loss_mean_run, t_loss_std_run, t_loss_run_max_list = get_statistics(train_loss) # write these to dict => pkl
# 		_, _, v_loss_himin_run, v_loss_lomin_run, v_loss_mean_run, v_loss_std_run, v_loss_run_max_list = get_statistics(val_loss) # write these to dict => pkl
# 		# put data into dict
# 		spec_perf_dict = { 'spec_name': spec_name,
# 							'n_runs': n_runs,
# 							# train
# 							't_mb': t_mb,
# 							't_himax_run': t_himax_run,
# 							't_lomax_run': t_lomax_run,
# 							't_mean_run': t_mean_run,
# 							't_std_run': t_std_run,
# 							't_loss_himin_run': t_loss_himin_run,
# 							't_loss_lomin_run': t_loss_lomin_run,
# 							't_loss_mean_run': t_loss_mean_run,
# 							't_loss_std_run': t_loss_std_run,
# 							't_run_max_list': t_run_max_list,
# 							't_loss_run_max_list': t_loss_run_max_list,
# 							# val
# 							'v_mb': v_mb,
# 							# 'afw_keys': afw_keys,
# 							# 'v_mean_afw': mean_afwh,
# 							'v_himax_run': v_himax_run,
# 							'v_lomax_run': v_lomax_run,
# 							'v_mean_run': v_mean_run,
# 							'v_std_run': v_std_run,
# 							'v_loss_himin_run': v_loss_himin_run,
# 							'v_loss_lomin_run': v_loss_lomin_run,
# 							'v_loss_mean_run': v_loss_mean_run,
# 							'v_loss_std_run': v_loss_std_run,
# 							'v_run_max_list': v_run_max_list,
# 							'v_loss_run_max_list': v_loss_run_max_list }
# 		# save dict as pickle file
# 		if not os.path.exists(analysis_savepath):
# 			os.makedirs(analysis_savepath)
# 		filename = 'performance_summary_'+spec_name+'.pkl'
# 		filepath = analysis_savepath + filename
# 		pickle.dump(spec_perf_dict, open(filepath,'wb'), protocol=3)
# 		print('[MESSAGE] file saved: %s (performance summary dict for "%s")' %(filepath, spec_name))
# 		# PLOTTING
# 		n_mb_total = int(np.max(t_mb))
# 		if make_plot:
# 			fig = plt.figure(figsize=(10,10))
# 			ax = fig.add_subplot(1,1,1)
# 			# plot af_weights
# 			# if axis_2 == 'af_weights':
# 			# 	ax2 = ax.twinx()
# 			# 	for w in range(mean_afwh.shape[1]):
# 			# 		weight_name = afw_keys[w].split('/')[1]
# 			# 		ax2.plot([0,np.max(v_mb)], [1.0,1.0], linewidth=1., color='black', alpha=0.8)
# 			# 		ax2.plot(v_mb, mean_afwh[:,w], linewidth=1., label=weight_name, alpha=0.5)
# 			# 		print(w,':',mean_afwh[:,w]) # debug why are these curves always different? # debug this row
# 			# 		ax2_ylim = [np.min(mean_afwh)-0.1,np.max(mean_afwh)/0.7]
# 			# 		ax2.set_ylim(ax2_ylim[0],ax2_ylim[1])
# 			# 		ax2.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), prop={'size': 11})
# 			# training performance
# 			if axis_2 == 'loss':
# 				ax2 = ax.twinx()
# 				ax2_ylim = [0.,1.]
# 				ax2.plot(t_mb, t_loss_mean_run, linewidth=1., color='red', label='training acc', alpha=0.2)
# 				ax2_ylim = [0.,np.max(t_loss_mean_run)]
# 				ax2.set_ylim(ax2_ylim[0],ax2_ylim[1])
# 			ax.plot(t_mb, t_mean_run, linewidth=1., color='green', label='training acc', alpha=0.4)
# 			# best and worst run
# 			ax.fill_between(v_mb, v_himax_run, v_lomax_run, linewidth=0., color='black', alpha=0.15)
# 			ax.plot(np.array([0,n_mb_total]), np.array([np.max(v_himax_run),np.max(v_himax_run)]), linewidth=1., linestyle='--', color='black', alpha=0.6)
# 			ax.plot(np.array([0,n_mb_total]), np.array([np.max(v_lomax_run),np.max(v_lomax_run)]), linewidth=1., linestyle='--', color='black', alpha=0.6)
# 			ax.plot(v_mb, v_himax_run, linewidth=1., linestyle='-', color='black', label='best run val acc (max = %.3f)'%(np.max(v_himax_run)), alpha=0.6)
# 			ax.plot(v_mb, v_lomax_run, linewidth=1., linestyle='-', color='black', label='worst run val acc (max = %.3f)'%(np.max(v_lomax_run)), alpha=0.6)
# 			# mean run
# 			ax.plot(np.array([0,len(t_mb)]), np.array([np.max(v_mean_run),np.max(v_mean_run)]), linewidth=1., linestyle='--', color='blue', alpha=0.8)
# 			ax.plot(v_mb, v_mean_run, linewidth=2., color='blue', label='mean val acc (max = %.3f)'%(np.max(v_mean_run)), alpha=0.8)
# 			# settings ax
# 			ax.set_ylim(0.1,1.)
# 			ax.set_xlim(0.,float(n_mb_total))
# 			ax.set_xticks(np.arange(0, float(n_mb_total)+.1, 1000))
# 			ax.set_xticks(np.arange(0, float(n_mb_total)+.1, 500), minor=True)
# 			ax.set_yticks(np.arange(0.1, 1.01, .1))
# 			ax.set_yticks(np.arange(0.1, 1.01, .05), minor=True)
# 			ax.grid(which='major', linewidth=0.5, linestyle='dotted', color='black', alpha=0.3)
# 			ax.grid(which='minor', linewidth=0.5, linestyle='dotted', color='black', alpha=0.3)
# 			ax.set_aspect(float(n_mb_total)+.1/0.901)
# 			ax.set_title('performance analysis: %s (%i runs)' %(spec_name, n_runs))
# 			ax.legend(loc='lower left', prop={'size': 11})
# 			# save
# 			plt.tight_layout()
# 			filename = 'performance_analysis_'+str(spec_name)+'.png'
# 			if not os.path.exists(analysis_savepath):
# 				os.makedirs(analysis_savepath)
# 			plt.savefig(analysis_savepath+filename, dpi = 120, transparent=False, bbox_inches='tight')
# 			print('[MESSAGE] file saved: %s (performance analysis plot for spec "%s")' %(spec_name, analysis_savepath+filename))
# 		# return relevant data
# 		return v_mb, v_himax_run, v_lomax_run, v_mean_run, v_std_run, n_runs, np.mean(v_run_max_list), np.var(v_run_max_list)
# 	else:
# 		return 0, 0, 0, 0, 0, 0, 0
#
# def get_statistics(data_array):
# 	# find himax and lomax run's indices
# 	run_max_list = []
# 	run_min_list = []
# 	for i in range(data_array.shape[0]):
# 		run = data_array[i,:]
# 		run_max_list.append(np.max(run))
# 		run_min_list.append(np.min(run))
# 	himax_run_idx = np.argmax(np.array(run_max_list))
# 	lomax_run_idx = np.argmin(np.array(run_max_list))
# 	himin_run_idx = np.argmax(np.array(run_min_list))
# 	lomin_run_idx = np.argmin(np.array(run_min_list))
# 	# isolate himax, mean and lomax run
# 	himax_run = data_array[himax_run_idx,:]
# 	lomax_run = data_array[lomax_run_idx,:]
# 	himin_run = data_array[himin_run_idx,:]
# 	lomin_run = data_array[lomin_run_idx,:]
# 	mean_run = np.mean(data_array, axis=0)
# 	std_run = np.std(data_array, axis=0)
# 	# return many things
# 	return himax_run, lomax_run, himin_run, lomin_run, mean_run, std_run, run_max_list
#
# def smooth_history(hist, n):
# 	smooth_hist = []
# 	for i in range(len(hist)):
# 		smooth_hist.append(sliding_window(hist[:i], n))
# 	return smooth_hist
#
# def sliding_window(l, n):
# 	if len(l) < n:
# 		n = len(l)
# 	return np.mean(l[-n:])

def lr_linear_decay(step, start_lr=0.001, stop_lr=0.00004, total_steps=10000): # default squeezenet: start_lr=0.04, stop_lr=0.00004, total_steps=100000
	if step < 0:
		return 0.
	if step < total_steps:
		return np.linspace(start_lr, stop_lr, num=total_steps)[step]
	return stop_lr

def args_to_txt(args, Paths):
	# prepare
	experiment_name = args['experiment_name'][0]
	spec_name = args['spec_name'][0]
	run = args['run'][0]
	network = args['network'][0]
	task = args['task'][0]
	mode = args['mode'][0]
	# write file
	if not os.path.exists(Paths.exp_spec_folder):
		os.makedirs(Paths.exp_spec_folder)
	filename = "settings_"+str(experiment_name)+"_"+str(spec_name)+"_run_"+str(run)+".txt"
	with open(Paths.exp_spec_folder+filename, "w+") as text_file:
		print("{:>25}".format('RUN SETTINGS:'), file=text_file)
		print("", file=text_file)
		print("{:>25} {:<30}".format('experiment_name:',experiment_name), file=text_file)
		print("{:>25} {:<30}".format('spec_name:', spec_name), file=text_file)
		print("{:>25} {:<30}".format('run:', run), file=text_file)
		print("", file=text_file)
		print("{:>25} {:<30}".format('network:', network), file=text_file)
		print("{:>25} {:<30}".format('task:', task), file=text_file)
		print("{:>25} {:<30}".format('mode:', mode), file=text_file)
		print("", file=text_file)
		for key in args.keys():
			if args[key] is not None and key not in ['experiment_name','spec_name','run','network','task','mode']:
				print("{:>25} {:<30}".format(key+':', str(args[key][0])), file=text_file)
