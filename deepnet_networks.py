"""
Collection of classes and functions relating to the network. Should not contain
any classes or functions that are independent of the network, or depend on the
task. Move such classes or functions to the _aux or corresonding task_ file.

@authors: Leon Sütfeld, Flemming Brieger
"""

import numpy as np
import tensorflow as tf
import os
import pickle
import json

class NetSettings(object):

	def __init__(self, args):

		assert args['experiment_name'] is not None, 'experiment_name must be specified.'
		assert args['spec_name'] is not None, 'spec_name must be specified.'
		assert args['run'] is not None, 'run must be specified.'
		assert args['task'] is not None, 'task must be specified.'
		assert args['minibatch_size'] is not None, 'minibatch_size must be specified.'
		assert args['network'] is not None, 'network must be specified.'
		assert args['af_set'] is not None, 'af_set must be specified.'
		assert args['task'] in ['imagenet', 'cifar10', 'cifar100'], 'task must be \'imagenet\', \'cifar10\', or \'cifar100\'.'
		assert args['af_weights_init'] is not None, 'af_weights_init must be specified.'
		assert args['af_weights_init'] in ['default','predefined'], 'requested setting %s for af_weights_init unknown.' %(args['af_weights_init'])
		assert args['blend_trainable'] is not None, 'blend_trainable must be specified.'
		assert args['blend_mode'] is not None, 'blend_mode must be specified.'
		assert args['swish_beta_trainable'] is not None, 'swish_beta_trainable must be specified.'
		assert args['blend_mode'] in ['unrestricted', 'normalized','posnormed','absnormed','softmaxed'], 'requested setting for blend_mode unknown.'
		assert args['load_af_weights_from'] is not None, 'load_af_weights_from must be specified.'
		assert args['norm_blendw_at_init'] is not None, 'norm_blendw_at_init must be specified.'
		assert args['optimizer'] is not None, 'optimizer must be specified.'
		assert args['lr'] is not None, 'lr must be specified for training.'
		assert args['lr_schedule_type'] is not None, 'lr_schedule_type must be specified for training.'
		assert args['lr_decay'] is not None, 'lr_decay must be specified for training.'
		assert args['lr_lin_min'] is not None, 'lr_lin_min must be specified for training.'
		assert args['lr_lin_steps'] is not None, 'lr_lin_steps must be specified for training.'
		assert args['lr_step_ep'] is not None, 'lr_step_ep must be specified for training.'
		assert args['lr_step_multi'] is not None, 'lr_step_multi must be specified for training.'
		assert args['use_wd'] is not None, 'use_wd must be specified.'
		assert args['wd_lambda'] is not None, 'wd_lambda must be specified.'
		assert args['preprocessing'] is not None, 'preprocessing must be specified.'

		self.mode = args['mode']
		self.task = args['task']
		self.experiment_name = args['experiment_name']
		self.spec_name = args['spec_name']
		self.run = args['run']
		self.minibatch_size = args['minibatch_size']
		self.image_x = 32
		self.image_y = 32
		self.image_z = 3
		if self.task == 'cifar10':
			self.logit_dims = 10
		elif self.task == 'cifar100':
			self.logit_dims = 100
		self.pre_processing = args['preprocessing']
		self.network_spec = args['network']
		self.optimizer_choice = args['optimizer']
		self.lr = args['lr']
		self.lr_schedule_type = args['lr_schedule_type'] 				# (constant, linear, step, decay)
		self.lr_decay = args['lr_decay']								# (e.g., 1e-6)
		self.lr_lin_min = args['lr_lin_min']							# (e.g., 4*1e-5)
		self.lr_lin_steps = args['lr_lin_steps']						# (e.g., 60000)
		self.lr_step_ep = args['lr_step_ep']
		self.lr_step_multi = args['lr_step_multi']
		self.use_wd = args['use_wd']
		self.wd_lambda = args['wd_lambda']
		self.af_set = args['af_set']
		self.af_weights_init = args['af_weights_init']
		self.blend_trainable = args['blend_trainable']
		self.blend_mode = args['blend_mode']
		self.swish_beta_trainable = args['swish_beta_trainable']
		self.init_blendweights_from_spec_name = args['load_af_weights_from']
		self.normalize_blend_weights_at_init = args['norm_blendw_at_init']
		self.print_overview()

	def print_overview(self):
		print('')
		print('###########################################')
		print('### NETWORK SETTINGS OVERVIEW #############')
		print('###########################################')
		print('')
		print(' - network spec: %s' %(self.network_spec))
		print(' - input image format: (%i,%i,%i)' %(self.image_x,self.image_y,self.image_z))
		print(' - output dims: %i' %(self.logit_dims))
		if self.mode in ['train','training','']:
			print(' - optimizer: %s' %(self.optimizer_choice))
			print(' - (initial) learning rate: %i' %(self.lr))
			print(' - multiply lr after epochs: %s' %(str(self.lr_step_ep)))
			print(' - multiply lr by: %s' %(str(self.lr_step_multi)))
			print(' - use weight decay: %s' %(str(self.use_wd)))
			print(' - weight decay lambda: %i' %(self.wd_lambda))
			print(' - AF set: %s' %(self.af_set))
			print(' - af weights initialization: %s' %(self.af_weights_init))
			print(' - blend weights trainable: %s' %(str(self.blend_trainable)))
			print(' - blend mode: %s' %(self.blend_mode))
			print(' - normalize blend_weights at init: %s' %(str(self.normalize_blend_weights_at_init)))
			if 'swish' in self.af_set:
				print(' - swish beta trainable: %s' %(str(self.swish_beta_trainable)))

class Network(object):

	# ======================== GENERAL NETWORK DEFINITION ======================

	def __init__(self, NetSettings, Paths, namescope=None, reuse=False):

		# --- settings & paths -------------------------------------------------
		self.NetSettings = NetSettings
		self.Paths = Paths
		# --- input  --------------------------------------------------
		self.X = tf.placeholder(tf.float32, [NetSettings.minibatch_size, 32,32,3], name='images')
		self.Y = tf.placeholder(tf.int64, [None], name='labels')
		self.lr = tf.placeholder(tf.float32, name='learning_rate')
		self.dropout_keep_prob = tf.placeholder(tf.float32, [None], name='dropout_keep_prob')
		self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
		self.reuse = reuse
		self.Xp = self.X

		# CHOOSE NETWORK ARCHITECTURE
		if self.NetSettings.network_spec == 'sfnet':
			self.logits = self.sfnet(namescope='sfnet')
		elif self.NetSettings.network_spec == 'smcn':
			self.logits = self.smcn(namescope='smcn')
		elif self.NetSettings.network_spec == 'smcnLin':
			self.logits = self.smcnLin(namescope='smcnLin')
		elif self.NetSettings.network_spec == 'smcnDeep':
			self.logits = self.smcnDeep(namescope='smcnDeep')
		elif self.NetSettings.network_spec == 'smcnBN':
			self.logits = self.smcnBN(namescope='smcnBN')
		else:
			print('[ERROR] requested network spec unknown (%s)' %(self.NetSettings.network_spec))

		# OBJECTIVE / EVALUATION
		with tf.name_scope('objective'):
			self.xentropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
			if NetSettings.use_wd:
				self.l2 = [tf.nn.l2_loss(v) for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if 'weight' in v.name and not 'blend_weight' in v.name]
				self.weights_norm = tf.reduce_sum(input_tensor = NetSettings.wd_lambda*tf.stack( self.l2 ), name='weights_norm')
				self.loss = self.xentropy + self.weights_norm
			else:
				self.loss = self.xentropy
			self.top1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.logits, self.Y, 1), tf.float32))
			self.top5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.logits, self.Y, 5), tf.float32))
			tf.summary.scalar('loss', self.loss)
			tf.summary.scalar('top1', self.top1)
			tf.summary.scalar('top5', self.top5)

		# OPTIMIZER
		with tf.name_scope('optimizer'):
			if self.NetSettings.optimizer_choice == 'Adam':
				self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
			elif self.NetSettings.optimizer_choice == 'SGD':
				self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
			elif self.NetSettings.optimizer_choice == 'Momentum':
				self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9, use_nesterov=False)
			elif self.NetSettings.optimizer_choice == 'Nesterov':
				self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9, use_nesterov=True)
			self.minimize = self.optimizer.minimize(self.loss)
			varlist = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
			self.gradients = self.optimizer.compute_gradients(self.loss, var_list=varlist)
			self.update = self.optimizer.apply_gradients(grads_and_vars=self.gradients)

			for grad, var in self.gradients:
				summary_label = var.name+'_gradient'
				summary_label = summary_label.replace('/','_').replace(':','_')
				variable_summaries(grad, summary_label)

	# ========================== NETWORK ARCHITECTURES =========================

	# simple modular convnet
	def smcn(self, namescope=None):
		with tf.name_scope(namescope):
			# input block
			self.state = self.conv2d_parallel_act_layer(layer_input=self.Xp, W_shape=[5,5,3,64], strides=[1,1,1,1], padding='SAME', bias_init=0.0, AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/conv1')
			self.state = tf.nn.dropout(self.state, keep_prob=self.dropout_keep_prob[0], name=namescope+'/dropout1')
			self.state = self.conv2d_parallel_act_layer(layer_input=self.state, W_shape=[3,3,64,64], strides=[1,1,1,1], padding='SAME', AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/conv2')
			# pooling layer
			self.state = tf.nn.max_pool(self.state, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name=namescope+'/pool2')
			# standard block
			self.state = self.conv2d_parallel_act_layer(layer_input=self.state, W_shape=[1,1,64,64], strides=[1,1,1,1], padding='SAME', AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/conv3')
			self.state = tf.nn.dropout(self.state, keep_prob=self.dropout_keep_prob[0], name=namescope+'/dropout3')
			self.state = self.conv2d_parallel_act_layer(layer_input=self.state, W_shape=[5,5,64,64], strides=[1,1,1,1], padding='SAME', AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/conv4')
			# pooling layer
			self.state = tf.nn.max_pool(self.state, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name=namescope+'/pool4')
			# output block
			self.state = self.dense_parallel_act_layer(layer_input=self.state, W_shape=[384], AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, reuse=self.reuse, swish_beta_trainable=self.NetSettings.swish_beta_trainable, varscope=namescope+'/dense5')
			self.state = tf.nn.dropout(self.state, keep_prob=self.dropout_keep_prob[0], name=namescope+'/dropout5')
			self.state = self.dense_parallel_act_layer(layer_input=self.state, W_shape=[192], AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, reuse=self.reuse, swish_beta_trainable=self.NetSettings.swish_beta_trainable, varscope=namescope+'/dense6')
			# output layer
			self.logits = self.dense_parallel_act_layer(layer_input=self.state, W_shape=[self.NetSettings.logit_dims], AF_set=None, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, reuse=self.reuse, swish_beta_trainable=self.NetSettings.swish_beta_trainable, varscope=namescope+'/denseout')
			return self.logits

	# simple modular convnet
	def smcnLin(self, namescope=None):
		with tf.name_scope(namescope):
			self.state = self.conv2d_parallel_act_layer(layer_input=self.Xp, W_shape=[5,5,3,64], strides=[1,1,1,1], padding='SAME', bias_init=0.0, AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/conv1')
			self.state = self.conv2d_parallel_act_layer(layer_input=self.state, W_shape=[3,3,64,64], strides=[1,1,1,1], padding='SAME', AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/conv2')
			self.state = tf.nn.avg_pool(self.state, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=namescope+'/avgpool2')
			self.state = self.conv2d_parallel_act_layer(layer_input=self.state, W_shape=[1,1,64,64], strides=[1,1,1,1], padding='SAME', AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/conv3')
			self.state = self.conv2d_parallel_act_layer(layer_input=self.state, W_shape=[5,5,64,64], strides=[1,1,1,1], padding='SAME', AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/conv4')
			self.state = tf.nn.avg_pool(self.state, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=namescope+'/avgpool4')
			self.state = self.dense_parallel_act_layer(layer_input=self.state, W_shape=[384], AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, reuse=self.reuse, swish_beta_trainable=self.NetSettings.swish_beta_trainable, varscope=namescope+'/dense5')
			self.state = self.dense_parallel_act_layer(layer_input=self.state, W_shape=[192], AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, reuse=self.reuse, swish_beta_trainable=self.NetSettings.swish_beta_trainable, varscope=namescope+'/dense6')
			self.logits = self.dense_parallel_act_layer(layer_input=self.state, W_shape=[self.NetSettings.logit_dims], AF_set=None, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, reuse=self.reuse, swish_beta_trainable=self.NetSettings.swish_beta_trainable, varscope=namescope+'/denseout')
			return self.logits

	# simple modular convnet
	def smcnDeep(self, namescope=None):
		with tf.name_scope(namescope):
			# input block
			self.state = self.conv2d_parallel_act_layer(layer_input=self.Xp, W_shape=[5,5,3,64], strides=[1,1,1,1], padding='SAME', bias_init=0.0, AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/conv1')
			self.state = tf.nn.dropout(self.state, keep_prob=self.dropout_keep_prob[0], name=namescope+'/dropout1')
			self.state = self.conv2d_parallel_act_layer(layer_input=self.state, W_shape=[3,3,64,64], strides=[1,1,1,1], padding='SAME', AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/conv2')
			# pooling layer
			self.state = tf.nn.max_pool(self.state, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name=namescope+'/pool2')
			# standard block
			self.state = self.conv2d_parallel_act_layer(layer_input=self.state, W_shape=[1,1,64,64], strides=[1,1,1,1], padding='SAME', AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/conv3')
			self.state = tf.nn.dropout(self.state, keep_prob=self.dropout_keep_prob[0], name=namescope+'/dropout3')
			self.state = self.conv2d_parallel_act_layer(layer_input=self.state, W_shape=[5,5,64,64], strides=[1,1,1,1], padding='SAME', AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/conv4')
			self.state = self.conv2d_parallel_act_layer(layer_input=self.state, W_shape=[1,1,64,64], strides=[1,1,1,1], padding='SAME', AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/conv5')
			self.state = tf.nn.dropout(self.state, keep_prob=self.dropout_keep_prob[0], name=namescope+'/dropout5')
			self.state = self.conv2d_parallel_act_layer(layer_input=self.state, W_shape=[5,5,64,64], strides=[1,1,1,1], padding='SAME', AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/conv6')
			self.state = self.conv2d_parallel_act_layer(layer_input=self.state, W_shape=[1,1,64,64], strides=[1,1,1,1], padding='SAME', AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/conv7')
			self.state = tf.nn.dropout(self.state, keep_prob=self.dropout_keep_prob[0], name=namescope+'/dropout7')
			self.state = self.conv2d_parallel_act_layer(layer_input=self.state, W_shape=[5,5,64,64], strides=[1,1,1,1], padding='SAME', AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/conv8')
			# pooling layer
			self.state = tf.nn.max_pool(self.state, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name=namescope+'/pool8')
			# output block
			self.state = self.dense_parallel_act_layer(layer_input=self.state, W_shape=[384], AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, reuse=self.reuse, swish_beta_trainable=self.NetSettings.swish_beta_trainable, varscope=namescope+'/dense9')
			self.state = tf.nn.dropout(self.state, keep_prob=self.dropout_keep_prob[0], name=namescope+'/dropout9')
			self.state = self.dense_parallel_act_layer(layer_input=self.state, W_shape=[192], AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, reuse=self.reuse, swish_beta_trainable=self.NetSettings.swish_beta_trainable, varscope=namescope+'/dense10')
			# output layer
			self.logits = self.dense_parallel_act_layer(layer_input=self.state, W_shape=[self.NetSettings.logit_dims], AF_set=None, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, reuse=self.reuse, swish_beta_trainable=self.NetSettings.swish_beta_trainable, varscope=namescope+'/denseout')
			return self.logits

	# simple modular convnet with (pre-act) batch-norm
	def smcnBN(self, namescope=None):
		with tf.name_scope(namescope):
			# input block
			self.state = self.conv2d_parallel_act_layer(layer_input=self.Xp, W_shape=[5,5,3,64], strides=[1,1,1,1], padding='SAME', bias_init=0.0, AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, preblend_batchnorm=True, reuse=self.reuse, varscope=namescope+'/conv1')
			self.state = self.conv2d_parallel_act_layer(layer_input=self.state, W_shape=[3,3,64,64], strides=[1,1,1,1], padding='SAME', AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, preblend_batchnorm=True, reuse=self.reuse, varscope=namescope+'/conv2')
			# pooling layer
			self.state = tf.nn.max_pool(self.state, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name=namescope+'/pool2')
			# standard block
			self.state = self.conv2d_parallel_act_layer(layer_input=self.state, W_shape=[1,1,64,64], strides=[1,1,1,1], padding='SAME', AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, preblend_batchnorm=True, reuse=self.reuse, varscope=namescope+'/conv3')
			self.state = self.conv2d_parallel_act_layer(layer_input=self.state, W_shape=[5,5,64,64], strides=[1,1,1,1], padding='SAME', AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, preblend_batchnorm=True, reuse=self.reuse, varscope=namescope+'/conv4')
			# pooling layer
			self.state = tf.nn.max_pool(self.state, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name=namescope+'/pool4')
			# output block
			self.state = self.dense_parallel_act_layer(layer_input=self.state, W_shape=[384], AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, reuse=self.reuse, swish_beta_trainable=self.NetSettings.swish_beta_trainable, preblend_batchnorm=True, varscope=namescope+'/dense5')
			self.state = self.dense_parallel_act_layer(layer_input=self.state, W_shape=[192], AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, reuse=self.reuse, swish_beta_trainable=self.NetSettings.swish_beta_trainable, preblend_batchnorm=True, varscope=namescope+'/dense6')
			# output layer
			self.logits = self.dense_parallel_act_layer(layer_input=self.state, W_shape=[self.NetSettings.logit_dims], AF_set=None, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, reuse=self.reuse, swish_beta_trainable=self.NetSettings.swish_beta_trainable, varscope=namescope+'/denseout')
			return self.logits

	# ============================= NETWORK LAYERS =============================

	def dense_parallel_act_layer(self, layer_input, W_shape, b_shape=[-1], bias_init=0.1, AF_set=None, af_weights_init='default', W_blend_trainable=True, AF_blend_mode='unrestricted', swish_beta_trainable=True, preblend_batchnorm=False, reuse=False, varscope=None):
		with tf.variable_scope(varscope, reuse=reuse):
			flat_input = tf.layers.flatten(layer_input)
			input_dims = flat_input.get_shape().as_list()[1]
			W_shape = [input_dims, W_shape[0]]
			if b_shape == [-1]:
				b_shape = [W_shape[-1]]
			W = tf.get_variable('weights', W_shape, initializer=tf.truncated_normal_initializer(stddev=tf.sqrt(2./(W_shape[0]*W_shape[1])))) # stddev=0.1
			variable_summaries(W, 'weights')
			b = tf.get_variable('biases', b_shape, initializer=tf.constant_initializer(bias_init))
			variable_summaries(b, 'biases')
			iState = tf.matmul(flat_input, W)
			if b_shape != [0]:
				iState += b
			if AF_set is None:
				return iState
			return self.activate(iState, AF_set, af_weights_init, W_blend_trainable, AF_blend_mode, swish_beta_trainable, varscope, batch_norm=preblend_batchnorm)

	def conv2d_parallel_act_layer(self, layer_input, W_shape, b_shape=[-1], strides=[1,1,1,1], padding='SAME', bias_init=0.1, AF_set=None, af_weights_init='default', W_blend_trainable=True, AF_blend_mode='unrestricted', swish_beta_trainable=True, preblend_batchnorm=False, reuse=False, varscope=None):
		with tf.variable_scope(varscope, reuse=reuse):
			if b_shape == [-1]:
				b_shape = [W_shape[-1]]
			W_initializer = tf.truncated_normal_initializer(stddev=tf.sqrt(2./(W_shape[0]*W_shape[1]*W_shape[2]))) # stddev=0.1
			W = tf.get_variable('weights', W_shape, initializer=W_initializer)
			variable_summaries(W, 'weights')
			b = tf.get_variable('biases', b_shape, initializer=tf.constant_initializer(bias_init))
			variable_summaries(b, 'biases')
			iState = tf.nn.conv2d(layer_input, W, strides, padding)
			if b_shape != [0]:
				iState += b
			if AF_set is None:
				return iState
			return self.activate(iState, AF_set, af_weights_init, W_blend_trainable, AF_blend_mode, swish_beta_trainable, varscope, batch_norm=preblend_batchnorm)

	# ============================== ACTIVATIONS ===============================

	def activate(self, preact, AF_set, af_weights_init, W_blend_trainable, AF_blend_mode, swish_beta_trainable, layer_name, batch_norm=False):
		assert af_weights_init in ['default','predefined'], 'specified initialization type for W_blend unknown.'
		assert AF_blend_mode in ['unrestricted','normalized','posnormed','absnormed','softmaxed'], 'specified blend mode unknown.'

		if batch_norm:
			preact = tf.layers.batch_normalization(preact, name='batchnorm', training=True)

		# INITIALIZE BLEND WEIGHTS
		if af_weights_init == 'default':
			n_AFs = int(AF_set.split('_')[0])
			af_weights_initializer = tf.fill([n_AFs], 1/float(n_AFs))
		elif af_weights_init == 'predefined':
			try:
				af_weights_initializer = self.get_predefined_af_weights(layer_name, w_type='blend')
			except Exception:
				raise IOError('\n[ERROR] Couldn\'t restore predefined blend weights, stopping run.\n')
		blend_weights_raw = tf.get_variable("blend_weights_raw", trainable=W_blend_trainable, initializer=af_weights_initializer)
		if AF_blend_mode == 'unrestricted':
			blend_weights = blend_weights_raw
		elif AF_blend_mode == 'normalized':
			blend_weights = tf.divide(blend_weights_raw, tf.reduce_sum(blend_weights_raw, keep_dims=True))
		elif AF_blend_mode == 'absnormed':
			blend_weights = tf.divide(blend_weights_raw, tf.reduce_sum(tf.abs(blend_weights_raw), keep_dims=True))
		elif AF_blend_mode == 'posnormed':
			blend_weights = tf.divide(tf.clip_by_value(blend_weights_raw, 0.0001, 1000.0), tf.reduce_sum(tf.clip_by_value(blend_weights_raw, 0.0001, 1000.0), keep_dims=True))
			blend_weights = tf.divide(blend_weights_raw, tf.reduce_sum(blend_weights_raw, keep_dims=True))
		elif AF_blend_mode == 'softmaxed':
			blend_weights = tf.exp(blend_weights_raw)
			blend_weights = tf.divide(blend_weights, tf.reduce_sum(blend_weights, keep_dims=True))

		# SWISH BLOCK
		if 'swish' in AF_set:
			if af_weights_init == 'default':
				swish_init = tf.fill([1], 1.0)
			elif af_weights_init == 'predefined':
				try:
					swish_init = self.get_predefined_af_weights(layer_name, w_type='swish_beta')
				except Exception:
					raise IOError('\n[ERROR] Couldn\'t restore predefined swish beta, stopping run.\n')
			swish_beta = tf.get_variable("swish_beta", trainable=swish_beta_trainable, initializer=swish_init)

		# AF sets
		if AF_set.startswith('1_'):
			# relu
			if 'relu' in AF_set:
				act_relu = tf.nn.relu(preact)
				return blend_weights[0] * act_relu
			# elu
			if 'jelu' in AF_set:
				act_elu = tf.nn.elu(preact)
				return blend_weights[0] * act_elu
			# tanh
			if 'tanh' in AF_set:
				act_tanh = tf.nn.tanh(preact)
				return blend_weights[0] * act_tanh
			# swish
			if 'jswish' in AF_set:
				act_swish = preact * tf.nn.sigmoid(swish_beta*preact)
				return blend_weights[0] * act_swish
			# linu
			if 'linu' in AF_set:
				act_linu = preact
				return blend_weights[0] * act_linu
			# selu
			if 'selu' in AF_set:
				act_selu = tf.nn.selu(preact)
				return blend_weights[0] * act_selu

		if 'blend5_swish' in AF_set:
			act_relu = tf.nn.relu(preact)
			act_elu = tf.nn.elu(preact)
			act_tanh = tf.nn.tanh(preact)
			act_swish = preact * tf.nn.sigmoid(swish_beta*preact)
			act_linu = preact
			if batch_norm:
				act_relu = tf.layers.batch_normalization(act_relu, name='batchnorm_relu', training=True)
				act_elu = tf.layers.batch_normalization(act_elu, name='batchnorm_elu', training=True)
				act_tanh = tf.layers.batch_normalization(act_tanh, name='batchnorm_tanh', training=True)
				act_swish = tf.layers.batch_normalization(act_swish, name='batchnorm_swish', training=True)
				act_linu = tf.layers.batch_normalization(act_linu, name='batchnorm_linu', training=True)
			return tf.add_n([blend_weights[0] * act_relu,
							 blend_weights[1] * act_elu,
							 blend_weights[2] * act_tanh,
							 blend_weights[3] * act_swish,
							 blend_weights[4] * act_linu])

	def get_predefined_af_weights(self, layer_name, w_type='blend'):
		# define which file to load the blend weights from
		blend_weights_files = [f for f in os.listdir(self.Paths.af_weight_dicts) if '.pkl' in f and 'run_'+str(self.NetSettings.run) in f and self.NetSettings.init_blendweights_from_spec_name in f]
		if len(blend_weights_files) > 0:
			file_num = 0
			# if there are multiple save files from different minibatches, find largest mb in list
			if len(blend_weights_files) > 1:
				highest_mb_count = 0
				for i in range(len(blend_weights_files)):
					mb_count = int(blend_weights_files[i].split('mb_')[-1].split('.')[0])
					if mb_count > highest_mb_count:
						highest_mb_count = mb_count
						file_num = i
			blend_weights_file = blend_weights_files[file_num]
			af_weights_dict = pickle.load(open(self.Paths.af_weight_dicts+blend_weights_file,'rb'))
			print('[MESSAGE] (predefined) blend weights loaded from file "%s"' %(self.Paths.af_weight_dicts+blend_weights_file))
		else:
			raise ValueError('Could not find af weights file containing "%s" and "run_%i"' %(self.NetSettings.init_blendweights_from_spec_name, self.NetSettings.run))
		# extract from dict
		if w_type == 'blend':
			layer_blend_weights = af_weights_dict[layer_name+'/blend_weights:0']
			if self.NetSettings.normalize_blend_weights_at_init:
				layer_blend_weights /= tf.reduce_sum(layer_blend_weights)
			return tf.convert_to_tensor(layer_blend_weights)
		if w_type == 'swish_beta':
			return tf.convert_to_tensor(af_weights_dict[layer_name+'/swish_beta:0'])

	def get_af_weights_dict(self, sess):
		af_weights_dict = {}
		# if trainable blend weights available put in list
		if len([v for v in tf.trainable_variables() if 'blend_weights' in v.name]) > 0:
			for name in [v.name for v in tf.trainable_variables() if 'blend_weights' in v.name]:
				af_weights_dict[name] = list(sess.run(name))
		# if trainable swish betas available put in list
		if len([v for v in tf.trainable_variables() if 'swish_beta' in v.name]) > 0:
			for name in [v.name for v in tf.trainable_variables() if 'swish_beta' in v.name]:
				af_weights_dict[name] = list(sess.run(name))
		return af_weights_dict

	def save_af_weights(self, sess, mb_count, print_messages=False):
		af_weights_dict_pkl = {}
		af_weights_dict_json = {}
		# if trainable blend weights available put in dicts
		if len([v for v in tf.trainable_variables() if 'blend_weights' in v.name]) > 0:
			for name in [v.name for v in tf.trainable_variables() if 'blend_weights' in v.name]:
				af_weights_dict_pkl[name] = list(sess.run(name))
				af_weights_dict_json[name] = str(list(sess.run(name)))
		# if trainable swish betas available put in dicts
		if len([v for v in tf.trainable_variables() if 'swish_beta' in v.name]) > 0:
			for name in [v.name for v in tf.trainable_variables() if 'swish_beta' in v.name]:
				af_weights_dict_pkl[name] = list(sess.run(name))
				af_weights_dict_json[name] = str(list(sess.run(name)))
		# save dicts in files
		if len(af_weights_dict_pkl.keys()) > 0:
			if not os.path.exists(self.Paths.af_weight_dicts):
				os.makedirs(self.Paths.af_weight_dicts)
			file_name = 'af_weights_'+self.NetSettings.spec_name+'_run_'+str(self.NetSettings.run)+'_mb_'+str(mb_count)
			pickle.dump(af_weights_dict_pkl, open(self.Paths.af_weight_dicts+file_name+'.pkl', 'wb'),protocol=3)
			json.dump(af_weights_dict_json, open(self.Paths.af_weight_dicts+file_name+'.json', 'w'), sort_keys=True, indent=4)
			if print_messages:
				print('[MESSAGE] file saved: %s (af weights)' %(self.Paths.af_weight_dicts+file_name+'.pkl'))
		else:
			if print_messages:
				print('[WARNING] no trainable variables "blend_weights" or "swish_beta" found - no af weights saved.')

	# ========================== AUXILIARY FUNCTIONS ===========================

	def save_all_weights(self, sess, mb_count, print_messages=False):
		if len([v for v in tf.trainable_variables()]) > 0:
			# create dict of all trainable variables in network
			filter_dict = {}
			for name in [v.name for v in tf.trainable_variables()]:
				filter_dict[name] = sess.run(name)
			# save dict in pickle file
			if not os.path.exists(self.Paths.all_weight_dicts):
				os.makedirs(self.Paths.all_weight_dicts)
			filename = 'all_weights_'+self.NetSettings.spec_name+'_run_'+str(self.NetSettings.run)+'_mb_'+str(mb_count)+'.pkl'
			pickle.dump(filter_dict, open(self.Paths.all_weight_dicts+filename,'wb'), protocol=3)
			if print_messages:
				print('[MESSAGE] file saved: %s (all weights)' %(self.Paths.all_weight_dicts+filename))
		else:
			if print_messages:
				print('[WARNING] no trainable variables found - no weights saved.')

	def getLayerShape(self, state, print_shape=False):
		N = int(state.get_shape()[1]) # feature map X
		M = int(state.get_shape()[2]) # feature map Y
		L = int(state.get_shape()[3]) # feature map Z
		if print_shape:
			print("layer shape: ",N,M,L)
		return [N,M,L]

	def print_all_trainable_var_names(self):
		if len([v for v in tf.trainable_variables()]) > 0:
			for name in [v.name for v in tf.trainable_variables()]:
				print(name)

# ##############################################################################
# ### CLASS END ################################################################
# ##############################################################################

def variable_summaries(var, label):
	# Attach a lot of summaries to a Tensor (for TensorBoard visualization)
	assert isinstance(label, str), 'label must be of type str.'
	with tf.name_scope(label):
		mean = tf.reduce_mean(var)
		stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.summary.scalar('mean', mean)
		tf.summary.scalar('stddev', stddev)
		tf.summary.scalar('max', tf.reduce_max(var))
		tf.summary.scalar('min', tf.reduce_min(var))
		tf.summary.histogram('histogram', var)
