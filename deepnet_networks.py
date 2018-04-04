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
		assert args['blend_mode'] in ['unrestricted', 'normalized', 'softmaxed'], 'requested setting for blend_mode unknown.'
		assert args['optimizer'] is not None, 'optimizer must be specified.'
		assert args['use_wd'] is not None, 'use_wd must be specified.'
		assert args['wd_lambda'] is not None, 'wd_lambda must be specified.'

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
		self.network_spec = args['network']
		self.std_type = 2 # 1 or 2 (2 == maxout style, goodfellow 2013)
		self.use_wd = args['use_wd']
		self.wd_lambda = args['wd_lambda']
		self.af_set = args['af_set']
		self.af_weights_init = args['af_weights_init']
		self.blend_trainable = args['blend_trainable']
		self.blend_mode = args['blend_mode']
		self.swish_beta_trainable = args['swish_beta_trainable']
		self.optimizer_choice = args['optimizer']
		self.print_overview()

		assert self.std_type in [1,2], 'std_type must be in [1,2].'

	def print_overview(self):
		print('')
		print('###########################################')
		print('### NETWORK SETTINGS OVERVIEW #############')
		print('###########################################')
		print('')
		print(' - input image format: (%i,%i,%i)' %(self.image_x,self.image_y,self.image_z))
		print(' - output dims: %i' %(self.logit_dims))
		if self.mode in ['train','training','']:
			print(' - optimizer: %s' %(self.optimizer_choice))
			print(' - AF set: %s' %(self.af_set))
			print(' - af weights initialization: %s' %(self.af_weights_init))
			print(' - blend weights trainable: %s' %(str(self.blend_trainable)))
			print(' - blend mode: %s' %(self.blend_mode))
			if 'swish' in self.af_set:
				print(' - swish beta trainable: %s' %(str(self.swish_beta_trainable)))

class Network(object):

	# ======================== GENERAL NETWORK DEFINITION ======================

	def __init__(self, NetSettings, Paths, namescope=None, reuse=False):

		# --- settings & paths -------------------------------------------------
		self.NetSettings = NetSettings
		self.Paths = Paths
		# --- input  --------------------------------------------------
		self.X = tf.placeholder(tf.float32, [NetSettings.minibatch_size, 3072], name='images')
		self.Y = tf.placeholder(tf.int64, [None], name='labels')
		self.s = tf.placeholder(tf.float32, [NetSettings.image_x, NetSettings.image_y, NetSettings.image_z], name='dataset_perpixel_std')
		self.lr = tf.placeholder(tf.float32, name='learning_rate')
		self.dropout_keep_prob = tf.placeholder(tf.float32, [None], name='dropout_keep_prob')
		self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
		self.reuse = reuse
		# --- cifar10 reshape --------------------------------------------------
		self.Xp = tf.reshape(self.X, [NetSettings.minibatch_size, NetSettings.image_z, NetSettings.image_x, NetSettings.image_y]) # [256,3,32,32]
		self.Xp = tf.transpose(self.Xp, [0,2,3,1]) # => [256,32,32,3]
		# --- per image standardization ----------------------------------------
		if NetSettings.std_type == 1:
			self.Xp_per_image_mean = tf.reduce_mean(self.Xp, [1,2,3], name='image_mean', keep_dims=True)
			_,self.Xp_per_image_std = tf.nn.moments(self.Xp, [1,2,3], name='image_std', keep_dims=True)
			self.Xp = tf.divide((self.Xp-self.Xp_per_image_mean),tf.maximum( self.Xp_per_image_std, tf.divide(tf.fill(self.Xp_per_image_std.get_shape(),1.),tf.sqrt(3072.)) ))
		elif NetSettings.std_type == 2:
			self.Xp = self.Xp-tf.reduce_mean(self.Xp, [1,2,3], name='image_mean', keep_dims=True)
			self.Xp = tf.divide(self.s * self.Xp, tf.maximum(tf.reduce_mean(tf.square(self.Xp))+self.std_lambda, tf.constant(10**(-8))))

		# CHOOSE NETWORK ARCHITECTURE
		if self.NetSettings.network_spec == 'sfnet':
			self.logits = self.sfnet(namescope='sfnet')
		elif self.NetSettings.network_spec == 'smcn':
			self.logits = self.smcn(namescope='smcn')
		elif self.NetSettings.network_spec == 'smcnLin':
			self.logits = self.smcnLin(namescope='smcnLin')
		elif self.NetSettings.network_spec == 'squeezenet':
			self.logits = self.squeezenet(namescope='squeezenet')
		elif self.NetSettings.network_spec == 'densenet':
			self.logits = self.densenet(namescope='densenet')
		elif self.NetSettings.network_spec == 'allcnnc':
			self.logits = self.allcnnc(namescope='allcnnc')
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
			# input block
			self.state = self.conv2d_parallel_act_layer(layer_input=self.Xp, W_shape=[5,5,3,64], strides=[1,1,1,1], padding='SAME', bias_init=0.0, AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/conv1')
			self.state = tf.nn.dropout(self.state, keep_prob=self.dropout_keep_prob[0], name=namescope+'/dropout1')
			self.state = self.conv2d_parallel_act_layer(layer_input=self.state, W_shape=[3,3,64,64], strides=[1,1,1,1], padding='SAME', AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/conv2')
			# pooling layer
			self.state = tf.nn.avg_pool(self.state, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=namescope+'/avgpool2')
			# standard block
			self.state = self.conv2d_parallel_act_layer(layer_input=self.state, W_shape=[1,1,64,64], strides=[1,1,1,1], padding='SAME', AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/conv3')
			self.state = tf.nn.dropout(self.state, keep_prob=self.dropout_keep_prob[0], name=namescope+'/dropout3')
			self.state = self.conv2d_parallel_act_layer(layer_input=self.state, W_shape=[5,5,64,64], strides=[1,1,1,1], padding='SAME', AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/conv4')
			# pooling layer
			self.state = tf.nn.avg_pool(self.state, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=namescope+'/avgpool4')
			# output block
			self.state = self.dense_parallel_act_layer(layer_input=self.state, W_shape=[384], AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, reuse=self.reuse, swish_beta_trainable=self.NetSettings.swish_beta_trainable, varscope=namescope+'/dense5')
			self.state = tf.nn.dropout(self.state, keep_prob=self.dropout_keep_prob[0])
			self.state = self.dense_parallel_act_layer(layer_input=self.state, W_shape=[192], AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, reuse=self.reuse, swish_beta_trainable=self.NetSettings.swish_beta_trainable, varscope=namescope+'/dense6')
			# output layer
			self.logits = self.dense_parallel_act_layer(layer_input=self.state, W_shape=[self.NetSettings.logit_dims], AF_set=None, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, reuse=self.reuse, swish_beta_trainable=self.NetSettings.swish_beta_trainable, varscope=namescope+'/denseout')
			return self.logits

	# Sonja's convnet
	def sfnet(self, namescope=None):
		with tf.name_scope(namescope):
			self.state = self.conv2d_parallel_act_layer(layer_input=self.Xp, W_shape=[5,5,3,64], strides=[1,1,1,1], bias_init=0.0, AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/conv1')
			self.state = tf.nn.max_pool(self.state, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name=namescope+'/pool1')
			self.state = tf.nn.local_response_normalization(self.state, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75, name=namescope+'/norm1')
			self.state = self.conv2d_parallel_act_layer(layer_input=self.state, W_shape=[5,5,64,64], strides=[1,1,1,1], AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/conv2')
			self.state = tf.nn.local_response_normalization(self.state, depth_radius=4, bias=1.0, alpha=0.0001/9.0, beta=0.75, name=namescope+'/norm2')
			self.state = tf.nn.max_pool(self.state, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name=namescope+'/pool2')
			self.state = self.conv2d_parallel_act_layer(layer_input=self.state, W_shape=[5,5,64,64], strides=[1,1,1,1], AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/conv3')
			self.state = tf.nn.local_response_normalization(self.state, depth_radius=4, bias=1.0, alpha=0.0001/9.0, beta=0.75, name=namescope+'/norm3')
			self.state = self.conv2d_parallel_act_layer(layer_input=self.state, W_shape=[5,5,64,64], strides=[1,1,1,1], AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/conv4')
			self.state = tf.nn.local_response_normalization(self.state, depth_radius=4, bias=1.0, alpha=0.0001/9.0, beta=0.75, name=namescope+'/norm4')
			self.state = self.dense_parallel_act_layer(layer_input=self.state, W_shape=[384], AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/dense5')
			self.state = self.dense_parallel_act_layer(layer_input=self.state, W_shape=[192], AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/dense6')
			self.logits = self.dense_parallel_act_layer(layer_input=self.state, W_shape=[self.NetSettings.logit_dims], bias_init=0.0, AF_set=None, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/dense7')
			return self.logits

	# standard squeezenet (adjusted for cifar10)
	def squeezenet(self, namescope=None):
		# commented out some max-poolings for cifar10, adjust state_3 & state_7 function call when commented in again
		with tf.name_scope(namescope):
			self.state = self.conv2d_parallel_act_layer(layer_input=self.Xp, W_shape=[7,7,3,96], strides=[1,2,2,1], padding='SAME', AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/conv1')
			# self.state = tf.nn.max_pool(self.state, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name=namescope+'/maxpool1')
			self.state = self.fire_parallel_act_module(self.state, 96, 16, 128, reuse=self.reuse, varscope=namescope+'/fire2')
			self.state = self.fire_parallel_act_module(self.state, 128, 16, 128, reuse=self.reuse, varscope=namescope+'/fire3')
			self.state = self.fire_parallel_act_module(self.state, 128, 32, 256, reuse=self.reuse, varscope=namescope+'/fire4')
			# self.state = tf.nn.max_pool(self.state, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name=namescope+'/maxpool4')
			self.state = self.fire_parallel_act_module(self.state, 256, 32, 256, reuse=self.reuse, varscope=namescope+'/fire5')
			self.state = self.fire_parallel_act_module(self.state, 256, 48, 384, reuse=self.reuse, varscope=namescope+'/fire6')
			self.state = self.fire_parallel_act_module(self.state, 384, 48, 384, reuse=self.reuse, varscope=namescope+'/fire7')
			self.state = self.fire_parallel_act_module(self.state, 384, 64, 512, reuse=self.reuse, varscope=namescope+'/fire8')
			self.state = tf.nn.max_pool(self.state, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name=namescope+'/maxpool8')
			self.state = self.fire_parallel_act_module(self.state, 512, 64, 512, reuse=self.reuse, varscope=namescope+'/fire9')
			self.state = tf.nn.dropout(self.state, self.dropout_keep_prob[0], name=namescope+'/dropout9')
			self.state = self.conv2d_parallel_act_layer(layer_input=self.state, W_shape=[1,1,512,self.NetSettings.logit_dims], padding='SAME', AF_set=None, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/conv10')
			self.state = tf.nn.avg_pool(self.state, ksize=[1,7,7,1], strides=[1,1,1,1], padding='VALID', name=namescope+'/avgpool10')
			self.logits = tf.squeeze(self.state, [1,2], name=namescope+'/logits')
			return self.logits

	# densenet architecture (adjusted for cifar10)
	def densenet(self, namescope=None):
		with tf.name_scope(namescope):
			# commented out some max-poolings for cifar10, adjust state_3 function call and global pooling size when commented in again
			self.state = self.conv2d_parallel_act_layer(layer_input=self.Xp, W_shape=[7,7,3,32], strides=[1,2,2,1], padding='SAME', AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/conv1')
			# self.state = tf.nn.max_pool(self.state, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name=namescope+'/maxpool1')
			self.state = self.parallel_act_dense_block(self.state, 32, varscope=namescope+'/denseblock3')
			self.state = self.conv2d_parallel_act_layer(layer_input=self.state, W_shape=[1,1,92,32], strides=[1,1,1,1], padding='SAME', AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/conv3')
			self.state = tf.nn.avg_pool(self.state, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name=namescope+'/avgpool4')
			self.state = self.parallel_act_dense_block(self.state, 32, varscope=namescope+'/denseblock6')
			self.state = self.conv2d_parallel_act_layer(layer_input=self.state, W_shape=[1,1,92,32], strides=[1,1,1,1], padding='SAME', AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/conv7')
			self.state = tf.nn.avg_pool(self.state, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name=namescope+'/avgpool7')
			self.state = self.parallel_act_dense_block(self.state, 32, varscope=namescope+'/denseblock9')
			self.state = tf.nn.avg_pool(self.state, ksize=[1,4,4,1], strides=[1,1,1,1], padding='VALID', name=namescope+'/avgpool10')
			self.logits = self.dense_parallel_act_layer(layer_input=self.state, W_shape=[self.NetSettings.logit_dims], AF_set=None, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/dense12')
			return self.logits

	def allcnnc(self, namescope=None):
		with tf.name_scope(namescope):
			# input
			self.state = tf.nn.dropout(self.Xp, keep_prob=self.dropout_keep_prob[0], name=namescope+'/dropout1')
			# block 1
			self.state = self.conv2d_parallel_act_layer(layer_input=self.state, W_shape=[3,3,3,96], strides=[1,1,1,1], padding='SAME', bias_init=0.0, AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/conv1')
			self.state = self.conv2d_parallel_act_layer(layer_input=self.state, W_shape=[3,3,96,96], strides=[1,1,1,1], padding='SAME', AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/conv2')
			self.state = self.conv2d_parallel_act_layer(layer_input=self.state, W_shape=[3,3,96,96], strides=[1,2,2,1], padding='SAME', AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/conv3')
			self.state = tf.nn.dropout(self.state, keep_prob=self.dropout_keep_prob[1], name=namescope+'/dropout2')
			# block 2
			self.state = self.conv2d_parallel_act_layer(layer_input=self.state, W_shape=[3,3,96,192], strides=[1,1,1,1], padding='SAME', AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/conv4')
			self.state = self.conv2d_parallel_act_layer(layer_input=self.state, W_shape=[3,3,192,192], strides=[1,1,1,1], padding='SAME', AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/conv5')
			self.state = self.conv2d_parallel_act_layer(layer_input=self.state, W_shape=[3,3,192,192], strides=[1,2,2,1], padding='SAME', AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/conv6')
			self.state = tf.nn.dropout(self.state, keep_prob=self.dropout_keep_prob[2], name=namescope+'/dropout3')
			# block 3
			self.state = self.conv2d_parallel_act_layer(layer_input=self.state, W_shape=[3,3,192,192], strides=[1,1,1,1], padding='VALID', AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/conv7')
			self.state = self.conv2d_parallel_act_layer(layer_input=self.state, W_shape=[1,1,192,192], strides=[1,1,1,1], padding='SAME', AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/conv8')
			self.state = self.conv2d_parallel_act_layer(layer_input=self.state, W_shape=[1,1,192,self.NetSettings.logit_dims], strides=[1,1,1,1], padding='SAME', AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope=namescope+'/conv9')
			# output
			self.state = tf.squeeze(tf.nn.avg_pool(self.state, ksize=[1,6,6,1], strides=[1,1,1,1], padding='VALID', name=namescope+'/avgpool10'))
			self.logits = self.dense_parallel_act_layer(layer_input=self.state, W_shape=[self.NetSettings.logit_dims], AF_set=None, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=False, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=False, reuse=self.reuse, varscope=namescope+'/dense10')
			return self.logits

	# ============================= NETWORK BLOCKS =============================

	def parallel_act_dense_block(self, X, input_depth, k=12, varscope=None):
		with tf.variable_scope(varscope):
			conv_1 = self.conv2d_parallel_act_layer(layer_input=X, W_shape=[1,1,input_depth,k], AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope='conv_1')
			concat = tf.concat([X,conv_1],3)
			conv_2 = self.conv2d_parallel_act_layer(layer_input=concat, W_shape=[1,1,input_depth+1*k,k], AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope='conv_2')
			concat = tf.concat([concat,conv_2],3)
			conv_3 = self.conv2d_parallel_act_layer(layer_input=concat, W_shape=[1,1,input_depth+2*k,k], AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope='conv_3')
			concat = tf.concat([concat,conv_3],3)
			conv_4 = self.conv2d_parallel_act_layer(layer_input=concat, W_shape=[1,1,input_depth+3*k,k], AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope='conv_4')
			concat = tf.concat([concat,conv_4],3)
			conv_5 = self.conv2d_parallel_act_layer(layer_input=concat, W_shape=[1,1,input_depth+4*k,k], AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=self.reuse, varscope='conv_5')
			concat = tf.concat([concat,conv_5],3)
			return concat

	def fire_parallel_act_module(self, X, input_depth, squeeze_depth, expand_depth, reuse=False, varscope=None):
		with tf.variable_scope(varscope):
			s1x1 = self.conv2d_parallel_act_layer(layer_input=X, W_shape=[1,1,input_depth,squeeze_depth], AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=reuse, varscope='s1x1')
			e1x1 = self.conv2d_parallel_act_layer(layer_input=s1x1, W_shape=[1,1,squeeze_depth,int(np.floor(expand_depth/2.))], AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=reuse, varscope='e1x1')
			e3x3 = self.conv2d_parallel_act_layer(layer_input=s1x1, W_shape=[3,3,squeeze_depth,int(np.ceil(expand_depth/2.))], AF_set=self.NetSettings.af_set, af_weights_init=self.NetSettings.af_weights_init, W_blend_trainable=self.NetSettings.blend_trainable, AF_blend_mode=self.NetSettings.blend_mode, swish_beta_trainable=self.NetSettings.swish_beta_trainable, reuse=reuse, varscope='e3x3')
			fire_out = tf.concat([e1x1,e3x3],3)
		return fire_out

	# ============================= NETWORK LAYERS =============================

	def dense_parallel_act_layer(self, layer_input, W_shape, b_shape=[-1], bias_init=0.1, AF_set=None, af_weights_init='default', W_blend_trainable=True, AF_blend_mode='unrestricted', swish_beta_trainable=True, reuse=False, varscope=None):
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
			return self.activate(iState, AF_set, af_weights_init, W_blend_trainable, AF_blend_mode, swish_beta_trainable, varscope)

	def conv2d_parallel_act_layer(self, layer_input, W_shape, b_shape=[-1], strides=[1,1,1,1], padding='SAME', bias_init=0.1, AF_set=None, af_weights_init='default', W_blend_trainable=True, AF_blend_mode='unrestricted', swish_beta_trainable=True, reuse=False, varscope=None):
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
			return self.activate(iState, AF_set, af_weights_init, W_blend_trainable, AF_blend_mode, swish_beta_trainable, varscope)

	def dense_layer(layer_input, W_shape, b_shape=[-1], bias_init=0.1, activation=tf.nn.relu, reuse=False, varscope=None):
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
			if not (activation is None):
				iState = activation(iState)
			return iState

	def conv2d_layer(layer_input, W_shape, b_shape=[-1], strides=[1,1,1,1], padding='SAME', bias_init=0.1, activation=tf.nn.relu, reuse=False, varscope=None):
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
			if not (activation is None):
				iState = activation(iState)
			return iState

	# ============================== ACTIVATIONS ===============================

	def activate(self, preact, AF_set, af_weights_init, W_blend_trainable, AF_blend_mode, swish_beta_trainable, layer_name):
		assert af_weights_init in ['default','predefined'], 'specified initialization type for W_blend unknown.'
		assert AF_blend_mode in ['unrestricted','normalized','softmaxed'], 'specified blend mode unknown.'

		# INITIALIZE BLEND WEIGHTS
		if af_weights_init == 'default':
			n_AFs = int(AF_set.split('_')[0])
			af_weights_initializer = tf.fill([n_AFs], 1/float(n_AFs))
		elif af_weights_init == 'predefined':
			try:
				af_weights_initializer = self.get_predefined_af_weights(layer_name, w_type='blend')
			except Exception:
				raise IOError('\n[ERROR] Couldn\'t restore predefined blend weights, stopping run.\n')
		blend_weights = tf.get_variable("blend_weights", trainable=W_blend_trainable, initializer=af_weights_initializer)

		# NORMALIZE BLEND WEIGHTS
		if AF_blend_mode == 'unrestricted':
			pass
		elif AF_blend_mode == 'normalized':
			blend_weights /= tf.reduce_sum(blend_weights)
		elif AF_blend_mode == 'softmaxed':
			blend_weights = tf.nn.softmax(blend_weights)

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
			act_swish = preact * tf.nn.sigmoid(swish_beta*preact)

		# ACTIVATIONS (standard)
		act_relu = tf.nn.relu(preact)
		act_selu = tf.nn.selu(preact)
		act_elu = tf.nn.elu(preact)
		act_softplus = tf.nn.softplus(preact)
		act_softsign = tf.nn.softsign(preact)
		act_tanh = tf.nn.tanh(preact)
		act_sigmoid = tf.nn.sigmoid(preact)
		act_linu = preact

		# scaling factors to archieve convergence to var = 1 over layers (see phase plot figure in SELU paper)
		if "scaled" in AF_set:
			scale_relu = 1.700
			scale_selu = 1.000
			scale_elu = 1.270
			scale_softplus = 1.880
			scale_softsign = 2.350
			scale_tanh = 1.593
			scale_sigmoid = 4.840
			scale_swish = 1.750
			scale_linu = 1.000
		else:
			scale_relu = 1.
			scale_selu = 1.
			scale_elu = 1.
			scale_softplus = 1.
			scale_softsign = 1.
			scale_tanh = 1.
			scale_sigmoid = 1.
			scale_swish = 1.
			scale_linu = 1.

		# AF sets
		if AF_set.startswith('1_'):
			if 'relu' in AF_set: return blend_weights[0] * scale_relu * act_relu
			if 'selu' in AF_set: return blend_weights[0] * scale_selu * act_selu
			if 'jelu' in AF_set: return blend_weights[0] * scale_elu * act_elu
			if 'softplus' in AF_set: return blend_weights[0] * scale_softplus * act_softplus
			if 'softsign' in AF_set: return blend_weights[0] * scale_softsign * act_softsign
			if 'tanh' in AF_set: return blend_weights[0] * scale_tanh * act_tanh
			if 'sigmoid' in AF_set: return blend_weights[0] * scale_sigmoid * act_sigmoid
			if 'jswish' in AF_set: return blend_weights[0] * scale_swish * act_swish
			if 'linu' in AF_set: return blend_weights[0] * scale_linu * act_linu

		if 'blend2' in AF_set:
			return tf.add_n([blend_weights[0] * scale_relu * act_relu,
							 blend_weights[1] * scale_tanh * act_tanh])

		if 'blend3' in AF_set:
			return tf.add_n([blend_weights[0] * scale_elu * act_elu,
							 blend_weights[1] * scale_tanh * act_tanh,
							 blend_weights[2] * scale_linu * act_linu])

		if 'blend5_swish' in AF_set:
			return tf.add_n([blend_weights[0] * scale_relu * act_relu,
							 blend_weights[1] * scale_elu * act_elu,
							 blend_weights[2] * scale_tanh * act_tanh,
							 blend_weights[3] * scale_swish * act_swish,
							 blend_weights[4] * scale_linu * act_linu])

		if 'blendSF7' in AF_set:
			return tf.add_n([blend_weights[0] * scale_relu * act_relu,
							 blend_weights[1] * scale_selu * act_selu,
							 blend_weights[2] * scale_elu * act_elu,
							 blend_weights[3] * scale_softplus * act_softplus,
							 blend_weights[4] * scale_softsign * act_softsign,
							 blend_weights[5] * scale_tanh * act_tanh,
							 blend_weights[6] * scale_sigmoid * act_sigmoid])

		if 'blend9_swish' in AF_set:
			return tf.add_n([blend_weights[0] * scale_relu * act_relu,
							 blend_weights[1] * scale_selu * act_selu,
							 blend_weights[2] * scale_elu * act_elu,
							 blend_weights[3] * scale_softplus * act_softplus,
							 blend_weights[4] * scale_softsign * act_softsign,
							 blend_weights[5] * scale_tanh * act_tanh,
							 blend_weights[6] * scale_sigmoid * act_sigmoid,
							 blend_weights[7] * scale_swish * act_swish,
							 blend_weights[8] * scale_linu * act_linu])

		# STEP BY STEP: 2 AFs
		if 'blendstep1_1' in AF_set:
			return tf.add_n([blend_weights[0] * scale_selu * act_selu,
							 blend_weights[1] * scale_relu * act_relu])
		if 'blendstep1_2' in AF_set:
			return tf.add_n([blend_weights[0] * scale_selu * act_selu,
							 blend_weights[1] * scale_elu * act_elu])
		if 'blendstep1_3' in AF_set:
			return tf.add_n([blend_weights[0] * scale_selu * act_selu,
							 blend_weights[1] * scale_softplus * act_softplus])
		if 'blendstep1_4' in AF_set:
			return tf.add_n([blend_weights[0] * scale_selu * act_selu,
							 blend_weights[1] * scale_softsign * act_softsign])
		if 'blendstep1_5' in AF_set:
			return tf.add_n([blend_weights[0] * scale_selu * act_selu,
							 blend_weights[1] * scale_tanh * act_tanh])
		if 'blendstep1_6' in AF_set:
			return tf.add_n([blend_weights[0] * scale_selu * act_selu,
							 blend_weights[1] * scale_sigmoid * act_sigmoid])
		if 'blendstep1_7_swish' in AF_set:
			return tf.add_n([blend_weights[0] * scale_selu * act_selu,
							 blend_weights[1] * scale_swish * act_swish])
		if 'blendstep1_8' in AF_set:
			return tf.add_n([blend_weights[0] * scale_selu * act_selu,
							 blend_weights[1] * scale_linu * act_linu])

		# STEP BY STEP: 3 AFs
		if 'blendstep2_1' in AF_set:
			return tf.add_n([blend_weights[0] * scale_selu * act_selu,
							 blend_weights[1] * scale_elu * act_elu,
							 blend_weights[2] * scale_relu * act_relu])
		if 'blendstep2_2' in AF_set:
			return tf.add_n([blend_weights[0] * scale_selu * act_selu,
							 blend_weights[1] * scale_elu * act_elu,
							 blend_weights[2] * scale_softplus * act_softplus])
		if 'blendstep2_3' in AF_set:
			return tf.add_n([blend_weights[0] * scale_selu * act_selu,
							 blend_weights[1] * scale_elu * act_elu,
							 blend_weights[2] * scale_softsign * act_softsign])
		if 'blendstep2_4' in AF_set:
			return tf.add_n([blend_weights[0] * scale_selu * act_selu,
							 blend_weights[1] * scale_elu * act_elu,
							 blend_weights[2] * scale_tanh * act_tanh])
		if 'blendstep2_5' in AF_set:
			return tf.add_n([blend_weights[0] * scale_selu * act_selu,
							 blend_weights[1] * scale_elu * act_elu,
							 blend_weights[2] * scale_sigmoid * act_sigmoid])
		if 'blendstep2_6_swish' in AF_set:
			return tf.add_n([blend_weights[0] * scale_selu * act_selu,
							 blend_weights[1] * scale_elu * act_elu,
							 blend_weights[2] * scale_swish * act_swish])
		if 'blendstep2_7' in AF_set:
			return tf.add_n([blend_weights[0] * scale_selu * act_selu,
							 blend_weights[1] * scale_elu * act_elu,
							 blend_weights[2] * scale_linu * act_linu])

	def get_predefined_af_weights(self, layer_name, w_type='blend'):
		# define which file to load the blend weights from
		if len([f for f in os.listdir(self.Paths.af_weights) if 'priority' in f]) > 0:
			blend_weights_file = [f for f in os.listdir(self.Paths.af_weights) if 'priority' in f and '.pkl' in f][0]
		else:
			blend_weights_file = [f for f in os.listdir(self.Paths.af_weights) if '.pkl' in f][0]
		# load from file
		af_weights_dict = pickle.load(open(self.Paths.af_weights+blend_weights_file,'rb'))
		print('[MESSAGE] (predefined) blend weights loaded from file "%s"' %(self.Paths.af_weights+blend_weights_file))
		if w_type == 'blend':
			return tf.convert_to_tensor(blend_weights_dict[layer_name+'/blend_weights:0'])
		if w_type == 'swish_beta':
			return tf.convert_to_tensor(blend_weights_dict[layer_name+'/swish_beta:0'])

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

	def save_af_weights(self, sess, counter, print_messages=False):
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
			if not os.path.exists(self.Paths.af_weights):
				os.makedirs(self.Paths.af_weights)
			file_name = 'af_weights_'+self.NetSettings.spec_name+'_run_'+str(self.NetSettings.run)+'_mb_'+str(counter.mb_count_total)
			pickle.dump(af_weights_dict_pkl, open(self.Paths.af_weights+file_name+'.pkl', 'wb'),protocol=3)
			json.dump(af_weights_dict_json, open(self.Paths.af_weights+file_name+'.json', 'w'), sort_keys=True, indent=4)
			if print_messages:
				print('[MESSAGE] file saved: %s (af weights)' %(self.Paths.af_weights+file_name+'.pkl'))
		else:
			if print_messages:
				print('[WARNING] no trainable variables "blend_weights" or "swish_beta" found - no af weights saved.')

	# ========================== AUXILIARY FUNCTIONS ===========================

	def save_all_weights(self, sess, counter, print_messages=False):
		if len([v for v in tf.trainable_variables()]) > 0:
			# create dict of all trainable variables in network
			filter_dict = {}
			for name in [v.name for v in tf.trainable_variables()]:
				filter_dict[name] = sess.run(name)
			# save dict in pickle file
			if not os.path.exists(self.Paths.weights):
				os.makedirs(self.Paths.weights)
			filename = 'all_weights_'+str(counter.mb_count_total).zfill(3)+'.pkl'
			pickle.dump(filter_dict, open(self.Paths.weights+filename,'wb'), protocol=3)
			if print_messages:
				print('[MESSAGE] file saved: %s (all weights)' %(self.Paths.weights+filename))
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

def lr_linear_decay(step, start_lr=0.001, stop_lr=0.00004, total_steps=100000): # default squeezenet: start_lr=0.04, stop_lr=0.00004, total_steps=100000
	if step < 0:
		return 0.
	if step < total_steps:
		return np.linspace(start_lr, stop_lr, num=total_steps)[step]
	return stop_lr
