"""
Main script. Handling the individual modules of the framework, i.e., managing
the tasks passed on by the supervisor.

@authors: Leon Sütfeld, Flemming Brieger
"""

import tensorflow as tf
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import deepnet_networks as net

# ##############################################################################
# ################################### MAIN #####################################
# ##############################################################################

if __name__ == '__main__':
	# parse input arguments
	parser = argparse.ArgumentParser(description='Run network session.')
	parser.add_argument('-task', metavar='TASK', default=None, type=str, nargs=1, help='chosen task (mandatory) - imagenet, cifar10, cifar100')
	parser.add_argument('-network', metavar='NETWORK', default=None, type=str, nargs=1, help='chosen network (mandatory) - squeezenet, SMCN.')
	parser.add_argument('-mode', metavar='MODE', default=None, type=str, nargs=1, help='mode of the network as str (optional) - training, filters, activations or dreams.')
	parser.add_argument('-n_minibatches', metavar='NMINIBATCHES', default=None, type=int, nargs=1, help='number of minibatches in run (mandatory for training)')
	parser.add_argument('-experiment_name', metavar='EXPERIMENTNAME', default=None, type=str, nargs=1, help='experiment_name as str (mandatory)')
	parser.add_argument('-spec_name', metavar='SPECNAME', default=None, type=str, nargs=1, help='spec_name of the network as str (mandatory)')
	parser.add_argument('-run', metavar='RUN', default=None, type=int, nargs=1, help='run to load as int (mandatory)')
	parser.add_argument('-minibatch_size', metavar='MINIBATCH_SIZE', default=None, type=int, nargs=1, help='minibatch size (mandatory) - recommended 256 or 128.')
	parser.add_argument('-optimizer', metavar='OPTIMIZER', default=None, type=str, nargs=1, help='Optimizer choice (e.g. \'Adam\').')
	parser.add_argument('-dropout_keep_probs', metavar='DROPOUT_KEEP_PROBS', default=None, type=str, nargs=1, help='keep_probs for dropout.')
	parser.add_argument('-dropout_keep_probs_inference', metavar='DROPOUT_KEEP_PROBS_INFERENCE', default=None, type=str, nargs=1, help='keep_probs for dropout during inferenve (usually all 1.0).')
	parser.add_argument('-lr', metavar='LEARNING_RATE', default=None, type=float, nargs=1, help='learning rate.')
	parser.add_argument('-lr_step_ep', metavar='LR_STEP_EP', default=None, type=str, nargs=1, help='epochs with steps in learning rate scheduler.')
	parser.add_argument('-lr_step_multi', metavar='LR_STEP_MULTI', default=None, type=str, nargs=1, help='multiplicative factors applied to lr after corresponding epoch was reached.')
	parser.add_argument('-use_wd', metavar='USE_WD', default=None, type=str, nargs=1, help='boolean, enables weight decay if true.')
	parser.add_argument('-wd_lambda', metavar='WD_LAMBDA', default=None, type=float, nargs=1, help='lambda for weight decay.')
	parser.add_argument('-training_schedule', metavar='TRAINING_SCHEDULE', default=None, type=str, nargs=1, help='\'epochs\' for sampling without replacement, \'random\' for sampling with replacement (mandatory).')
	parser.add_argument('-create_val_set', metavar='CREATE_VAL_SET', default=None, type=str, nargs=1, help='defines if a validation set is to be created (mandatory).')
	parser.add_argument('-val_set_fraction', metavar='VAL_SET_FRACTION', default=None, type=str, nargs=1, help='defines size of the validation set as a fraction of the whole training set (mandatory).')
	parser.add_argument('-af_set', metavar='AF_SET', default=None, type=str, nargs=1, help='defines the AF or set of AFs')
	parser.add_argument('-af_weights_init', metavar='AF_WEIGHTS_INIT', default=None, type=str, nargs=1, help='defines initialization of blend weights and swish beta - \'default\' or \'predefined\'. (mandatory when using blended AFs)')
	parser.add_argument('-blend_trainable', metavar='BLEND_TRAINABLE', default=None, type=str, nargs=1, help='defines if the blend weights should be trainable. (mandatory when using blended AFs)')
	parser.add_argument('-blend_mode', metavar='BLEND_MODE', default=None, type=str, nargs=1, help='mode of the network as str (optional) - training, filters, activations or dreams.')
	parser.add_argument('-swish_beta_trainable', metavar='SWISH_BETA_TRAINABLE', default=None, type=str, nargs=1, help='defines if the beta parameter in a swish AF should be trainable. (mandatory when using swish AF)')
	parser.add_argument('-maxmi_alpha', metavar='ALPHA', default=None, type=float, nargs=1, help='maxmi_alpha > 1 adds in the maximization of translation loss between the squeezenet branches. (madatory)')
	parser.add_argument('-n2n_lambda', metavar='LAMBDA', default=None, type=float, nargs=1, help='n2n_lambda > 1 adds in the minimization of net2net loss between the squeezenet branches. (madatory)')

	args = vars(parser.parse_args())
	# translate boolean args
	if  args['blend_trainable'] is not None:
		assert args['blend_trainable'][0] in ['True','False'], 'blend_trainable must be set to \'True\' or \'False\'.'
		if args['blend_trainable'][0] == 'True':
			args['blend_trainable'][0] = True
		elif args['blend_trainable'][0] == 'False':
			args['blend_trainable'][0] = False
	if  args['swish_beta_trainable'] is not None:
		assert args['swish_beta_trainable'][0] in ['True','False'], 'swish_beta_trainable must be set to \'True\' or \'False\'.'
		if args['swish_beta_trainable'][0] == 'True':
			args['swish_beta_trainable'][0] = True
		elif args['swish_beta_trainable'][0] == 'False':
			args['swish_beta_trainable'][0] = False
	if  args['create_val_set'] is not None:
		assert args['create_val_set'][0] in ['True','False'], 'create_val_set must be set to \'True\' or \'False\'.'
		if args['create_val_set'][0] == 'True':
			args['create_val_set'][0] = True
		elif args['create_val_set'][0] == 'False':
			args['create_val_set'][0] = False
	if  args['use_wd'] is not None:
		assert args['use_wd'][0] in ['True','False'], 'use_wd must be set to \'True\' or \'False\'.'
		if args['use_wd'][0] == 'True':
			args['use_wd'][0] = True
		elif args['use_wd'][0] == 'False':
			args['use_wd'][0] = False

	if args['task'][0] == 'cifar10':
		import deepnet_task_cifar10 as task
		import deepnet_aux_cifar as aux
	elif args['task'][0] == 'cifar100':
		import deepnet_task_cifar100 as task
		import deepnet_aux_cifar as aux

	# training
	if args['mode'][0] in ['train', 'training', '']:
		NetSettings = net.NetSettings(args)
		TaskSettings = task.TaskSettings(args)
		Paths = task.Paths(TaskSettings)
		training_handler = task.TrainingHandler(TaskSettings, Paths)
		validation_handler = task.ValidationHandler(TaskSettings, Paths)
		test_handler = task.TestHandler(TaskSettings, Paths)
		rec = task.PerformanceRecorder(TaskSettings, Paths)
		counter = task.Counter(training_handler)
		timer = task.SessionTimer()
		Network = net.Network(NetSettings, Paths, namescope='Network')
		task.train(TaskSettings, Paths, Network, training_handler, validation_handler, test_handler, counter, timer, rec, args)

	# analysis
	if args['mode'][0] in ['analysis']:
		TaskSettings = task.TaskSettings(args)
		Paths = task.Paths(TaskSettings)
		aux.analysis(TaskSettings, Paths)

	# testing
	if args['mode'][0] in ['test', 'testing']:
		NetSettings = net.NetSettings(args)
		TaskSettings = task.TaskSettings(args)
		Paths = task.Paths(TaskSettings)
		test_handler = task.TestHandler(TaskSettings, Paths)
		timer = task.SessionTimer()
		Network = net.Network(NetSettings, Paths, namescope='Network')
		task.test_saved_model(TaskSettings, Paths, Network, test_handler)
