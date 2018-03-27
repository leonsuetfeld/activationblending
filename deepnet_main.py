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
	parser.add_argument('-task', type=str, nargs=1, help='cifar10, cifar100')
	parser.add_argument('-network', type=str, nargs=1, help='smcn, allcnnc, ...')
	parser.add_argument('-mode', type=str, nargs=1, help='(optional) training, analysis, test')
	parser.add_argument('-n_minibatches', type=int, nargs=1)
	parser.add_argument('-experiment_name', type=str, nargs=1)
	parser.add_argument('-spec_name', type=str, nargs=1)
	parser.add_argument('-run', type=int, nargs=1)
	parser.add_argument('-minibatch_size', type=int, nargs=1)
	parser.add_argument('-optimizer', type=str, nargs=1)
	parser.add_argument('-dropout_keep_probs', nargs='+', help='keep_probs for all dropout layers')
	parser.add_argument('-dropout_keep_probs_inference', nargs='+', help='keep_probs for all dropout layers during inferenve (usually all 1.0)')
	parser.add_argument('-lr', type=float, nargs=1, help='main or start learning rate')
	parser.add_argument('-lr_step_ep', type=str, nargs='*', help='epochs with steps in learning rate scheduler')
	parser.add_argument('-lr_step_multi', type=str, nargs='*', help='multiplicative factors applied to lr after corresponding step was reached')
	parser.add_argument('-use_wd', type=str, nargs=1, help='weight decay')
	parser.add_argument('-wd_lambda', type=float, nargs=1, help='weight decay lambda')
	parser.add_argument('-training_schedule', type=str, nargs=1, help='\'epochs\' for sampling without replacement, \'random\' for sampling with replacement')
	parser.add_argument('-create_val_set', type=str, nargs=1, help='enables validation')
	parser.add_argument('-val_set_fraction', type=str, nargs=1, help='fraction of training set used for validation')
	parser.add_argument('-af_set', type=str, nargs=1, help='AF / set of AFs to use')
	parser.add_argument('-af_weights_init', type=str, nargs=1, help='defines initialization of blend weights and swish beta - \'default\' or \'predefined\'')
	parser.add_argument('-blend_trainable', type=str, nargs=1, help='enables adaptive blend / AF weights')
	parser.add_argument('-blend_mode',  type=str, nargs=1, help='\'unrestricted\', \'normalized\', \'softmaxed\'')
	parser.add_argument('-swish_beta_trainable', type=str, nargs=1)

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
