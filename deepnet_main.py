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

	def str2bool(v):
	    if v.lower() in ('yes', 'true', 't', 'y', '1'):
	        return True
	    elif v.lower() in ('no', 'false', 'n', 'f', '0'):
	        return False
	    else:
	        raise argparse.ArgumentTypeError('Boolean value expected.')

	# parse input arguments
	parser = argparse.ArgumentParser(description='Run network session.')
	parser.add_argument('-task', type=str, help='cifar10, cifar100')
	parser.add_argument('-preprocessing', type=str, help='none, ztrans, gcn_zca')
	parser.add_argument('-network', type=str, help='smcn, allcnnc, ...')
	parser.add_argument('-mode', type=str, help='(optional) training, analysis, test')
	parser.add_argument('-n_minibatches', type=int)
	parser.add_argument('-experiment_name', type=str)
	parser.add_argument('-spec_name', type=str)
	parser.add_argument('-run', type=int)
	parser.add_argument('-minibatch_size', type=int)
	parser.add_argument('-optimizer', type=str)
	parser.add_argument('-dropout_keep_probs', type=float, nargs='*', help='keep_probs for all dropout layers')
	parser.add_argument('-dropout_keep_probs_inference', type=float, nargs='*', help='keep_probs for all dropout layers during inference (usually all 1.0)')
	parser.add_argument('-lr', type=float, help='main or start learning rate')
	parser.add_argument('-lr_step_ep', type=int, nargs='*', help='epochs with steps in learning rate scheduler')
	parser.add_argument('-lr_step_multi', type=float, nargs='*', help='multiplicative factors applied to lr after corresponding step was reached')
	parser.add_argument('-use_wd', type=str2bool, help='weight decay')
	parser.add_argument('-wd_lambda', type=float, help='weight decay lambda')
	parser.add_argument('-training_schedule', type=str, help='\'epochs\' for sampling without replacement, \'random\' for sampling with replacement')
	parser.add_argument('-create_val_set', type=str2bool, help='enables validation')
	parser.add_argument('-val_set_fraction', type=float, help='fraction of training set used for validation')
	parser.add_argument('-af_set', type=str, help='AF / set of AFs to use')
	parser.add_argument('-af_weights_init', type=str, help='defines initialization of blend weights and swish beta - \'default\' or \'predefined\'')
	parser.add_argument('-blend_trainable', type=str2bool, help='enables adaptive blend / AF weights')
	parser.add_argument('-blend_mode',  type=str, help='\'unrestricted\', \'normalized\', \'softmaxed\'')
	parser.add_argument('-swish_beta_trainable', type=str2bool)
	args = vars(parser.parse_args())

	if args['task'] == 'cifar10':
		import deepnet_task_cifar10 as task
		import deepnet_aux_cifar as aux
	elif args['task'] == 'cifar100':
		import deepnet_task_cifar100 as task
		import deepnet_aux_cifar as aux

	# training
	if args['mode'] in ['training', 'train', '']:
		NetSettings = net.NetSettings(args)
		TaskSettings = task.TaskSettings(args)
		Paths = task.Paths(TaskSettings)
		training_handler = task.TrainingHandler(TaskSettings, Paths, args)
		test_handler = task.TestHandler(TaskSettings, Paths, args)
		rec = task.PerformanceRecorder(TaskSettings, Paths)
		counter = task.Counter(training_handler)
		timer = task.SessionTimer()
		Network = net.Network(NetSettings, Paths, namescope='Network')
		task.train(TaskSettings, Paths, Network, training_handler, test_handler, counter, timer, rec, args)

	# analysis
	if args['mode'] in ['analysis']:
		TaskSettings = task.TaskSettings(args)
		Paths = task.Paths(TaskSettings)
		aux.analysis(TaskSettings, Paths)

	# testing
	if args['mode'] in ['test', 'testing']:
		NetSettings = net.NetSettings(args)
		TaskSettings = task.TaskSettings(args)
		Paths = task.Paths(TaskSettings)
		test_handler = task.TestHandler(TaskSettings, Paths)
		timer = task.SessionTimer()
		Network = net.Network(NetSettings, Paths, namescope='Network')
		task.test_saved_model(TaskSettings, Paths, Network, test_handler)
