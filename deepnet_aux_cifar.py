import os
import os.path
import numpy as np
import pickle
import time
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
import json
import csv
import sys
import argparse
import subprocess
from sklearn.utils import shuffle

def analysis(TaskSettings, Paths, make_plot=True, make_hrtf=True):
	experiment_name = Paths.exp_folder.split('/')[-2]
	# get spec spec names from their respective folder names
	spec_list = [f for f in os.listdir(Paths.exp_folder) if (os.path.isdir(os.path.join(Paths.exp_folder,f)) and not f.startswith('_'))]
	# put the pieces together and call spec_analysis() for each spec of the experiment
	spec_name_list = []
	n_runs_list = []
	mb_list = []
	best_run_per_spec = []
	mean_run_per_spec = []
	worst_run_per_spec = []
	std_per_spec = []
	t_min_per_spec = []
	t_max_per_spec = []
	t_mean_per_spec = []
	t_var_per_spec = []
	v_min_per_spec = []
	v_max_per_spec = []
	v_mean_per_spec = []
	v_var_per_spec = []
	test_min_per_spec = []
	test_max_per_spec = []
	test_mean_per_spec = []
	test_var_per_spec = []
	print('')
	print('=================================================================================================================================================================================================')
	for spec_name in spec_list:
		path = Paths.exp_folder+spec_name+'/'+Paths.performance_sub
		spec_perf_dict = spec_analysis(TaskSettings, Paths, spec_name=spec_name, perf_files_path=path, make_plot=True)
		# general info about spec
		if spec_perf_dict:
			spec_name_list.append(spec_perf_dict['spec_name'])
			n_runs_list.append(spec_perf_dict['n_runs'])
			# full runs for plotting
			mb_list.append(spec_perf_dict['v_mb'])
			best_run_per_spec.append(spec_perf_dict['v_top1_himax_run'])
			mean_run_per_spec.append(spec_perf_dict['v_top1_mean_run'])
			worst_run_per_spec.append(spec_perf_dict['v_top1_lomax_run'])
			std_per_spec.append(spec_perf_dict['v_top1_std_run'])
			# performance info for text output
			t_min_per_spec.append(spec_perf_dict['t_top1_run_max_min'])
			t_max_per_spec.append(spec_perf_dict['t_top1_run_max_max'])
			t_mean_per_spec.append(spec_perf_dict['t_top1_run_max_mean'])
			t_var_per_spec.append(spec_perf_dict['t_top1_run_max_var'])
			v_min_per_spec.append(spec_perf_dict['v_top1_run_max_min'])
			v_max_per_spec.append(spec_perf_dict['v_top1_run_max_max'])
			v_mean_per_spec.append(spec_perf_dict['v_top1_run_max_mean'])
			v_var_per_spec.append(spec_perf_dict['v_top1_run_max_var'])
			test_min_per_spec.append(spec_perf_dict['test_top1_min'])
			test_max_per_spec.append(spec_perf_dict['test_top1_max'])
			test_mean_per_spec.append(spec_perf_dict['test_top1_mean'])
			test_var_per_spec.append(spec_perf_dict['test_top1_var'])
	print('=================================================================================================================================================================================================')

	# BIG FINAL PLOT
	if make_plot and len(mb_list[0])>0:
		n_mb_total = int(np.max(mb_list[0]))
		fig = plt.figure(figsize=(10,10))
		ax = fig.add_subplot(1,1,1)
		cmap = matplotlib.cm.get_cmap('nipy_spectral')#('Dark2') ('nipy_spectral')
		color_list = ['black','blue','green','red']
		# layer 1
		for spec in range(len(spec_list)):
			# ax.plot(mb_list[spec], best_run_per_spec[spec], linewidth=1.5, color=cmap(spec/len(spec_list)), alpha=0.15)
			ax.plot(np.array([0,300]), np.array([np.max(mean_run_per_spec[spec]),np.max(mean_run_per_spec[spec])]), linewidth=1.5, linestyle='-', color=cmap(spec/len(spec_list)), alpha=0.8)
		# layer 2
		for spec in range(len(spec_list)):
			ax.plot(mb_list[spec], mean_run_per_spec[spec], linewidth=2.0, color=cmap(spec/len(spec_list)), label='[%i / m %.4f / v %.6f] %s' %(n_runs_list[spec], 100*test_mean_per_spec[spec], 100*test_var_per_spec[spec], spec_list[spec]), alpha=0.8)
		# settings
		ax.set_ylim(0.1,1.)
		ax.set_xlim(0.,float(n_mb_total))
		ax.set_xticks(np.arange(0, float(n_mb_total)+.1, float(n_mb_total)/10.))
		ax.set_xticks(np.arange(0, float(n_mb_total)+.1, float(n_mb_total)/40.), minor=True)
		ax.set_yticks(np.arange(0.1, 1.01, .1))
		ax.set_yticks(np.arange(0.1, 1.01, .05), minor=True)
		ax.grid(which='major', linewidth=0.5, linestyle='dotted', color='black', alpha=0.3)
		ax.grid(which='minor', linewidth=0.5, linestyle='dotted', color='black', alpha=0.3)
		ax.set_aspect(float(n_mb_total)+.1/0.901)
		ax.set_title('performance analysis (validation accuracy): %s' %(experiment_name))
		ax.legend(loc=4)
		plt.tight_layout()
		# save
		savepath = Paths.analysis
		plot_filename = 'PA_'+str(experiment_name)+'.png'
		if not os.path.exists(savepath):
			os.makedirs(savepath)
		plt.savefig(savepath+plot_filename, dpi = 120, transparent=False, bbox_inches='tight')
		print('[MESSAGE] file saved: %s (performance analysis plot for experiment "%s")' %(savepath+plot_filename, experiment_name))

	# HUMAN READABLE TEXT FILE:
	if make_hrtf:
		savepath = Paths.analysis
		hrtf_filename = 'main_performance_analysis.csv'
		with open(savepath+hrtf_filename, 'w') as csvfile:
			writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
			writer.writerow(['spec_name','n_runs',
							 't_top1_mean','t_top1_var','t_top1_max','t_top1_min',
							 'v_top1_mean','v_top1_var','v_top1_max','v_top1_min',
							 'test_top1_mean','test_top1_var','test_top1_max','test_top1_min'])
			for spec in range(len(spec_name_list)):
				writer.writerow([spec_name_list[spec], n_runs_list[spec],
								t_mean_per_spec[spec], t_var_per_spec[spec], t_max_per_spec[spec], t_min_per_spec[spec],
								v_mean_per_spec[spec], v_var_per_spec[spec], v_max_per_spec[spec], v_min_per_spec[spec],
								test_mean_per_spec[spec], test_var_per_spec[spec], test_max_per_spec[spec], test_min_per_spec[spec]])
		print('[MESSAGE] file saved: %s (performance analysis csv for experiment "%s")' %(savepath+hrtf_filename, experiment_name))

def spec_analysis(TaskSettings, Paths, spec_name=None, perf_files_path=None, axis_2='af_weights', make_plot=False, return_loss=False):
	assert (axis_2 in ['af_weights', 'loss']) or axis_2 is None, 'axis_2 needs to be defined as None, \'af_weights\', or \'loss\'.'
	analysis_savepath = Paths.analysis
	if spec_name is None:
		spec_name = TaskSettings.spec_name

	# get list of all performance files (pickle dicts) within a spec
	if perf_files_path == None:
		perf_files_path = Paths.performance
	perf_files_list = [f for f in os.listdir(perf_files_path) if (os.path.isfile(os.path.join(perf_files_path, f)) and ('.pkl' in f) and not ('test_performance_' in f))]

	# prepare extraction from dicts
	afw_hist_store = []
	train_mb_n_hist_store = []
	train_top1_hist_store = []
	train_loss_hist_store = []
	val_mb_n_hist_store = []
	val_top1_hist_store = []
	val_loss_hist_store = []
	test_top1_store = []
	test_loss_store = []
	run_number_store = []

	# criteria for exclusion: incomplete runs. this section makes a list of all completed runs
	n_val_samples_list = []
	for run_file in perf_files_list:
		p_dict = pickle.load( open( perf_files_path+run_file, "rb" ) )
		run_number = int(run_file.split('run_')[1].split('.')[0])
		n_val_samples = len(p_dict['val_mb_n_hist'])
		n_val_samples_list.append(n_val_samples)
	run_length = np.max(n_val_samples_list)
	complete_runs = []
	for run_file in perf_files_list:
		p_dict = pickle.load( open( perf_files_path+run_file, "rb" ) )
		run_number = int(run_file.split('run_')[1].split('.')[0])
		if len(p_dict['test_top1'])> 0: # put 'if len(p_dict['val_mb_n_hist']) == run_length:' for runs without test
			complete_runs.append(run_file)

	# extract data from files
	for run_file in complete_runs:
		p_dict = pickle.load( open( perf_files_path+run_file, "rb" ) )
		run_number = int(run_file.split('run_')[1].split('.')[0])
		n_val_samples = len(p_dict['val_mb_n_hist'])
		run_val_mean = np.mean(p_dict['val_top1_hist'])
		test_t1 = p_dict['test_top1'][0]
		# exclude bad runs
		if (n_val_samples > 0) and (run_val_mean > .95 or run_val_mean < .3):
			print('[WARNING] bad run detected and excluded from analysis (based on validation performance): %s, run %i' %(spec_name, run_number))
		if test_t1 and test_t1 > 0. and test_t1 < .3:
			print('[WARNING] bad run detected and excluded from analysis (based on test performance): %s, run %i' %(spec_name, run_number))
		else:
			train_mb_n_hist_store.append(np.array(p_dict['train_mb_n_hist']))
			train_top1_hist_store.append(np.array(p_dict['train_top1_hist']))
			train_loss_hist_store.append(np.array(p_dict['train_loss_hist']))
			val_mb_n_hist_store.append(np.array(p_dict['val_mb_n_hist']))
			val_top1_hist_store.append(np.array(p_dict['val_top1_hist']))
			val_loss_hist_store.append(np.array(p_dict['val_loss_hist']))
			test_loss_store.append(p_dict['test_loss'])
			test_top1_store.append(p_dict['test_top1'])
			run_number_store.append(run_number)

	# if more than one run was done in this spec, build spec performance summary dict
	v_run_max_list = []
	spec_perf_dict = {}
	if len(run_number_store) > 0:
		n_runs = len(run_number_store)

		# get train statistics
		train_mb_n = np.array(train_mb_n_hist_store)
		train_top1 = np.array(train_top1_hist_store)
		train_loss = np.array(train_loss_hist_store)
		t_mb = train_mb_n[0,:]
		t_himax_run, t_lomax_run, _, _, t_mean_run, t_std_run, t_var_run, t_run_max_list, t_run_min_list = get_statistics(train_top1)
		_, _, t_loss_himin_run, t_loss_lomin_run, t_loss_mean_run, t_loss_std_run, t_loss_var_run, t_loss_run_max_list, t_loss_run_min_list = get_statistics(train_loss)

		# get val statistics
		val_mb_n = np.array(val_mb_n_hist_store)
		val_top1 = np.array(val_top1_hist_store)
		val_loss = np.array(val_loss_hist_store)
		v_mb = val_mb_n[0,:]
		v_himax_run, v_lomax_run, _, _, v_mean_run, v_std_run, v_var_run, v_run_max_list, v_run_min_list = get_statistics(val_top1)
		_, _, v_loss_himin_run, v_loss_lomin_run, v_loss_mean_run, v_loss_std_run, v_loss_var_run, v_loss_run_max_list, v_loss_run_min_list = get_statistics(val_loss)

		# put data into dict
		spec_perf_dict = { 'spec_name': spec_name,
							'n_runs': n_runs,
							# training loss main
							't_mb': t_mb,
							't_loss_run_min_list': t_loss_run_min_list,
							't_loss_run_min_max': np.max(t_loss_run_min_list),
							't_loss_run_min_min': np.min(t_loss_run_min_list),
							't_loss_run_min_mean': np.mean(t_loss_run_min_list),
							't_loss_run_min_var': np.var(t_loss_run_min_list),
							't_loss_run_min_std': np.std(t_loss_run_min_list),
							# training loss full runs for plotting
							't_loss_himin_run': t_loss_himin_run,
							't_loss_lomin_run': t_loss_lomin_run,
							't_loss_mean_run': t_loss_mean_run,
							't_loss_std_run': t_loss_std_run,
							't_loss_var_run': t_loss_var_run,
							# training acc main
							't_top1_run_max_list': t_run_max_list,
							't_top1_run_max_max': np.max(t_run_max_list),
							't_top1_run_max_min': np.min(t_run_max_list),
							't_top1_run_max_mean': np.mean(t_run_max_list),
							't_top1_run_max_std': np.std(t_run_max_list),
							't_top1_run_max_var': np.var(t_run_max_list),
							# training acc full runs for plotting
							't_top1_himax_run': t_himax_run,
							't_top1_lomax_run': t_lomax_run,
							't_top1_mean_run': t_mean_run,
							't_top1_std_run': t_std_run,
							't_top1_var_run': t_var_run,

							# val loss main
							'v_mb': v_mb, # training minibatch numbers corresponding to validation runs
							'v_loss_run_min_list': v_loss_run_min_list,
							'v_loss_run_min_max': np.max(v_loss_run_min_list),
							'v_loss_run_min_min': np.min(v_loss_run_min_list),
							'v_loss_run_min_mean': np.mean(v_loss_run_min_list),
							'v_loss_run_min_std': np.std(v_loss_run_min_list),
							'v_loss_run_min_var': np.var(v_loss_run_min_list),
							# val loss full runs for plotting
							'v_loss_himin_run': v_loss_himin_run,
							'v_loss_lomin_run': v_loss_lomin_run,
							'v_loss_mean_run': v_loss_mean_run,
							'v_loss_std_run': v_loss_std_run,
							'v_loss_var_run': v_loss_var_run,
							# var acc main
							'v_top1_run_max_list': v_run_max_list, 	# all runs' max values
							'v_top1_run_max_max': np.max(v_run_max_list),
							'v_top1_run_max_min': np.min(v_run_max_list),
							'v_top1_run_max_mean': np.mean(v_run_max_list),
							'v_top1_run_max_std': np.std(v_run_max_list),
							'v_top1_run_max_var': np.var(v_run_max_list),
							# val acc full runs for plotting
							'v_top1_himax_run': v_himax_run,	# complete best run (highest max)
							'v_top1_lomax_run': v_lomax_run,	# complete worst run (lowest max)
							'v_top1_mean_run': v_mean_run,		# complete mean run
							'v_top1_std_run': v_std_run,		# complete std around mean run
							'v_top1_var_run': v_var_run,		# complete var around mean run

							# test loss main
							'test_loss_list': test_loss_store,
							'test_loss_max': np.max(test_loss_store),
							'test_loss_min': np.min(test_loss_store),
							'test_loss_mean': np.mean(test_loss_store),
							'test_loss_var': np.var(test_loss_store),
							'test_loss_std': np.std(test_loss_store),
							# test acc main
							'test_top1_list': test_top1_store,
							'test_top1_max': np.max(test_top1_store),
							'test_top1_min': np.min(test_top1_store),
							'test_top1_mean': np.mean(test_top1_store),
							'test_top1_var': np.var(test_top1_store),
							'test_top1_std': np.std(test_top1_store) }

		# save dict as pickle file
		if not os.path.exists(analysis_savepath):
			os.makedirs(analysis_savepath)
		filename = 'performance_summary_'+spec_name+'.pkl'
		filepath = analysis_savepath + filename
		pickle.dump(spec_perf_dict, open(filepath,'wb'), protocol=3)
		print('[MESSAGE] file saved: %s (performance summary dict for "%s")' %(filepath, spec_name))

	# if validation was done and more than one run was done in this spec, make spec plot
	if len(v_run_max_list) > 1:
		n_mb_total = int(np.max(t_mb))
		if make_plot:
			fig = plt.figure(figsize=(10,10))
			ax = fig.add_subplot(1,1,1)
			if axis_2 == 'loss':
				ax2 = ax.twinx()
				ax2_ylim = [0.,1.]
				ax2.plot(t_mb, t_loss_mean_run, linewidth=1., color='red', label='training acc', alpha=0.2)
				ax2_ylim = [0.,np.max(t_loss_mean_run)]
				ax2.set_ylim(ax2_ylim[0],ax2_ylim[1])
			ax.plot(t_mb, t_mean_run, linewidth=1., color='green', label='training acc', alpha=0.4)
			# best and worst run
			ax.fill_between(v_mb, v_himax_run, v_lomax_run, linewidth=0., color='black', alpha=0.15)
			ax.plot(np.array([0,n_mb_total]), np.array([np.max(v_himax_run),np.max(v_himax_run)]), linewidth=1., linestyle='--', color='black', alpha=0.6)
			ax.plot(np.array([0,n_mb_total]), np.array([np.max(v_lomax_run),np.max(v_lomax_run)]), linewidth=1., linestyle='--', color='black', alpha=0.6)
			ax.plot(v_mb, v_himax_run, linewidth=1., linestyle='-', color='black', label='best run val acc (max = %.3f)'%(np.max(v_himax_run)), alpha=0.6)
			ax.plot(v_mb, v_lomax_run, linewidth=1., linestyle='-', color='black', label='worst run val acc (max = %.3f)'%(np.max(v_lomax_run)), alpha=0.6)
			# mean run
			ax.plot(np.array([0,len(t_mb)]), np.array([np.max(v_mean_run),np.max(v_mean_run)]), linewidth=1., linestyle='--', color='blue', alpha=0.8)
			ax.plot(v_mb, v_mean_run, linewidth=2., color='blue', label='mean val acc (max = %.3f)'%(np.max(v_mean_run)), alpha=0.8)
			# settings ax
			ax.set_ylim(0.1,1.)
			ax.set_xlim(0.,float(n_mb_total))
			ax.set_xticks(np.arange(0, float(n_mb_total)+.1, float(n_mb_total)/10.))
			ax.set_xticks(np.arange(0, float(n_mb_total)+.1, float(n_mb_total)/40.), minor=True)
			ax.set_yticks(np.arange(0.1, 1.01, .1))
			ax.set_yticks(np.arange(0.1, 1.01, .05), minor=True)
			ax.grid(which='major', linewidth=0.5, linestyle='dotted', color='black', alpha=0.3)
			ax.grid(which='minor', linewidth=0.5, linestyle='dotted', color='black', alpha=0.3)
			ax.set_aspect(float(n_mb_total)+.1/0.901)
			ax.set_title('performance analysis: %s (%i runs)' %(spec_name, n_runs))
			ax.legend(loc='lower left', prop={'size': 11})
			# save
			plt.tight_layout()
			filename = 'performance_analysis_'+str(spec_name)+'.png'
			if not os.path.exists(analysis_savepath):
				os.makedirs(analysis_savepath)
			plt.savefig(analysis_savepath+filename, dpi = 120, transparent=False, bbox_inches='tight')
			print('[MESSAGE] file saved: %s (performance analysis plot for spec "%s")' %(spec_name, analysis_savepath+filename))

	# return spec_perf_dict
	return spec_perf_dict

def get_statistics(data_array):
	# function gets a all full runs of a spec (train / val) and returns stats. do not use this for single values as in test.
	assert len(data_array.shape) == 2, 'data_array must be 2-dimensional: number of runs * number of performance measurement per run'
	if data_array.shape[1] > 0:
		# find himax and lomax run's indices
		run_max_list = []
		run_min_list = []
		for i in range(data_array.shape[0]):
			run = data_array[i,:]
			run_max_list.append(np.max(run))
			run_min_list.append(np.min(run))
		himax_run_idx = np.argmax(np.array(run_max_list))
		lomax_run_idx = np.argmin(np.array(run_max_list))
		himin_run_idx = np.argmax(np.array(run_min_list))
		lomin_run_idx = np.argmin(np.array(run_min_list))
		# isolate himax, mean and lomax run
		himax_run = data_array[himax_run_idx,:]
		lomax_run = data_array[lomax_run_idx,:]
		himin_run = data_array[himin_run_idx,:]
		lomin_run = data_array[lomin_run_idx,:]
		mean_run = np.mean(data_array, axis=0)
		std_run = np.std(data_array, axis=0)
		var_run = np.var(data_array, axis=0)
		# return many things
		return himax_run, lomax_run, himin_run, lomin_run, mean_run, std_run, var_run, run_max_list, run_min_list
	else:
		return 0, 0, 0, 0, 0, 0, 0, [0], [0]

def smooth_history(hist, n):
	smooth_hist = []
	for i in range(len(hist)):
		smooth_hist.append(sliding_window(hist[:i], n))
	return smooth_hist

def sliding_window(l, n):
	if len(l) < n:
		n = len(l)
	return np.mean(l[-n:])
