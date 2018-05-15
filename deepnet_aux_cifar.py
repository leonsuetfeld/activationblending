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
from matplotlib import cm
import matplotlib.mlab as mlab

def analysis(TaskSettings, Paths, make_plot=True, make_hrtf=True):
	experiment_name = Paths.experiment.split('/')[-2]
	# get spec spec names from their respective folder names
	spec_list = [f for f in os.listdir(Paths.experiment) if (os.path.isdir(os.path.join(Paths.experiment,f)) and not f.startswith('0_'))]
	# put the pieces together and call spec_analysis() for each spec of the experiment
	spec_name_list, n_runs_list, mb_list, test_earlys_mb_mean, test_earlys_mb_std = [], [], [], [], []
	median_run_per_spec, best_run_per_spec, mean_run_per_spec, worst_run_per_spec, std_per_spec = [], [], [], [], []
	t_min_per_spec, t_max_per_spec, t_median_per_spec, t_mean_per_spec, t_var_per_spec, t_std_per_spec = [], [], [], [], [], []
	v_min_per_spec, v_max_per_spec, v_median_per_spec, v_mean_per_spec, v_var_per_spec, v_std_per_spec = [], [], [], [], [], []
	test_min_per_spec, test_max_per_spec, test_median_per_spec, test_mean_per_spec, test_var_per_spec, test_std_per_spec = [], [], [], [], [], []
	print('')
	print('================================================================================================================================================================================================================')
	spec_list_filtered = [] # will only contain specs that actually have completed runs
	for spec_name in spec_list:
		spec_path = Paths.experiment+spec_name+'/'
		spec_perf_dict = spec_analysis(TaskSettings, Paths, spec_name, spec_path, make_plot=True)
		# general info about spec
		if spec_perf_dict:
			spec_list_filtered.append(spec_name)
			spec_name_list.append(spec_perf_dict['spec_name'])
			n_runs_list.append(spec_perf_dict['n_runs'])
			test_earlys_mb_mean.append(spec_perf_dict['test_mb_n_mean'])
			test_earlys_mb_std.append(spec_perf_dict['test_mb_n_std'])
			# full runs for plotting
			mb_list.append(spec_perf_dict['v_mb'])
			median_run_per_spec.append(spec_perf_dict['v_top1_median_run'])
			best_run_per_spec.append(spec_perf_dict['v_top1_himax_run'])
			mean_run_per_spec.append(spec_perf_dict['v_top1_mean_run'])
			worst_run_per_spec.append(spec_perf_dict['v_top1_lomax_run'])
			std_per_spec.append(spec_perf_dict['v_top1_std_run'])
			# performance info for text output
			t_min_per_spec.append(spec_perf_dict['t_top1_run_max_min'])
			t_max_per_spec.append(spec_perf_dict['t_top1_run_max_max'])
			t_mean_per_spec.append(spec_perf_dict['t_top1_run_max_mean'])
			t_median_per_spec.append(spec_perf_dict['t_top1_run_max_median'])
			t_var_per_spec.append(spec_perf_dict['t_top1_run_max_var'])
			t_std_per_spec.append(spec_perf_dict['t_top1_run_max_std'])
			v_min_per_spec.append(spec_perf_dict['v_top1_run_max_min'])
			v_max_per_spec.append(spec_perf_dict['v_top1_run_max_max'])
			v_mean_per_spec.append(spec_perf_dict['v_top1_run_max_mean'])
			v_median_per_spec.append(spec_perf_dict['v_top1_run_max_median'])
			v_var_per_spec.append(spec_perf_dict['v_top1_run_max_var'])
			v_std_per_spec.append(spec_perf_dict['v_top1_run_max_std'])
			test_min_per_spec.append(spec_perf_dict['test_top1_min'])
			test_max_per_spec.append(spec_perf_dict['test_top1_max'])
			test_mean_per_spec.append(spec_perf_dict['test_top1_mean'])
			test_median_per_spec.append(spec_perf_dict['test_top1_median'])
			test_var_per_spec.append(spec_perf_dict['test_top1_var'])
			test_std_per_spec.append(spec_perf_dict['test_top1_std'])
		print('[MESSAGE] spec analysis complete:', spec_name)
	print('================================================================================================================================================================================================================')

	# BIG FINAL PLOT
	if make_plot and len(mb_list[0])>0:
		n_mb_total = int(np.max(mb_list[0]))
		fig = plt.figure(figsize=(10,10))
		ax = fig.add_subplot(1,1,1)
		cmap = matplotlib.cm.get_cmap('nipy_spectral')#('Dark2') ('nipy_spectral')
		color_list = ['black','blue','green','red']
		# layer 1
		for spec in range(len(spec_list_filtered)):
			# ax.plot(mb_list[spec], best_run_per_spec[spec], linewidth=1.5, color=cmap(spec/len(spec_list)), alpha=0.15)
			ax.plot(np.array([0,300]), np.array([np.max(mean_run_per_spec[spec]),np.max(mean_run_per_spec[spec])]), linewidth=1.5, linestyle='-', color=cmap(spec/len(spec_list_filtered)), alpha=0.8)
		# layer 2
		for spec in range(len(spec_list_filtered)):
			ax.plot(mb_list[spec], mean_run_per_spec[spec], linewidth=2.0, color=cmap(spec/len(spec_list_filtered)), label='[%i / m %.4f / v %.6f] %s' %(n_runs_list[spec], 100*test_mean_per_spec[spec], 100*test_var_per_spec[spec], spec_list_filtered[spec]), alpha=0.8)
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
		plt.close()
		print('[MESSAGE] file saved: %s (performance analysis plot for experiment "%s")' %(savepath+plot_filename, experiment_name))

	# HUMAN READABLE TEXT FILE:
	if make_hrtf:
		savepath = Paths.analysis
		hrtf_filename = 'main_performance_analysis.csv'
		with open(savepath+hrtf_filename, 'w') as csvfile:
			writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
			writer.writerow(['spec_name','n_runs', 'test / e.s. mb [mean]', 'test / e.s. mb [std]',
							 'train acc [median]','train acc [mean]','train acc [var]','train acc [std]','train acc [max]','train acc [min]',
							 'val acc [median]','val acc [mean]','val acc [var]','val acc [std]','val acc [max]','val acc [min]',
							 'test acc [median]','test acc [mean]','test acc [var]','test acc [std]','test acc [max]','test acc [min]'])
			for spec in range(len(spec_name_list)):
				writer.writerow([spec_name_list[spec], n_runs_list[spec], test_earlys_mb_mean[spec], test_earlys_mb_std[spec],
								t_median_per_spec[spec], t_mean_per_spec[spec], t_var_per_spec[spec], t_std_per_spec[spec], t_max_per_spec[spec], t_min_per_spec[spec],
								v_median_per_spec[spec], v_mean_per_spec[spec], v_var_per_spec[spec], v_std_per_spec[spec], v_max_per_spec[spec], v_min_per_spec[spec],
								test_median_per_spec[spec], test_mean_per_spec[spec], test_var_per_spec[spec], test_std_per_spec[spec], test_max_per_spec[spec], test_min_per_spec[spec]])
		print('[MESSAGE] file saved: %s (performance analysis csv for experiment "%s")' %(savepath+hrtf_filename, experiment_name))

def spec_analysis(TaskSettings, Paths, spec_name, spec_path, axis_2='af_weights', make_plot=False, return_loss=False):
	assert (axis_2 in ['af_weights', 'loss']) or axis_2 is None, 'axis_2 needs to be defined as None, \'af_weights\', or \'loss\'.'
	analysis_savepath = Paths.analysis
	# get list of all performance files (pickle dicts) within a spec
	rec_files_list = []
	if os.path.isdir(spec_path):
		run_folder_list = [f for f in os.listdir(spec_path) if (os.path.isdir(os.path.join(spec_path, f)) and ('run_' in f))]
		for run_folder in run_folder_list:
			run_dir = os.path.join(spec_path, run_folder)+'/'
			relative_path_perf_file = [f for f in os.listdir(run_dir) if (('.pkl' in f) and ('record' in f)) ]
			if len(relative_path_perf_file) > 1:
				print('[WARNING] more than one record file found in %s. Using first file (%s).' %(run_dir, relative_path_perf_file[0]))
			if len(relative_path_perf_file) > 0:
				rec_files_list.append(os.path.join(run_dir, relative_path_perf_file[0]))

	# prepare extraction from dicts
	afw_hist_store, run_number_store = [], []
	train_mb_n_hist_store, train_top1_hist_store, train_loss_hist_store = [], [], []
	val_mb_n_hist_store, val_top1_hist_store, val_loss_hist_store = [], [], []
	test_mb_n_hist_store, test_top1_store, test_loss_store = [], [], []

	# criteria for exclusion: incomplete runs. this section makes a list of all completed runs
	n_val_samples_list = []
	for rec_file in rec_files_list:
		rec_dict = pickle.load( open( rec_file, "rb" ) )
		run_number = int(rec_file.split('/')[-2].split('run_')[-1])
		n_val_samples = len(rec_dict['val_mb_n_hist'])
		n_val_samples_list.append(n_val_samples)
	if len(n_val_samples_list) > 0:
		run_length = np.max(n_val_samples_list)
	complete_runs = []
	for rec_file in rec_files_list:
		if os.path.getsize(rec_file) > 0:
			rec_dict = pickle.load( open( rec_file, "rb" ) )
			run_number = int(rec_file.split('/')[-2].split('run_')[-1])
			if len(rec_dict['test_top1']) > 0: # put 'if len(rec_dict['val_mb_n_hist']) == run_length:' for runs without test
				complete_runs.append(rec_file)
			else:
				print('[WARNING] spec %s run %i found to be incomplete (no test accuracy data found), excluded from analysis.'%(spec_name, run_number))
	if len(complete_runs) == 0:
		print('[WARNING] No complete run found for spec %s' %(spec_name))
		return {}

	# extract data from files
	for rec_file in complete_runs:
		rec_dict = pickle.load( open( rec_file, "rb" ) )
		run_number = int(rec_file.split('/')[-2].split('run_')[-1])
		n_val_samples = len(rec_dict['val_mb_n_hist'])
		run_val_mean = np.mean(rec_dict['val_top1_hist'])
		test_t1 = rec_dict['test_top1'][0]
		# exclude bad runs
		if (n_val_samples > 0) and (run_val_mean > .95 or run_val_mean < .3):
			print('[WARNING] bad run detected and excluded from analysis (based on validation performance): %s, run %i' %(spec_name, run_number))
		if test_t1 and test_t1 > 0. and test_t1 < .3:
			print('[WARNING] bad run detected and excluded from analysis (based on test performance): %s, run %i' %(spec_name, run_number))
		else:
			train_mb_n_hist_store.append(np.array(rec_dict['train_mb_n_hist']))
			train_top1_hist_store.append(np.array(rec_dict['train_top1_hist']))
			train_loss_hist_store.append(np.array(rec_dict['train_loss_hist']))
			test_mb_n_hist_store.append(rec_dict['test_mb_n_hist'])
			test_loss_store.append(rec_dict['test_loss'])
			test_top1_store.append(rec_dict['test_top1'])
			run_number_store.append(run_number)
			# special treatment for validation data: remove double mb counts from initial validation after restore
			vmb_hist = rec_dict['val_mb_n_hist']
			vt1_hist = rec_dict['val_top1_hist']
			vlo_hist = rec_dict['val_loss_hist']
			vmb_hist, vt1_hist, vlo_hist = remove_double_logs(vmb_hist, vt1_hist, vlo_hist)
			val_mb_n_hist_store.append(np.array(vmb_hist))
			val_top1_hist_store.append(np.array(vt1_hist))
			val_loss_hist_store.append(np.array(vlo_hist))

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
		t_median_run, t_himax_run, t_lomax_run, _, _, t_mean_run, t_std_run, t_var_run, t_run_max_list, _ = get_statistics(train_top1)
		t_loss_median_run, _, _, t_loss_himin_run, t_loss_lomin_run, t_loss_mean_run, t_loss_std_run, t_loss_var_run, _, t_loss_run_min_list = get_statistics(train_loss)

		# get val statistics
		val_mb_n = np.array(val_mb_n_hist_store)
		val_top1 = np.array(val_top1_hist_store)
		val_loss = np.array(val_loss_hist_store)
		v_mb = val_mb_n[0,:]
		v_median_run, v_himax_run, v_lomax_run, _, _, v_mean_run, v_std_run, v_var_run, v_run_max_list, _ = get_statistics(val_top1)
		v_loss_median_run, _, _, v_loss_himin_run, v_loss_lomin_run, v_loss_mean_run, v_loss_std_run, v_loss_var_run, _, v_loss_run_min_list = get_statistics(val_loss)

		# put data into dict
		spec_perf_dict = { 'spec_name': spec_name,
							'n_runs': n_runs,
							# training loss main
							't_mb': t_mb,
							't_loss_run_min_list': t_loss_run_min_list,
							't_loss_run_min_median': np.median(t_loss_run_min_list),
							't_loss_run_min_max': np.max(t_loss_run_min_list),
							't_loss_run_min_min': np.min(t_loss_run_min_list),
							't_loss_run_min_mean': np.mean(t_loss_run_min_list),
							't_loss_run_min_var': np.var(t_loss_run_min_list),
							't_loss_run_min_std': np.std(t_loss_run_min_list),
							# training loss full runs for plotting
							't_loss_median_run': t_loss_median_run,
							't_loss_himin_run': t_loss_himin_run,
							't_loss_lomin_run': t_loss_lomin_run,
							't_loss_mean_run': t_loss_mean_run,
							't_loss_std_run': t_loss_std_run,
							't_loss_var_run': t_loss_var_run,
							# training acc main
							't_top1_run_max_list': t_run_max_list,
							't_top1_run_max_median': np.median(t_run_max_list),
							't_top1_run_max_max': np.max(t_run_max_list),
							't_top1_run_max_min': np.min(t_run_max_list),
							't_top1_run_max_mean': np.mean(t_run_max_list),
							't_top1_run_max_std': np.std(t_run_max_list),
							't_top1_run_max_var': np.var(t_run_max_list),
							# training acc full runs for plotting
							't_top1_median_run': t_median_run,
							't_top1_himax_run': t_himax_run,
							't_top1_lomax_run': t_lomax_run,
							't_top1_mean_run': t_mean_run,
							't_top1_std_run': t_std_run,
							't_top1_var_run': t_var_run,

							# val loss main
							'v_mb': v_mb, # training minibatch numbers corresponding to validation runs
							'v_loss_run_min_list': v_loss_run_min_list,
							'v_loss_run_min_median': np.median(v_loss_run_min_list),
							'v_loss_run_min_max': np.max(v_loss_run_min_list),
							'v_loss_run_min_min': np.min(v_loss_run_min_list),
							'v_loss_run_min_mean': np.mean(v_loss_run_min_list),
							'v_loss_run_min_std': np.std(v_loss_run_min_list),
							'v_loss_run_min_var': np.var(v_loss_run_min_list),
							# val loss full runs for plotting
							'v_loss_median_run': v_loss_median_run,
							'v_loss_himin_run': v_loss_himin_run,
							'v_loss_lomin_run': v_loss_lomin_run,
							'v_loss_mean_run': v_loss_mean_run,
							'v_loss_std_run': v_loss_std_run,
							'v_loss_var_run': v_loss_var_run,
							# var acc main
							'v_top1_run_max_list': v_run_max_list, 	# all runs' max values
							'v_top1_run_max_median': np.median(v_run_max_list),
							'v_top1_run_max_max': np.max(v_run_max_list),
							'v_top1_run_max_min': np.min(v_run_max_list),
							'v_top1_run_max_mean': np.mean(v_run_max_list),
							'v_top1_run_max_std': np.std(v_run_max_list),
							'v_top1_run_max_var': np.var(v_run_max_list),
							# val acc full runs for plotting
							'v_top1_median_run': v_median_run,	# complete median run
							'v_top1_himax_run': v_himax_run,	# complete best run (highest max)
							'v_top1_lomax_run': v_lomax_run,	# complete worst run (lowest max)
							'v_top1_mean_run': v_mean_run,		# complete mean run
							'v_top1_std_run': v_std_run,		# complete std around mean run
							'v_top1_var_run': v_var_run,		# complete var around mean run

							# test loss main
							'test_loss_list': test_loss_store,
							'test_loss_median': np.median(test_loss_store),
							'test_loss_max': np.max(test_loss_store),
							'test_loss_min': np.min(test_loss_store),
							'test_loss_mean': np.mean(test_loss_store),
							'test_loss_var': np.var(test_loss_store),
							'test_loss_std': np.std(test_loss_store),
							# test acc main
							'test_top1_list': test_top1_store,
							'test_top1_median': np.median(test_top1_store),
							'test_top1_max': np.max(test_top1_store),
							'test_top1_min': np.min(test_top1_store),
							'test_top1_mean': np.mean(test_top1_store),
							'test_top1_var': np.var(test_top1_store),
							'test_top1_std': np.std(test_top1_store),

							# early stopping minibatches
							'test_mb_n_hist': test_mb_n_hist_store,
							'test_mb_n_min': np.amin(test_mb_n_hist_store),
							'test_mb_n_max': np.amax(test_mb_n_hist_store),
							'test_mb_n_mean': np.mean(test_mb_n_hist_store),
							'test_mb_n_median': np.median(test_mb_n_hist_store),
							'test_mb_n_std': np.std(test_mb_n_hist_store) }

		# save dict as pickle file
		if not os.path.exists(analysis_savepath):
			os.makedirs(analysis_savepath)
		filename = 'performance_summary_'+spec_name+'.pkl'
		filepath = analysis_savepath + filename
		pickle.dump(spec_perf_dict, open(filepath,'wb'), protocol=3)
		print('[MESSAGE] file saved: %s (performance summary dict for "%s")' %(filepath, spec_name))

		# for the current spec: put all runs' val learning curves in one plot
		n_mb_total = int(np.max(t_mb))
		if make_plot:
			fig = plt.figure(figsize=(10,10))
			ax = fig.add_subplot(1,1,1)
			for r in range(len(train_mb_n_hist_store)):
				colors = [ cm.plasma(x) for x in np.linspace(0.01, 0.99, len(train_mb_n_hist_store))  ]
				ax.plot(val_mb_n_hist_store[r], val_top1_hist_store[r], linewidth=0.5, color=colors[r], label='run '+str(run_number_store[r]), alpha=0.5)
			ax.plot(np.array([0,len(t_mb)]), np.array([np.max(v_himax_run),np.max(v_himax_run)]), linewidth=1., linestyle='--', color='black', alpha=0.5, label='highest max')
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
			ax.set_title('all runs: %s (%i runs)' %(spec_name, n_runs))
			# ax.legend(loc='lower left', prop={'size': 11})
			# save
			plt.tight_layout()
			filename = 'all_runs_'+str(spec_name)+'.png'
			if not os.path.exists(analysis_savepath):
				os.makedirs(analysis_savepath)
			plt.savefig(analysis_savepath+filename, dpi = 120, transparent=False, bbox_inches='tight')
			plt.close()
			print('[MESSAGE] file saved: %s (performance analysis plot for spec "%s")' %(spec_name, analysis_savepath+filename))

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
			plt.close()
			print('[MESSAGE] file saved: %s (performance analysis plot for spec "%s")' %(spec_name, analysis_savepath+filename))

	# return spec_perf_dict
	return spec_perf_dict

def remove_double_logs(vmb_hist, vt1_hist, vlo_hist):
	vmb_new, vt1_new, vlo_new = [], [], []
	for i in range(len(vmb_hist)):
		if not vmb_hist[i] in vmb_new:
			vmb_new.append(vmb_hist[i])
			vt1_new.append(vt1_hist[i])
			vlo_new.append(vlo_hist[i])
	return vmb_new, vt1_new, vlo_new

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
		median_run_idx = np.argsort(np.array(run_min_list))[len(np.array(run_min_list))//2]
		# isolate himax, mean and lomax run
		himax_run = data_array[himax_run_idx,:]
		lomax_run = data_array[lomax_run_idx,:]
		himin_run = data_array[himin_run_idx,:]
		lomin_run = data_array[lomin_run_idx,:]
		median_run = data_array[median_run_idx,:]
		mean_run = np.mean(data_array, axis=0)
		std_run = np.std(data_array, axis=0)
		var_run = np.var(data_array, axis=0)
		# return many things
		return median_run, himax_run, lomax_run, himin_run, lomin_run, mean_run, std_run, var_run, run_max_list, run_min_list
	else:
		return 0, 0, 0, 0, 0, 0, 0, 0, [0], [0]

def smooth_history(hist, n):
	smooth_hist = []
	for i in range(len(hist)):
		smooth_hist.append(sliding_window(hist[:i], n))
	return smooth_hist

def sliding_window(l, n):
	if len(l) < n:
		n = len(l)
	return np.mean(l[-n:])

def visualize_performance(TaskSettings, Paths):
	# load
	filename = Paths.recorder_files+'record_'+str(TaskSettings.spec_name)+'_run_'+str(TaskSettings.run)+'.pkl'
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
	ax.set_xticks(np.arange(0, n_mb_total, n_mb_total//6))
	ax.set_yticks(np.arange(0., 1.1, .1))
	ax.set_yticks(np.arange(0., 1.1, .02), minor=True)
	ax.grid(which='major', alpha=0.6)
	ax.grid(which='minor', alpha=0.1)
	ax.set_aspect(float(n_mb_total))
	ax.set_title('accuracy over epochs ('+str(TaskSettings.spec_name)+', run_'+str(TaskSettings.run)+')')
	ax.legend(loc=4)
	plt.tight_layout()
	# save
	savepath = Paths.run_learning_curves
	filename = 'learning_curves_'+str(TaskSettings.spec_name)+'_run_'+str(TaskSettings.run)+'.png'
	if not os.path.exists(savepath):
		os.makedirs(savepath)
	plt.savefig(savepath+filename, dpi = 120, transparent=False, bbox_inches='tight')
	plt.close('all')
	print('================================================================================================================================================================================================================')
	print('[MESSAGE] performance figure saved: %s' %(savepath+filename))
	print('================================================================================================================================================================================================================')

def lr_scheduler(TaskSettings, current_mb): # new variablse: TaskSettings.lr_schedule_type (constant, step, continuous), TaskSettings.lr_decay (e.g., 1e-6)
	if TaskSettings.lr_schedule_type == 'constant':
		return TaskSettings.lr
	if TaskSettings.lr_schedule_type == 'decay':
		lr = TaskSettings.lr * (1. / (1. + TaskSettings.lr_decay * current_mb))
		return lr
	if TaskSettings.lr_schedule_type == 'linear':
		if current_mb < TaskSettings.lr_lin_steps:
			return np.linspace(TaskSettings.lr, TaskSettings.lr_lin_min, num=TaskSettings.lr_lin_steps)[current_mb] #start_lr=0.04, stop_lr=0.00004
		else:
			return TaskSettings.lr_lin_min
	if TaskSettings.lr_schedule_type == 'step':
		mb_per_ep = 50000*(1-TaskSettings.val_set_fraction)//TaskSettings.minibatch_size
		lr_step_mb = np.array(TaskSettings.lr_step_ep) * mb_per_ep
		step_mbs_relative = lr_step_mb - current_mb
		for i in range(len(step_mbs_relative)):
			if step_mbs_relative[i] > 0:
				if i == 0: return TaskSettings.lr
				else: return TaskSettings.lr * TaskSettings.lr_step_multi[i-1]
		return TaskSettings.lr * TaskSettings.lr_step_multi[-1]

def args_to_txt(args, Paths, training_complete_info='', test_complete_info=''):
	# prepare
	experiment_name = args['experiment_name']
	spec_name = args['spec_name']
	run = args['run']
	network = args['network']
	task = args['task']
	mode = args['mode']
	# write file
	if not os.path.exists(Paths.experiment_spec_run):
		os.makedirs(Paths.experiment_spec_run)
	filename = "run_info_"+str(experiment_name)+"_"+str(spec_name)+"_run_"+str(run)+".txt"
	with open(Paths.experiment_spec_run+filename, "w+") as text_file:
		print("{:>35}".format('RUN SETTINGS:'), file=text_file)
		print("", file=text_file)
		print("{:>35} {:<30}".format('experiment_name:',experiment_name), file=text_file)
		print("{:>35} {:<30}".format('spec_name:', spec_name), file=text_file)
		print("{:>35} {:<30}".format('run:', run), file=text_file)
		if len(training_complete_info) > 0:
			print("", file=text_file)
			print("{:>35} {:<30}".format('training complete:', training_complete_info), file=text_file)
		if len(test_complete_info) > 0:
			print("{:>35} {:<30}".format('test complete:', test_complete_info), file=text_file)
		print("", file=text_file)
		print("{:>35} {:<30}".format('network:', network), file=text_file)
		print("{:>35} {:<30}".format('task:', task), file=text_file)
		print("{:>35} {:<30}".format('mode:', mode), file=text_file)
		print("", file=text_file)
		for key in args.keys():
			if args[key] is not None and key not in ['experiment_name','spec_name','run','network','task','mode']:
				print("{:>35} {:<30}".format(key+':', str(args[key])), file=text_file)
