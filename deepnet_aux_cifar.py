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

def analysis(TaskSettings, Paths):
	experiment_name = Paths.exp_folder.split('/')[-2]
	# get spec spec names from their respective folder names
	spec_list = [f for f in os.listdir(Paths.exp_folder) if (os.path.isdir(os.path.join(Paths.exp_folder,f)) and not f.startswith('_'))]
	# put the pieces together and call spec_analysis() for each spec of the experiment
	spec_name_list = []
	n_runs_list = []
	mb_list = []
	best_run_list = []
	mean_run_list = []
	worst_run_list = []
	std_run_list = []
	mean_run_max_list = []
	var_run_max_list = []
	print('')
	print('=================================================================================================================================================================================================')
	for spec_name in spec_list:
		path = Paths.exp_folder+spec_name+'/'+Paths.performance_sub
		mb, best_run, worst_run, mean_run, std_run, n_runs, mean_run_max, var_run_max = spec_analysis(TaskSettings, Paths, spec_name=spec_name, perf_files_path=path, make_plot=True)
		spec_name_list.append(spec_name)
		n_runs_list.append(n_runs)
		mb_list.append(mb)
		best_run_list.append(best_run)
		mean_run_list.append(mean_run)
		worst_run_list.append(worst_run)
		std_run_list.append(std_run)
		mean_run_max_list.append(mean_run_max)
		var_run_max_list.append(var_run_max)
	print('=================================================================================================================================================================================================')
	# PLOTTING
	n_mb_total = int(np.max(mb_list[0]))
	fig = plt.figure(figsize=(10,10))
	ax = fig.add_subplot(1,1,1)
	cmap = matplotlib.cm.get_cmap('nipy_spectral')#('Dark2') ('nipy_spectral')
	color_list = ['black','blue','green','red']
	# layer 1
	for spec in range(len(spec_list)):
		# ax.plot(mb_list[spec], best_run_list[spec], linewidth=1.5, color=cmap(spec/len(spec_list)), alpha=0.15)
		ax.plot(np.array([0,300]), np.array([np.max(mean_run_list[spec]),np.max(mean_run_list[spec])]), linewidth=1.5, linestyle='-', color=cmap(spec/len(spec_list)), alpha=0.8)
	# layer 2
	for spec in range(len(spec_list)):
		ax.plot(mb_list[spec], mean_run_list[spec], linewidth=2.0, color=cmap(spec/len(spec_list)), label='[%i / m %.4f / v %.6f] %s (%.2f / %.2f)' %(n_runs_list[spec], 100*mean_run_max_list[spec], 100*var_run_max_list[spec], spec_list[spec], 100*np.max(mean_run_list[spec]), 100*np.max(best_run_list[spec])), alpha=0.8)
	# settings
	ax.set_ylim(0.1,1.)
	ax.set_xlim(0.,float(n_mb_total))
	ax.set_xticks(np.arange(0, float(n_mb_total)+.1, 1000))
	ax.set_xticks(np.arange(0, float(n_mb_total)+.1, 500), minor=True)
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
	filename = 'PA_'+str(experiment_name)+'.png'
	if not os.path.exists(savepath):
		os.makedirs(savepath)
	plt.savefig(savepath+filename, dpi = 120, transparent=False, bbox_inches='tight')
	print('[MESSAGE] file saved: %s (performance analysis plot for experiment "%s")' %(savepath+filename, experiment_name))

def spec_analysis(TaskSettings, Paths, spec_name=None, perf_files_path=None, axis_2='af_weights', make_plot=False):
	assert (axis_2 in ['af_weights', 'loss']) or axis_2 is None, 'axis_2 needs to be defined as None, \'af_weights\', or \'loss\'.'
	# define save path
	analysis_savepath = Paths.analysis
	# define spec name
	if spec_name is None:
		spec_name = TaskSettings.spec_name
	# get list of all performance files (pickle dicts) within a spec
	if perf_files_path == None:
		perf_files_path = Paths.performance
	perf_files_list = [f for f in os.listdir(perf_files_path) if (os.path.isfile(os.path.join(perf_files_path, f)) and ('.pkl' in f) and not ('test_performance_' in f))]
	# extract from dicts
	afw_hist_store = []
	train_mb_n_hist_store = []
	train_top1_hist_store = []
	train_loss_hist_store = []
	val_mb_n_hist_store = []
	val_top1_hist_store = []
	val_loss_hist_store = []
	run_number_store = []
	# make a list of all completed runs
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
		if len(p_dict['val_mb_n_hist']) == run_length:
			complete_runs.append(run_file)
	# extract data from files
	for run_file in complete_runs:
		p_dict = pickle.load( open( perf_files_path+run_file, "rb" ) )
		run_number = int(run_file.split('run_')[1].split('.')[0])
		n_val_samples = len(p_dict['val_mb_n_hist'])
		if np.mean(p_dict['val_top1_hist']) < .95 and np.mean(p_dict['val_top1_hist']) > .3:
			# if len(p_dict['val_af_weights_hist']) > 0:
				# build numpy array from af_weights_hist
				# af_weights_hist = p_dict['val_af_weights_hist']
				# afw_keys = []
				# for key in af_weights_hist[0]:
				#    afw_keys.append(key)
				# afw_hist = np.zeros((len(af_weights_hist), len(afw_keys), len(af_weights_hist[0][afw_keys[0]])))
				# for mb in range(len(af_weights_hist)):
				# 	for w in range(len(afw_keys)):
				# 		weights = np.array(af_weights_hist[mb][afw_keys[w]][:])
				# 		print(weights.shape, afw_hist.shape, afw_keys[w])
				# 		afw_hist[mb,w,:] = weights
				# afw_hist_store.append(afw_hist)
			train_mb_n_hist_store.append(np.array(p_dict['train_mb_n_hist']))
			train_top1_hist_store.append(np.array(p_dict['train_top1_hist']))
			train_loss_hist_store.append(np.array(p_dict['train_loss_hist']))
			val_mb_n_hist_store.append(np.array(p_dict['val_mb_n_hist']))
			val_top1_hist_store.append(np.array(p_dict['val_top1_hist']))
			val_loss_hist_store.append(np.array(p_dict['val_loss_hist']))
			run_number_store.append(run_number)
		else:
			print('[WARNING] bad run detected and excluded from analysis: %s, run %i' %(spec_name, run_number))
	if len(run_number_store) > 0:

		# if len(afw_hist_store) > 0:
		# 	val_afw_hist = np.array(afw_hist_store)
		# 	mean_afwh = np.mean(val_afw_hist, axis=0)
		# 	print(mean_afwh.shape,'hihihi') # correct so far
		# else:
		# 	mean_afwh = np.zeros(n_val_samples,1)

		train_mb_n = np.array(train_mb_n_hist_store)
		train_top1 = np.array(train_top1_hist_store)
		train_loss = np.array(train_loss_hist_store)
		val_mb_n = np.array(val_mb_n_hist_store)
		val_top1 = np.array(val_top1_hist_store)
		val_loss = np.array(val_loss_hist_store)
		# get relevant numbers
		n_runs = len(run_number_store)
		t_mb = train_mb_n[0,:]
		v_mb = val_mb_n[0,:]
		t_himax_run, t_lomax_run, _, _, t_mean_run, t_std_run, t_run_max_list = get_statistics(train_top1) # write these to dict => pkl
		v_himax_run, v_lomax_run, _, _, v_mean_run, v_std_run, v_run_max_list = get_statistics(val_top1) # write these to dict => pkl
		_, _, t_loss_himin_run, t_loss_lomin_run, t_loss_mean_run, t_loss_std_run, t_loss_run_max_list = get_statistics(train_loss) # write these to dict => pkl
		_, _, v_loss_himin_run, v_loss_lomin_run, v_loss_mean_run, v_loss_std_run, v_loss_run_max_list = get_statistics(val_loss) # write these to dict => pkl
		# put data into dict
		spec_perf_dict = { 'spec_name': spec_name,
							'n_runs': n_runs,
							# train
							't_mb': t_mb,
							't_himax_run': t_himax_run,
							't_lomax_run': t_lomax_run,
							't_mean_run': t_mean_run,
							't_std_run': t_std_run,
							't_loss_himin_run': t_loss_himin_run,
							't_loss_lomin_run': t_loss_lomin_run,
							't_loss_mean_run': t_loss_mean_run,
							't_loss_std_run': t_loss_std_run,
							't_run_max_list': t_run_max_list,
							't_loss_run_max_list': t_loss_run_max_list,
							# val
							'v_mb': v_mb,
							# 'afw_keys': afw_keys,
							# 'v_mean_afw': mean_afwh,
							'v_himax_run': v_himax_run,
							'v_lomax_run': v_lomax_run,
							'v_mean_run': v_mean_run,
							'v_std_run': v_std_run,
							'v_loss_himin_run': v_loss_himin_run,
							'v_loss_lomin_run': v_loss_lomin_run,
							'v_loss_mean_run': v_loss_mean_run,
							'v_loss_std_run': v_loss_std_run,
							'v_run_max_list': v_run_max_list,
							'v_loss_run_max_list': v_loss_run_max_list }
		# save dict as pickle file
		if not os.path.exists(analysis_savepath):
			os.makedirs(analysis_savepath)
		filename = 'performance_summary_'+spec_name+'.pkl'
		filepath = analysis_savepath + filename
		pickle.dump(spec_perf_dict, open(filepath,'wb'), protocol=3)
		print('[MESSAGE] file saved: %s (performance summary dict for "%s")' %(filepath, spec_name))
		# PLOTTING
		n_mb_total = int(np.max(t_mb))
		if make_plot:
			fig = plt.figure(figsize=(10,10))
			ax = fig.add_subplot(1,1,1)
			# plot af_weights
			# if axis_2 == 'af_weights':
			# 	ax2 = ax.twinx()
			# 	for w in range(mean_afwh.shape[1]):
			# 		weight_name = afw_keys[w].split('/')[1]
			# 		ax2.plot([0,np.max(v_mb)], [1.0,1.0], linewidth=1., color='black', alpha=0.8)
			# 		ax2.plot(v_mb, mean_afwh[:,w], linewidth=1., label=weight_name, alpha=0.5)
			# 		print(w,':',mean_afwh[:,w]) # debug why are these curves always different? # debug this row
			# 		ax2_ylim = [np.min(mean_afwh)-0.1,np.max(mean_afwh)/0.7]
			# 		ax2.set_ylim(ax2_ylim[0],ax2_ylim[1])
			# 		ax2.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), prop={'size': 11})
			# training performance
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
			ax.set_xticks(np.arange(0, float(n_mb_total)+.1, 1000))
			ax.set_xticks(np.arange(0, float(n_mb_total)+.1, 500), minor=True)
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
		# return relevant data
		return v_mb, v_himax_run, v_lomax_run, v_mean_run, v_std_run, n_runs, np.mean(v_run_max_list), np.var(v_run_max_list)
	else:
		return 0, 0, 0, 0, 0, 0, 0

def get_statistics(data_array):
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
	# return many things
	return himax_run, lomax_run, himin_run, lomin_run, mean_run, std_run, run_max_list

def smooth_history(hist, n):
	smooth_hist = []
	for i in range(len(hist)):
		smooth_hist.append(sliding_window(hist[:i], n))
	return smooth_hist

def sliding_window(l, n):
	if len(l) < n:
		n = len(l)
	return np.mean(l[-n:])
