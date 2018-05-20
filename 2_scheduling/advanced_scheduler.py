import time
import os
import sys
import subprocess
import numpy as np
import csv
from filelock import Timeout, FileLock

# ##############################################################################
# ### ADVANCED SCHEDULER #######################################################
# ##############################################################################

def default_spec():
	spec = {"path_relative":                  './',
			"experiment_name":                'default_experiment',
			"spec_name":                      'default_spec',
			"run":                            '1',
			"task":                           'cifar10',
			"preprocessing":                  'ztrans',
			"network":                        'smcn',
			"mode":                           'training',
			"n_minibatches":                  '60000',
			"minibatch_size":                 '256',
			"dropout_keep_probs":             '0.5',
			"dropout_keep_probs_inference":   '1.0',
			"optimizer":                      'Adam',
			"lr":                             '0.001',
			"lr_schedule_type":               'constant',
			"lr_decay":            			  '0.00004',
			"lr_lin_min":            		  '0.0004',
			"lr_lin_steps":            		  '60000',
			"lr_step_ep":                     '200 250 300 ',
			"lr_step_multi":                  '0.1 0.01 0.001',
			"use_wd":                         'False',
			"wd_lambda":                      '0.01',
			"create_val_set":                 'True',
			"val_set_fraction":               '0.05',
			"af_set":                         '1_linu',
			"af_weights_init":                'default',
			"load_af_weights_from":           'none',
			"norm_blendw_at_init":            'False',
			"safe_af_ws_n":                   '60',
			"safe_all_ws_n":                  '2',
			"blend_trainable":                'False',
			"blend_mode":                     'unrestricted',
			"swish_beta_trainable":           'False',
			"walltime":                       '89.0',
			"create_checkpoints":             'True',
			"epochs_between_checkpoints":	  '8',
			"save_af_weights_at_test_mb":	  'True',
			"save_all_weights_at_test_mb":    'True',
			"create_lc_on_the_fly":           'True' }
	return spec

def get_spec_dict(dict_of_changes={}):
	spec = default_spec()
	if dict_of_changes:
		for key, value in dict_of_changes.items():
			spec[key] = value
	return spec

def dict_to_command_str(dict):
	command = 'python3 deepnet_main.py'
	for key, value in dict.items():
		command += ' -'+key+' '+value
	return command

def get_command(experiment_name, spec_name, run):
	# name changes made to every dict
	changes_dict = { "experiment_name": experiment_name, "spec_name": spec_name, "run": str(run) }
	if '_c100_' in spec_name:
		changes_dict["task"] = 'cifar100'
	if 'smcnb_' in spec_name:
		changes_dict["network"] = 'smcnLin'
	if 'smcnc_' in spec_name:
		changes_dict["network"] = 'smcnDeep'
	if '_sgdm_l_' in spec_name:
		changes_dict["optimizer"] = 'Momentum'
		changes_dict["lr"] = '0.01'
		changes_dict["lr_schedule_type"] = 'linear'
		changes_dict["lr_lin_min"] = '0.0004'
		changes_dict["lr_lin_step"] = '60000'
	if '_sgdm_d1_' in spec_name:
		changes_dict["optimizer"] = 'Momentum'
		changes_dict["lr"] = '0.01'
		changes_dict["lr_schedule_type"] = 'decay'
		changes_dict["lr_decay"] = '0.0003'
	if '_sgdm_d2_' in spec_name:
		changes_dict["optimizer"] = 'Momentum'
		changes_dict["lr"] = '0.01'
		changes_dict["lr_schedule_type"] = 'decay'
		changes_dict["lr_decay"] = '0.00003'
	if '_sgdm_d3_' in spec_name:
		changes_dict["optimizer"] = 'Momentum'
		changes_dict["lr"] = '0.01'
		changes_dict["lr_schedule_type"] = 'decay'
		changes_dict["lr_decay"] = '0.00001'
	if '_sgdm_d4_' in spec_name:
		changes_dict["optimizer"] = 'Momentum'
		changes_dict["lr"] = '0.01'
		changes_dict["lr_schedule_type"] = 'decay'
		changes_dict["lr_decay"] = '0.000001'
	if '_sgdm_c_' in spec_name:
		changes_dict["optimizer"] = 'Momentum'
		changes_dict["lr"] = '0.01'
		changes_dict["lr_schedule_type"] = 'constant'
	# I
	if '_I' in spec_name and not '_alpha_I' in spec_name:
		pass
	# alpha I
	elif '_alpha_I' in spec_name:
		changes_dict["blend_trainable"] = 'True'
	# tanh
	elif '_tanh' in spec_name and not '_alpha_tanh' in spec_name:
		changes_dict["af_set"] = '1_tanh'
	# alpha tanh
	elif '_alpha_tanh' in spec_name:
		changes_dict["af_set"] = '1_tanh'
		changes_dict["blend_trainable"] = 'True'
	# relu
	elif '_relu' in spec_name and not '_alpha_relu' in spec_name:
		changes_dict["af_set"] = '1_relu'
	# alpha relu
	elif '_alpha_relu' in spec_name:
		changes_dict["af_set"] = '1_relu'
		changes_dict["blend_trainable"] = 'True'
	# elu
	elif '_elu' in spec_name and not '_alpha_elu' in spec_name:
		changes_dict["af_set"] = '1_jelu'
	# alpha elu
	elif '_alpha_elu' in spec_name:
		changes_dict["af_set"] = '1_jelu'
		changes_dict["blend_trainable"] = 'True'
	# selu
	elif '_selu' in spec_name and not '_alpha_selu' in spec_name:
		changes_dict["af_set"] = '1_selu'
	# alpha selu
	elif '_alpha_selu' in spec_name:
		changes_dict["af_set"] = '1_selu'
		changes_dict["blend_trainable"] = 'True'
	# swish
	elif '_swish' in spec_name and not '_alpha_swish' in spec_name:
		changes_dict["af_set"] = '1_jswish'
		changes_dict["swish_beta_trainable"] = 'True'
	# alpha swish
	elif '_alpha_swish' in spec_name:
		changes_dict["af_set"] = '1_jswish'
		changes_dict["blend_trainable"] = 'True'
		changes_dict["swish_beta_trainable"] = 'True'
	# ABU_TERIS
	elif '_ABU_TERIS' in spec_name:
		changes_dict["af_set"] = '5_blend5_swish'
		changes_dict["blend_trainable"] = 'True'
		changes_dict["swish_beta_trainable"] = 'True'
	# ABU_N_TERIS
	elif '_ABU_N_TERIS' in spec_name:
		changes_dict["af_set"] = '5_blend5_swish'
		changes_dict["blend_trainable"] = 'True'
		changes_dict["blend_mode"] = 'normalized'
		changes_dict["swish_beta_trainable"] = 'True'
	# ABU_P_TERIS
	elif '_ABU_P_TERIS' in spec_name:
		changes_dict["af_set"] = '5_blend5_swish'
		changes_dict["blend_trainable"] = 'True'
		changes_dict["blend_mode"] = 'posnormed'
		changes_dict["swish_beta_trainable"] = 'True'
	# ABU_A_TERIS
	elif '_ABU_A_TERIS' in spec_name:
		changes_dict["af_set"] = '5_blend5_swish'
		changes_dict["blend_trainable"] = 'True'
		changes_dict["blend_mode"] = 'absnormed'
		changes_dict["swish_beta_trainable"] = 'True'
	# ABU_S_TERIS
	elif '_ABU_S_TERIS' in spec_name:
		changes_dict["af_set"] = '5_blend5_swish'
		changes_dict["blend_trainable"] = 'True'
		changes_dict["blend_mode"] = 'softmaxed'
		changes_dict["swish_beta_trainable"] = 'True'
	else:
		raise IOError('\n[ERROR] Requested spec not found (%s).'%(spec_name))
	if '_pre' in spec_name:
		changes_dict["load_af_weights_from"] = spec_name.split('_pre_')[0]
	if '_prenorm' in spec_name:
		changes_dict["load_af_weights_from"] = spec_name.split('_pre_')[0]
		changes_dict["norm_blendw_at_init"] = True
	if '_notrain' in spec_name:
		changes_dict["blend_trainable"] = 'False'
		changes_dict["swish_beta_trainable"] = 'False'
	return dict_to_command_str(get_spec_dict(changes_dict))

def csv_update(csv_path, filename, experiment_path, experiment_name, mark_running_as_incomplete=False):
	finished_list = []
	unfinished_list = []
	# look through recorder files
	if os.path.exists(os.getcwd()+experiment_path+experiment_name+'/'):
		spec_folders = [f for f in os.listdir(os.getcwd()+experiment_path+experiment_name+'/') if os.path.isdir(os.path.join(os.getcwd()+experiment_path+experiment_name+'/',f))]
		for spec in spec_folders:
			run_folders = [f for f in os.listdir(os.getcwd()+experiment_path+experiment_name+'/'+spec) if os.path.isdir(os.path.join(os.getcwd()+experiment_path+experiment_name+'/'+spec+'/',f))]
			for run_folder in run_folders:
				run = int(run_folder.split('_')[-1])
				run_info_files = [f for f in os.listdir(os.getcwd()+experiment_path+experiment_name+'/'+spec+'/'+run_folder) if ('run_info_' in f)]
				if len(run_info_files) > 0:
					training_complete = False
					test_complete = False
					for line in open(os.getcwd()+experiment_path+experiment_name+'/'+spec+'/'+run_folder+'/'+run_info_files[0],'r'):
						if 'training complete: True' in line:
							training_complete = True
						if 'test complete: True' in line:
							test_complete = True
					if training_complete and test_complete:
						finished_list.append([spec, run])
					else:
						unfinished_list.append([spec, run])
				else:
					unfinished_list.append([spec, run])
	# update csv
	csv = open_csv_as_list(csv_path, filename)
	if len(finished_list) > 0:
		for entry in finished_list:
			spec_idx = csv_spec_to_num(csv, entry[0])
			run = entry[1]
			csv[spec_idx][run] = 'finished'
	if len(unfinished_list) > 0:
		for entry in unfinished_list:
			spec_idx = csv_spec_to_num(csv, entry[0])
			run = entry[1]
			if mark_running_as_incomplete:
				csv[spec_idx][run] = 'incomplete'
			else:
				cell = csv[spec_idx][run]
				if 'running' in cell:
					if (time.time()-float(cell.split(' ')[-1]))/60. > 90.0:
						csv[spec_idx][run] = 'incomplete'
				else:
					csv[spec_idx][run] = 'incomplete'
	save_list_as_csv(csv, csv_path, filename)
	return ((np.array(csv).shape[0]-1)*(np.array(csv).shape[1]-1)) - len(finished_list)

def csv_spec_to_num(csv, spec):
	for i, row in enumerate(csv):
		if spec in row[0]:
			return i

def csv_lookup_spec_run(csv_path, filename, valid=['','incomplete'], preferred=['incomplete'], random=True):
	csv = open_csv_as_list(csv_path, filename)
	n_specs = len(csv)-1
	n_runs = len(csv[0])-1
	valid_spec_run_combinations = []
	preferred_spec_run_combinations = []
	for spec_id in range(n_specs):
		for run_id in range(n_runs):
			if csv[spec_id+1][run_id+1] in valid:
				valid_spec_run_combinations.append([spec_id+1, run_id+1])
			if csv[spec_id+1][run_id+1] in preferred:
				preferred_spec_run_combinations.append([spec_id+1, run_id+1])
	if len(valid_spec_run_combinations) == 0:
		return -1, -1
	if random:
		if len(preferred_spec_run_combinations) > 0:
			[spec_idx, run] = preferred_spec_run_combinations[np.random.randint(0, len(preferred_spec_run_combinations))]
		else:
			[spec_idx, run] = valid_spec_run_combinations[np.random.randint(0, len(valid_spec_run_combinations))]
	else:
		if len(preferred_spec_run_combinations) > 0:
			[spec_idx, run] = preferred_spec_run_combinations[0]
		else:
			[spec_idx, run] = valid_spec_run_combinations[0]
	csv[spec_idx][run] = 'running '+str(time.time())
	save_list_as_csv(csv, csv_path, filename)
	return spec_idx, run

def open_csv_as_list(path, filename):
	print('ASC // opening', os.getcwd()+path+filename, '('+str(time.time())+')')
	with open(os.getcwd()+path+filename, 'r') as f:
		reader = csv.reader(f)
		csv_array = list(reader)
	for row in range(len(csv_array)):
		formatted_row = csv_array[row][0].split(';')
		csv_array[row] = formatted_row
	return csv_array

def save_list_as_csv(array, path, filename, print_message=False):
	print('ASC // saving', os.getcwd()+path+filename, '('+str(time.time())+')')
	with open(os.getcwd()+path+filename, 'w') as csvfile:
		writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		for row in array:
			writer.writerow(row)
	if print_message:
		print('[MESSAGE] csv saved: %s' %(path+filename))

def create_scheduler_csv(spec_list, n_runs, experiment_name, path_relative='/2_scheduling/'):
	top_row = [experiment_name]
	for run in range(n_runs):
		top_row.append('run_'+str(run+1))
	csv_array = []
	csv_array.append(top_row)
	for spec in spec_list:
		next_row = [spec]
		for run in range(n_runs):
			next_row.append('')
		csv_array.append(next_row)
	save_list_as_csv(csv_array, path_relative, experiment_name+'.csv') # needs fixing

################################################################################
### SETTINGS ###################################################################
################################################################################

def get_settings():
	scheduling_subfolder = '/2_scheduling/'
	experiment_path = '/3_output_cifar/'
	experiment_name = 'ASC_test_normalization'                                    # change when swapping between deployed and development folders
	spec_list = ['smcn_adam_c10_ABU_N_TERIS',
				 'smcn_adam_c10_ABU_P_TERIS',
				 'smcn_adam_c10_ABU_A_TERIS',
				 'smcn_adam_c10_ABU_S_TERIS']
	n_runs = 5
	gridjob_command = 'qsub 2_scheduling/advanced_scheduler_gridjob.sh'
	return scheduling_subfolder, experiment_path, experiment_name, spec_list, n_runs, gridjob_command

################################################################################
### SCRIPT #####################################################################
################################################################################

if __name__ == '__main__':

	# GET SETTINGS
	scheduling_subfolder, experiment_path, experiment_name, spec_list, n_runs, gridjob_command = get_settings()

	# UPDATE CSV AND LOOK UP SPECS FOR NEXT RUN
	with FileLock(os.getcwd()+scheduling_subfolder+experiment_name+'.csv.lock'):
		gridjob_active = (len(sys.argv) > 1)
		gridjob_inactive = not gridjob_active
		n_unfinished_runs = csv_update(scheduling_subfolder, experiment_name+'.csv', experiment_path, experiment_name, mark_running_as_incomplete=gridjob_inactive)
		if n_unfinished_runs > 0:
			spec_csv_idx, run = csv_lookup_spec_run(scheduling_subfolder, experiment_name+'.csv')

	# AT THE END OF GRIDJOB, RESCHEDULE GRIDJOB
	if len(sys.argv) > 1:
		if int(sys.argv[1]) == 5 and n_unfinished_runs > 0:
			os.system(gridjob_command)

	if spec_csv_idx == -1 and run == -1:
		time.sleep(300.0)
		with FileLock(os.getcwd()+scheduling_subfolder+experiment_name+'.csv.lock'):
			gridjob_active = (len(sys.argv) > 1)
			gridjob_inactive = not gridjob_active
			n_unfinished_runs = csv_update(scheduling_subfolder, experiment_name+'.csv', experiment_path, experiment_name, mark_running_as_incomplete=gridjob_inactive)
			if n_unfinished_runs > 0:
				spec_csv_idx, run = csv_lookup_spec_run(scheduling_subfolder, experiment_name+'.csv')

	if n_unfinished_runs == 0:
		print('[MESSAGE] Experiment already complete. Please check if %s.csv shows every run to be finished.' %(scheduling_subfolder+experiment_name))
	elif run > 0:
		print('\n=================================================================================================================================================================================')
		print('SCHEDULING: EXPERIMENT "%s" / SPEC "%s" / RUN "%i"' %(experiment_name, spec_list[spec_csv_idx-1], run))
		print('=================================================================================================================================================================================\n')
		os.system("nvidia-smi")
		subprocess.run(get_command(experiment_name, spec_list[spec_csv_idx-1], run), shell=True)
