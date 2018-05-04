"""
Script to manage the schedule / experiments. Controls repetitive script calls
and paricular parameter settings.

@authors: Leon Sütfeld, Flemming Brieger
"""

import time
import os
import sys
import subprocess

"""
################################################################################
### RULES TO CONSTRUCT "AF_SET" ################################################
################################################################################

- must start with the number of AFs, then "_", then the set name
- must contain "swish" if swish AF is used at all
- must contain "scaled" if the scaled version of the respective AF(s) is to be used
- set name is the AFs name (in lowercase) or the name of the AF blend set, e.g., "blendSF7" or "blend9_swish"
- "elu" is called "jelu" ("just elu") to differentiate it from relu and selu
- "lu" (linear unit) is called "linu" for the same reason
- "swish" alone is called "jswish" ("just swish")
- [EXAMPLE] af_set='1_relu'
- [EXAMPLE] af_set='1_jswish'
- [EXAMPLE] af_set='1_scaled_jelu'
- [EXAMPLE] af_set='9_blend9_siwsh_scaled'

################################################################################
################################################################################
################################################################################
"""

### RUN THIS ONLY ON RIGHTY AND TAKE RIGHTY OFF THE GRID FOR THIS -- COMPUTE TIME TEST ###

TASK_ID = int(sys.argv[1])
TASK_ID += 9

################################################################################
if TASK_ID in [1,4,7]:
	RUN = TASK_ID
	os.system("nvidia-smi")
	command = "python3 "          				+ 'deepnet_main.py' + \
			  " -experiment_name "  			+ 'SBF_10' + \
			  " -spec_name "        			+ 'relu_compute_time_test' + \
			  " -run "              			+ str(RUN) + \
			  " -task="             			+ 'cifar10' + \
			  " -preprocessing="				+ 'ztrans' +\
			  " -network="          			+ 'smcn' + \
			  " -mode "             			+ 'training' + \
			  " -n_minibatches "    			+ '10000' + \
			  " -minibatch_size "   			+ '256' + \
			  " -dropout_keep_probs "   		+ '0.5' + \
			  " -dropout_keep_probs_inference "	+ '1.0' + \
			  " -optimizer "            		+ 'Adam' + \
			  " -lr "               			+ '0.001' + \
			  " -lr_step_ep "           		+ '0' + \
			  " -lr_step_multi "        		+ '1' + \
			  " -use_wd "        				+ 'False' + \
			  " -wd_lambda "        			+ '0.01' + \
			  " -training_schedule "			+ 'epochs' + \
			  " -create_val_set "				+ 'False' + \
			  " -val_set_fraction "				+ '0.0' + \
			  " -af_set "           			+ '1_relu' +\
			  " -af_weights_init "  			+ 'default' + \
			  " -load_af_weights_from "  		+ 'none' + \
			  " -norm_blendw_at_init "  		+ 'False' + \
			  " -safe_af_ws_n "  				+ '0' + \
			  " -safe_all_ws_n "  				+ '0' + \
			  " -blend_trainable "  			+ 'False' + \
			  " -blend_mode "       			+ 'unrestricted' + \
			  " -swish_beta_trainable " 		+ 'False'
	subprocess.run(command, shell=True)

################################################################################
elif TASK_ID in [2,5,8]:
	RUN = TASK_ID
	os.system("nvidia-smi")
	command = "python3 "          				+ 'deepnet_main.py' + \
			  " -experiment_name "  			+ 'SBF_10' + \
			  " -spec_name "        			+ 'adaptive_relu_compute_time_test' + \
			  " -run "              			+ str(RUN) + \
			  " -task="             			+ 'cifar10' + \
			  " -preprocessing="				+ 'ztrans' +\
			  " -network="          			+ 'smcn' + \
			  " -mode "             			+ 'training' + \
			  " -n_minibatches "    			+ '10000' + \
			  " -minibatch_size "   			+ '256' + \
			  " -dropout_keep_probs "   		+ '0.5' + \
			  " -dropout_keep_probs_inference "	+ '1.0' + \
			  " -optimizer "            		+ 'Adam' + \
			  " -lr "               			+ '0.001' + \
			  " -lr_step_ep "           		+ '0' + \
			  " -lr_step_multi "        		+ '1' + \
			  " -use_wd "        				+ 'False' + \
			  " -wd_lambda "        			+ '0.01' + \
			  " -training_schedule "			+ 'epochs' + \
			  " -create_val_set "				+ 'False' + \
			  " -val_set_fraction "				+ '0.0' + \
			  " -af_set "           			+ '1_relu' +\
			  " -af_weights_init "  			+ 'default' + \
			  " -load_af_weights_from "  		+ 'none' + \
			  " -norm_blendw_at_init "  		+ 'False' + \
			  " -safe_af_ws_n "  				+ '0' + \
			  " -safe_all_ws_n "  				+ '0' + \
			  " -blend_trainable "  			+ 'True' + \
			  " -blend_mode "       			+ 'unrestricted' + \
			  " -swish_beta_trainable " 		+ 'False'
	subprocess.run(command, shell=True)

################################################################################
elif TASK_ID in [3,6,9]:
	RUN = TASK_ID
	os.system("nvidia-smi")
	command = "python3 "          				+ 'deepnet_main.py' + \
			  " -experiment_name "  			+ 'SBF_10' + \
			  " -spec_name "        			+ 'blend5_unrest_compute_time_test' + \
			  " -run "              			+ str(RUN) + \
			  " -task="             			+ 'cifar10' + \
			  " -preprocessing="				+ 'ztrans' +\
			  " -network="          			+ 'smcn' + \
			  " -mode "             			+ 'training' + \
			  " -n_minibatches "    			+ '10000' + \
			  " -minibatch_size "   			+ '256' + \
			  " -dropout_keep_probs "   		+ '0.5' + \
			  " -dropout_keep_probs_inference "	+ '1.0' + \
			  " -optimizer "            		+ 'Adam' + \
			  " -lr "               			+ '0.001' + \
			  " -lr_step_ep "           		+ '0' + \
			  " -lr_step_multi "        		+ '1' + \
			  " -use_wd "        				+ 'False' + \
			  " -wd_lambda "        			+ '0.01' + \
			  " -training_schedule "			+ 'epochs' + \
			  " -create_val_set "				+ 'False' + \
			  " -val_set_fraction "				+ '0.0' + \
			  " -af_set "           			+ '5_blend5_swish' +\
			  " -af_weights_init "  			+ 'default' + \
			  " -load_af_weights_from "  		+ 'none' + \
			  " -norm_blendw_at_init "  		+ 'False' + \
			  " -safe_af_ws_n "  				+ '0' + \
			  " -safe_all_ws_n "  				+ '0' + \
			  " -blend_trainable "  			+ 'True' + \
			  " -blend_mode "       			+ 'unrestricted' + \
			  " -swish_beta_trainable " 		+ 'True'
	subprocess.run(command, shell=True)

################################################################################
elif TASK_ID in [10,13,16]:
	RUN = TASK_ID
	os.system("nvidia-smi")
	command = "python3 "          				+ 'deepnet_main.py' + \
			  " -experiment_name "  			+ 'SBF_10' + \
			  " -spec_name "        			+ 'smcnDeep_relu_compute_time_test' + \
			  " -run "              			+ str(RUN) + \
			  " -task="             			+ 'cifar10' + \
			  " -preprocessing="				+ 'ztrans' +\
			  " -network="          			+ 'smcnDeep' + \
			  " -mode "             			+ 'training' + \
			  " -n_minibatches "    			+ '10000' + \
			  " -minibatch_size "   			+ '256' + \
			  " -dropout_keep_probs "   		+ '0.5' + \
			  " -dropout_keep_probs_inference "	+ '1.0' + \
			  " -optimizer "            		+ 'Adam' + \
			  " -lr "               			+ '0.001' + \
			  " -lr_step_ep "           		+ '0' + \
			  " -lr_step_multi "        		+ '1' + \
			  " -use_wd "        				+ 'False' + \
			  " -wd_lambda "        			+ '0.01' + \
			  " -training_schedule "			+ 'epochs' + \
			  " -create_val_set "				+ 'False' + \
			  " -val_set_fraction "				+ '0.0' + \
			  " -af_set "           			+ '1_relu' +\
			  " -af_weights_init "  			+ 'default' + \
			  " -load_af_weights_from "  		+ 'none' + \
			  " -norm_blendw_at_init "  		+ 'False' + \
			  " -safe_af_ws_n "  				+ '0' + \
			  " -safe_all_ws_n "  				+ '0' + \
			  " -blend_trainable "  			+ 'False' + \
			  " -blend_mode "       			+ 'unrestricted' + \
			  " -swish_beta_trainable " 		+ 'False'
	subprocess.run(command, shell=True)

################################################################################
elif TASK_ID in [11,14,17]:
	RUN = TASK_ID
	os.system("nvidia-smi")
	command = "python3 "          				+ 'deepnet_main.py' + \
			  " -experiment_name "  			+ 'SBF_10' + \
			  " -spec_name "        			+ 'relu_momentum_compute_time_test' + \
			  " -run "              			+ str(RUN) + \
			  " -task="             			+ 'cifar10' + \
			  " -preprocessing="				+ 'ztrans' +\
			  " -network="          			+ 'smcn' + \
			  " -mode "             			+ 'training' + \
			  " -n_minibatches "    			+ '10000' + \
			  " -minibatch_size "   			+ '256' + \
			  " -dropout_keep_probs "   		+ '0.5' + \
			  " -dropout_keep_probs_inference "	+ '1.0' + \
			  " -optimizer "            		+ 'Momentum' + \
			  " -lr "               			+ '0.01' + \
			  " -lr_step_ep "           		+ '150 200 250' + \
			  " -lr_step_multi "        		+ '0.1 0.01 0.001' + \
			  " -use_wd "        				+ 'False' + \
			  " -wd_lambda "        			+ '0.01' + \
			  " -training_schedule "			+ 'epochs' + \
			  " -create_val_set "				+ 'False' + \
			  " -val_set_fraction "				+ '0.0' + \
			  " -af_set "           			+ '1_relu' +\
			  " -af_weights_init "  			+ 'default' + \
			  " -load_af_weights_from "  		+ 'none' + \
			  " -norm_blendw_at_init "  		+ 'False' + \
			  " -safe_af_ws_n "  				+ '0' + \
			  " -safe_all_ws_n "  				+ '0' + \
			  " -blend_trainable "  			+ 'False' + \
			  " -blend_mode "       			+ 'unrestricted' + \
			  " -swish_beta_trainable " 		+ 'False'
	subprocess.run(command, shell=True)

################################################################################
elif TASK_ID in [12,15,18]:
	RUN = TASK_ID
	os.system("nvidia-smi")
	command = "python3 "          				+ 'deepnet_main.py' + \
			  " -experiment_name "  			+ 'SBF_10' + \
			  " -spec_name "        			+ 'smcnDeep_blend5_unrest_momentum_compute_time_test' + \
			  " -run "              			+ str(RUN) + \
			  " -task="             			+ 'cifar10' + \
			  " -preprocessing="				+ 'ztrans' +\
			  " -network="          			+ 'smcnDeep' + \
			  " -mode "             			+ 'training' + \
			  " -n_minibatches "    			+ '10000' + \
			  " -minibatch_size "   			+ '256' + \
			  " -dropout_keep_probs "   		+ '0.5' + \
			  " -dropout_keep_probs_inference "	+ '1.0' + \
			  " -optimizer "            		+ 'Momentum' + \
			  " -lr "               			+ '0.01' + \
			  " -lr_step_ep "           		+ '150 200 250' + \
			  " -lr_step_multi "        		+ '0.1 0.01 0.001' + \
			  " -use_wd "        				+ 'False' + \
			  " -wd_lambda "        			+ '0.01' + \
			  " -training_schedule "			+ 'epochs' + \
			  " -create_val_set "				+ 'False' + \
			  " -val_set_fraction "				+ '0.0' + \
			  " -af_set "           			+ '5_blend5_swish' +\
			  " -af_weights_init "  			+ 'default' + \
			  " -load_af_weights_from "  		+ 'none' + \
			  " -norm_blendw_at_init "  		+ 'False' + \
			  " -safe_af_ws_n "  				+ '0' + \
			  " -safe_all_ws_n "  				+ '0' + \
			  " -blend_trainable "  			+ 'True' + \
			  " -blend_mode "       			+ 'unrestricted' + \
			  " -swish_beta_trainable " 		+ 'True'
	subprocess.run(command, shell=True)
