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

for run in range(5):
	os.system("nvidia-smi")
	command = "python3 "          				+ 'deepnet_main.py' + \
			  " -experiment_name "  			+ 'debug_new_version' + \
			  " -spec_name "        			+ 'spec_one' + \
			  " -run "              			+ '8' + \
			  " -task="             			+ 'cifar10' + \
			  " -preprocessing="				+ 'ztrans' +\
			  " -network="          			+ 'smcn' + \
			  " -mode "             			+ 'training' + \
			  " -n_minibatches "    			+ '20000' + \
			  " -minibatch_size "   			+ '256' + \
			  " -dropout_keep_probs "   		+ '0.5' + \
			  " -dropout_keep_probs_inference "	+ '1.0' + \
			  " -optimizer "            		+ 'Adam' + \
			  " -lr "               			+ '0.001' + \
			  " -lr_step_ep "           		+ '0' + \
			  " -lr_step_multi "        		+ '1' + \
			  " -use_wd "        				+ 'False' + \
			  " -wd_lambda "        			+ '0.01' + \
			  " -create_val_set "				+ 'True' + \
			  " -val_set_fraction "				+ '0.05' + \
			  " -af_set "           			+ '1_relu' +\
			  " -af_weights_init "  			+ 'default' + \
			  " -load_af_weights_from "  		+ 'none' + \
			  " -norm_blendw_at_init "  		+ 'False' + \
			  " -safe_af_ws_n "  				+ '10' + \
			  " -safe_all_ws_n "  				+ '2' + \
			  " -blend_trainable "  			+ 'True' + \
			  " -blend_mode "       			+ 'unrestricted' + \
			  " -swish_beta_trainable " 		+ 'False' + \
			  " -walltime "						+ '12' + \
			  " -create_checkpoints "			+ 'True' + \
			  " -epochs_between_checkpoints "	+ '8' + \
			  " -save_af_weights_at_test_mb "	+ 'True' + \
			  " -save_all_weights_at_test_mb "	+ 'True' + \
			  " -create_lc_on_the_fly "			+ 'True'
	subprocess.run(command, shell=True)
