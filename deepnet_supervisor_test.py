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
#
# ################################################################################
# os.system("nvidia-smi")
# command = "python3 "          				+ 'deepnet_main.py' + \
# 		  " -experiment_name="  			+ 'debugging' + \
# 		  " -spec_name="        			+ 'smcn_cifar10' + \
# 		  " -run="              			+ str(1) + \
# 		  " -task="             			+ 'cifar10' + \
# 		  " -network="          			+ 'smcn' + \
# 		  " -mode="             			+ 'training' + \
# 		  " -n_minibatches="    			+ str(1000) + \
# 		  " -minibatch_size="   			+ str(256) + \
# 		  " -dropout_keep_probs="   		+ '[0.5]' + \
# 		  " -dropout_keep_probs_inference="	+ '[1.0]' + \
# 		  " -optimizer="            		+ 'Adam' + \
# 		  " -lr="               			+ str(0.001) + \
# 		  " -lr_step_ep="           		+ '[]' + \
# 		  " -lr_step_multi="        		+ '[]' + \
# 		  " -use_wd="        				+ str(False) + \
# 		  " -wd_lambda="        			+ str(0.0) + \
# 		  " -training_schedule="			+ 'epochs' + \
# 		  " -create_val_set="				+ str(True) + \
# 		  " -val_set_fraction="				+ str(0.05) + \
# 		  " -af_set="           			+ '1_jelu' +\
# 		  " -af_weights_init="  			+ 'default' + \
# 		  " -blend_trainable="  			+ str(False) + \
# 		  " -blend_mode="       			+ 'unrestricted' + \
# 		  " -swish_beta_trainable=" 		+ str(False)
# subprocess.run(command, shell=True)
#
# ################################################################################
# os.system("nvidia-smi")
# command = "python3 "          				+ 'deepnet_main.py' + \
# 		  " -experiment_name="  			+ 'debugging' + \
# 		  " -spec_name="        			+ 'smcn_cifar100' + \
# 		  " -run="              			+ str(2) + \
# 		  " -task="             			+ 'cifar100' + \
# 		  " -network="          			+ 'smcn' + \
# 		  " -mode="             			+ 'training' + \
# 		  " -n_minibatches="    			+ str(1000) + \
# 		  " -minibatch_size="   			+ str(256) + \
# 		  " -dropout_keep_probs="   		+ '[.5]' + \
# 		  " -dropout_keep_probs_inference="	+ '[1.0]' + \
# 		  " -optimizer="            		+ 'Adam' + \
# 		  " -lr="               			+ str(0.001) + \
# 		  " -lr_step_ep="           		+ '[]' + \
# 		  " -lr_step_multi="        		+ '[]' + \
# 		  " -use_wd="        				+ str(False) + \
# 		  " -wd_lambda="        			+ str(0.0) + \
# 		  " -training_schedule="			+ 'epochs' + \
# 		  " -create_val_set="				+ str(False) + \
# 		  " -val_set_fraction="				+ str(0.05) + \
# 		  " -af_set="           			+ '1_jelu' +\
# 		  " -af_weights_init="  			+ 'default' + \
# 		  " -blend_trainable="  			+ str(False) + \
# 		  " -blend_mode="       			+ 'unrestricted' + \
# 		  " -swish_beta_trainable=" 		+ str(False)
# subprocess.run(command, shell=True)
#
# ################################################################################
# os.system("nvidia-smi")
# command = "python3 "          				+ 'deepnet_main.py' + \
# 		  " -experiment_name="  			+ 'debugging' + \
# 		  " -spec_name="        			+ 'smcn_cifar10' + \
# 		  " -run="              			+ str(1) + \
# 		  " -task="             			+ 'cifar10' + \
# 		  " -network="          			+ 'smcn' + \
# 		  " -mode="             			+ 'training' + \
# 		  " -n_minibatches="    			+ str(1000) + \
# 		  " -minibatch_size="   			+ str(256) + \
# 		  " -dropout_keep_probs="   		+ '[.5]' + \
# 		  " -dropout_keep_probs_inference="	+ '[1.0]' + \
# 		  " -optimizer="            		+ 'Adam' + \
# 		  " -lr="               			+ str(0.001) + \
# 		  " -lr_step_ep="           		+ '[]' + \
# 		  " -lr_step_multi="        		+ '[]' + \
# 		  " -use_wd="        				+ str(True) + \
# 		  " -wd_lambda="        			+ str(0.001) + \
# 		  " -training_schedule="			+ 'epochs' + \
# 		  " -create_val_set="				+ str(False) + \
# 		  " -val_set_fraction="				+ str(0.05) + \
# 		  " -af_set="           			+ '5_blend5_swish' +\
# 		  " -af_weights_init="  			+ 'default' + \
# 		  " -blend_trainable="  			+ str(True) + \
# 		  " -blend_mode="       			+ 'unrestricted' + \
# 		  " -swish_beta_trainable=" 		+ str(False)
# subprocess.run(command, shell=True)


################################################################################
os.system("nvidia-smi")
command = "python3 "          				+ 'deepnet_main.py' + \
		  " -experiment_name="  			+ 'debugging' + \
		  " -spec_name="        			+ 'allcnnc_cifar10' + \
		  " -run="              			+ str(1) + \
		  " -task="             			+ 'cifar10' + \
		  " -network="          			+ 'allcnnc' + \
		  " -mode="             			+ 'training' + \
		  " -n_minibatches="    			+ str(1000) + \
		  " -minibatch_size="   			+ str(256) + \
		  " -dropout_keep_probs="   		+ '[0.8, 0.5, 0.5]' + \
		  " -dropout_keep_probs_inference="	+ '[1.0, 1.0, 1.0]' + \
		  " -optimizer="            		+ 'Adam' + \
		  " -lr="               			+ str(0.001) + \
		  " -lr_step_ep="           		+ '[]' + \
		  " -lr_step_multi="        		+ '[]' + \
		  " -use_wd="        				+ str(True) + \
		  " -wd_lambda="        			+ str(0.001) + \
		  " -training_schedule="			+ 'epochs' + \
		  " -create_val_set="				+ str(True) + \
		  " -val_set_fraction="				+ str(0.05) + \
		  " -af_set="           			+ '5_blend5_swish' +\
		  " -af_weights_init="  			+ 'default' + \
		  " -blend_trainable="  			+ str(True) + \
		  " -blend_mode="       			+ 'unrestricted' + \
		  " -swish_beta_trainable=" 		+ str(False)
subprocess.run(command, shell=True)

# ################################################################################
# os.system("nvidia-smi")
# command = "python3 "          				+ 'deepnet_main.py' + \
# 		  " -experiment_name="  			+ 'debugging' + \
# 		  " -spec_name="        			+ 'allcnnc_cifar10' + \
# 		  " -run="              			+ str(1) + \
# 		  " -task="             			+ 'cifar10' + \
# 		  " -network="          			+ 'allcnnc' + \
# 		  " -mode="             			+ 'training' + \
# 		  " -n_minibatches="    			+ str(1000) + \
# 		  " -minibatch_size="   			+ str(256) + \
# 		  " -dropout_keep_probs="   		+ '[0.8,0.5,0.5]' + \
# 		  " -dropout_keep_probs_inference="	+ '[1.0,1.0,1.0]' + \
# 		  " -optimizer="            		+ 'Momentum' + \
# 		  " -lr="               			+ str(0.01) + \
# 		  " -lr_step_ep="           		+ '[200,250,300]' + \
# 		  " -lr_step_multi="        		+ '[0.1,0.01,0.001]' + \
# 		  " -use_wd="        				+ str(True) + \
# 		  " -wd_lambda="        			+ str(0.001) + \
# 		  " -training_schedule="			+ 'epochs' + \
# 		  " -create_val_set="				+ str(False) + \
# 		  " -val_set_fraction="				+ str(0.05) + \
# 		  " -af_set="           			+ '1_jelu' +\
# 		  " -af_weights_init="  			+ 'default' + \
# 		  " -blend_trainable="  			+ str(False) + \
# 		  " -blend_mode="       			+ 'unrestricted' + \
# 		  " -swish_beta_trainable=" 		+ str(False)
# subprocess.run(command, shell=True)
