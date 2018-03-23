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

TASK_ID = sys.argv[1]
# path_to_main = '/net/store/ni/users/lsuetfel/squeezeMI/'

################################################################ 1-10 RELU #####
if int(TASK_ID) < 11:
	RUN = int(TASK_ID)-0
	os.system("nvidia-smi")
	command = "python3 "          		+ 'deepnet_main.py' + \
			  " -experiment_name="  	+ 'SBF_3c_sneakpeak' + \
			  " -spec_name="        	+ 'blend2_smax' + \
			  " -run="              	+ str(RUN) + \
			  " -task="             	+ 'cifar10' + \
			  " -network="          	+ 'smcn' + \
			  " -mode="             	+ 'training' + \
			  " -n_minibatches="    	+ str(10000) + \
			  " -minibatch_size="   	+ str(256) + \
			  " -optimizer="            + 'Adam' + \
			  " -lr="               	+ str(0.001) + \
			  " -training_schedule="	+ 'epochs' + \
			  " -af_set="           	+ '2_blend2' +\
			  " -af_weights_init="  	+ 'default' + \
			  " -blend_trainable="  	+ str(True) + \
			  " -blend_mode="       	+ 'softmaxed' + \
			  " -swish_beta_trainable=" + str(False)
	subprocess.run(command, shell=True)

############################################################### 11-20 SELU #####
elif int(TASK_ID) < 21:
	RUN = int(TASK_ID)-10
	os.system("nvidia-smi")
	command = "python3 "          		+ 'deepnet_main.py' + \
			  " -experiment_name="  	+ 'SBF_3c_sneakpeak' + \
			  " -spec_name="        	+ 'blend3_smax' + \
			  " -run="              	+ str(RUN) + \
			  " -task="             	+ 'cifar10' + \
			  " -network="          	+ 'smcn' + \
			  " -mode="             	+ 'training' + \
			  " -n_minibatches="    	+ str(10000) + \
			  " -minibatch_size="   	+ str(256) + \
			  " -optimizer="            + 'Adam' + \
			  " -lr="               	+ str(0.001) + \
			  " -training_schedule="	+ 'epochs' + \
			  " -af_set="           	+ '3_blend3' +\
			  " -af_weights_init="  	+ 'default' + \
			  " -blend_trainable="  	+ str(True) + \
			  " -blend_mode="       	+ 'softmaxed' + \
			  " -swish_beta_trainable=" + str(False)
	subprocess.run(command, shell=True)

################################################################ 21-30 ELU #####
elif int(TASK_ID) < 31:
	RUN = int(TASK_ID)-20
	os.system("nvidia-smi")
	command = "python3 "          		+ 'deepnet_main.py' + \
			  " -experiment_name="  	+ 'SBF_3c_sneakpeak' + \
			  " -spec_name="        	+ 'blend5_smax' + \
			  " -run="              	+ str(RUN) + \
			  " -task="             	+ 'cifar10' + \
			  " -network="          	+ 'smcn' + \
			  " -mode="             	+ 'training' + \
			  " -n_minibatches="    	+ str(10000) + \
			  " -minibatch_size="   	+ str(256) + \
			  " -optimizer="            + 'Adam' + \
			  " -lr="               	+ str(0.001) + \
			  " -training_schedule="	+ 'epochs' + \
			  " -af_set="           	+ '5_blend5_swish' +\
			  " -af_weights_init="  	+ 'default' + \
			  " -blend_trainable="  	+ str(True) + \
			  " -blend_mode="       	+ 'softmaxed' + \
			  " -swish_beta_trainable=" + str(True)
	subprocess.run(command, shell=True)

########################################################### 31-40 SOFTPLUS #####
elif int(TASK_ID) < 41:
	RUN = int(TASK_ID)-30
	os.system("nvidia-smi")
	command = "python3 "          		+ 'deepnet_main.py' + \
			  " -experiment_name="  	+ 'SBF_3c_sneakpeak' + \
			  " -spec_name="        	+ 'blend7_smax' + \
			  " -run="              	+ str(RUN) + \
			  " -task="             	+ 'cifar10' + \
			  " -network="          	+ 'smcn' + \
			  " -mode="             	+ 'training' + \
			  " -n_minibatches="    	+ str(10000) + \
			  " -minibatch_size="   	+ str(256) + \
			  " -optimizer="            + 'Adam' + \
			  " -lr="               	+ str(0.001) + \
			  " -training_schedule="	+ 'epochs' + \
			  " -af_set="           	+ '7_blend7' +\
			  " -af_weights_init="  	+ 'default' + \
			  " -blend_trainable="  	+ str(True) + \
			  " -blend_mode="       	+ 'softmaxed' + \
			  " -swish_beta_trainable=" + str(False)
	subprocess.run(command, shell=True)

########################################################### 41-50 SOFTSIGN #####
elif int(TASK_ID) < 51:
	RUN = int(TASK_ID)-40
	os.system("nvidia-smi")
	command = "python3 "          		+ 'deepnet_main.py' + \
			  " -experiment_name="  	+ 'SBF_3c_sneakpeak' + \
			  " -spec_name="        	+ 'blend9_smax' + \
			  " -run="              	+ str(RUN) + \
			  " -task="             	+ 'cifar10' + \
			  " -network="          	+ 'smcn' + \
			  " -mode="             	+ 'training' + \
			  " -n_minibatches="    	+ str(10000) + \
			  " -minibatch_size="   	+ str(256) + \
			  " -optimizer="            + 'Adam' + \
			  " -lr="               	+ str(0.001) + \
			  " -training_schedule="	+ 'epochs' + \
			  " -af_set="           	+ '9_blend9_swish' +\
			  " -af_weights_init="  	+ 'default' + \
			  " -blend_trainable="  	+ str(True) + \
			  " -blend_mode="       	+ 'softmaxed' + \
			  " -swish_beta_trainable=" + str(True)
	subprocess.run(command, shell=True)
