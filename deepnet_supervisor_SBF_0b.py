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

if int(TASK_ID) < 11:
	RUN = int(TASK_ID)-0
	os.system("nvidia-smi")
	command = "python3 "          		+ 'deepnet_main.py' + \
			  " -experiment_name="  	+ 'SBF_0bcd' + \
			  " -spec_name="        	+ 'memorytest_1900M' + \
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
			  " -blend_mode="       	+ 'unrestricted' + \
			  " -swish_beta_trainable=" + str(True)
	subprocess.run(command, shell=True)
