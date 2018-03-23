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
			  " -experiment_name="  	+ 'SBF_0' + \
			  " -spec_name="        	+ 'smcn_blend2_swish' + \
			  " -run="              	+ str(RUN) + \
			  " -task="             	+ 'cifar10' + \
			  " -network="          	+ 'smcn' + \
			  " -mode="             	+ 'training' + \
			  " -n_minibatches="    	+ str(6000) + \
			  " -minibatch_size="   	+ str(256) + \
			  " -optimizer="            + 'Adam' + \
			  " -lr="               	+ str(0.001) + \
			  " -training_schedule="	+ 'epochs' + \
			  " -af_set="           	+ '2_blend2_swish' +\
			  " -af_weights_init="  	+ 'default' + \
			  " -blend_trainable="  	+ str(False) + \
			  " -blend_mode="       	+ 'unrestricted' + \
			  " -swish_beta_trainable=" + str(False)
	subprocess.run(command, shell=True)

elif int(TASK_ID) < 21:
	RUN = int(TASK_ID)-10
	os.system("nvidia-smi")
	command = "python3 "          		+ 'deepnet_main.py' + \
			  " -experiment_name="  	+ 'SBF_0' + \
			  " -spec_name="        	+ 'smcn_blend4_swish' + \
			  " -run="              	+ str(RUN) + \
			  " -task="             	+ 'cifar10' + \
			  " -network="          	+ 'smcn' + \
			  " -mode="             	+ 'training' + \
			  " -n_minibatches="    	+ str(6000) + \
			  " -minibatch_size="   	+ str(256) + \
			  " -optimizer="            + 'Adam' + \
			  " -lr="               	+ str(0.001) + \
			  " -training_schedule="	+ 'epochs' + \
			  " -af_set="           	+ '4_blend4_swish' +\
			  " -af_weights_init="  	+ 'default' + \
			  " -blend_trainable="  	+ str(False) + \
			  " -blend_mode="       	+ 'unrestricted' + \
			  " -swish_beta_trainable=" + str(False)
	subprocess.run(command, shell=True)

elif int(TASK_ID) < 31:
	RUN = int(TASK_ID)-20
	os.system("nvidia-smi")
	command = "python3 "          		+ 'deepnet_main.py' + \
			  " -experiment_name="  	+ 'SBF_0' + \
			  " -spec_name="        	+ 'smcn_blendSF7' + \
			  " -run="              	+ str(RUN) + \
			  " -task="             	+ 'cifar10' + \
			  " -network="          	+ 'smcn' + \
			  " -mode="             	+ 'training' + \
			  " -n_minibatches="    	+ str(6000) + \
			  " -minibatch_size="   	+ str(256) + \
			  " -optimizer="            + 'Adam' + \
			  " -lr="               	+ str(0.001) + \
			  " -training_schedule="	+ 'epochs' + \
			  " -af_set="           	+ '7_blendSF7' +\
			  " -af_weights_init="  	+ 'default' + \
			  " -blend_trainable="  	+ str(False) + \
			  " -blend_mode="       	+ 'unrestricted' + \
			  " -swish_beta_trainable=" + str(False)
	subprocess.run(command, shell=True)

elif int(TASK_ID) < 41:
	RUN = int(TASK_ID)-30
	os.system("nvidia-smi")
	command = "python3 "          		+ 'deepnet_main.py' + \
			  " -experiment_name="  	+ 'SBF_0' + \
			  " -spec_name="        	+ 'smcn_blend9_swish' + \
			  " -run="              	+ str(RUN) + \
			  " -task="             	+ 'cifar10' + \
			  " -network="          	+ 'smcn' + \
			  " -mode="             	+ 'training' + \
			  " -n_minibatches="    	+ str(6000) + \
			  " -minibatch_size="   	+ str(256) + \
			  " -optimizer="            + 'Adam' + \
			  " -lr="               	+ str(0.001) + \
			  " -training_schedule="	+ 'epochs' + \
			  " -af_set="           	+ '9_blend9_swish' +\
			  " -af_weights_init="  	+ 'default' + \
			  " -blend_trainable="  	+ str(False) + \
			  " -blend_mode="       	+ 'unrestricted' + \
			  " -swish_beta_trainable=" + str(False)
	subprocess.run(command, shell=True)

elif int(TASK_ID) < 51:
	RUN = int(TASK_ID)-40
	os.system("nvidia-smi")
	command = "python3 "          		+ 'deepnet_main.py' + \
			  " -experiment_name="  	+ 'SBF_0' + \
			  " -spec_name="        	+ 'MaxLrn_blend2_swish' + \
			  " -run="              	+ str(RUN) + \
			  " -task="             	+ 'cifar10' + \
			  " -network="          	+ 'smcnMaxLrn' + \
			  " -mode="             	+ 'training' + \
			  " -n_minibatches="    	+ str(6000) + \
			  " -minibatch_size="   	+ str(256) + \
			  " -optimizer="            + 'Adam' + \
			  " -lr="               	+ str(0.001) + \
			  " -training_schedule="	+ 'epochs' + \
			  " -af_set="           	+ '2_blend2_swish' +\
			  " -af_weights_init="  	+ 'default' + \
			  " -blend_trainable="  	+ str(False) + \
			  " -blend_mode="       	+ 'unrestricted' + \
			  " -swish_beta_trainable=" + str(False)
	subprocess.run(command, shell=True)

elif int(TASK_ID) < 61:
	RUN = int(TASK_ID)-50
	os.system("nvidia-smi")
	command = "python3 "          		+ 'deepnet_main.py' + \
			  " -experiment_name="  	+ 'SBF_0' + \
			  " -spec_name="        	+ 'MaxLrn_blend4_swish' + \
			  " -run="              	+ str(RUN) + \
			  " -task="             	+ 'cifar10' + \
			  " -network="          	+ 'smcnMaxLrn' + \
			  " -mode="             	+ 'training' + \
			  " -n_minibatches="    	+ str(6000) + \
			  " -minibatch_size="   	+ str(256) + \
			  " -optimizer="            + 'Adam' + \
			  " -lr="               	+ str(0.001) + \
			  " -training_schedule="	+ 'epochs' + \
			  " -af_set="           	+ '4_blend4_swish' +\
			  " -af_weights_init="  	+ 'default' + \
			  " -blend_trainable="  	+ str(False) + \
			  " -blend_mode="       	+ 'unrestricted' + \
			  " -swish_beta_trainable=" + str(False)
	subprocess.run(command, shell=True)

elif int(TASK_ID) < 71:
	RUN = int(TASK_ID)-60
	os.system("nvidia-smi")
	command = "python3 "          		+ 'deepnet_main.py' + \
			  " -experiment_name="  	+ 'SBF_0' + \
			  " -spec_name="        	+ 'MaxLrn_blendSF7' + \
			  " -run="              	+ str(RUN) + \
			  " -task="             	+ 'cifar10' + \
			  " -network="          	+ 'smcnMaxLrn' + \
			  " -mode="             	+ 'training' + \
			  " -n_minibatches="    	+ str(6000) + \
			  " -minibatch_size="   	+ str(256) + \
			  " -optimizer="            + 'Adam' + \
			  " -lr="               	+ str(0.001) + \
			  " -training_schedule="	+ 'epochs' + \
			  " -af_set="           	+ '7_blendSF7' +\
			  " -af_weights_init="  	+ 'default' + \
			  " -blend_trainable="  	+ str(False) + \
			  " -blend_mode="       	+ 'unrestricted' + \
			  " -swish_beta_trainable=" + str(False)
	subprocess.run(command, shell=True)

elif int(TASK_ID) < 81:
	RUN = int(TASK_ID)-70
	os.system("nvidia-smi")
	command = "python3 "          		+ 'deepnet_main.py' + \
			  " -experiment_name="  	+ 'SBF_0' + \
			  " -spec_name="        	+ 'MaxLrn_blend9_swish' + \
			  " -run="              	+ str(RUN) + \
			  " -task="             	+ 'cifar10' + \
			  " -network="          	+ 'smcnMaxLrn' + \
			  " -mode="             	+ 'training' + \
			  " -n_minibatches="    	+ str(6000) + \
			  " -minibatch_size="   	+ str(256) + \
			  " -optimizer="            + 'Adam' + \
			  " -lr="               	+ str(0.001) + \
			  " -training_schedule="	+ 'epochs' + \
			  " -af_set="           	+ '9_blend9_swish' +\
			  " -af_weights_init="  	+ 'default' + \
			  " -blend_trainable="  	+ str(False) + \
			  " -blend_mode="       	+ 'unrestricted' + \
			  " -swish_beta_trainable=" + str(False)
	subprocess.run(command, shell=True)

# ##############################################################################

elif int(TASK_ID) < 91:
	RUN = int(TASK_ID)-80
	os.system("nvidia-smi")
	command = "python3 "          		+ 'deepnet_main.py' + \
			  " -experiment_name="  	+ 'SBF_0' + \
			  " -spec_name="        	+ 'MaxLrn_relu' + \
			  " -run="              	+ str(RUN) + \
			  " -task="             	+ 'cifar10' + \
			  " -network="          	+ 'smcnMaxLrn' + \
			  " -mode="             	+ 'training' + \
			  " -n_minibatches="    	+ str(6000) + \
			  " -minibatch_size="   	+ str(256) + \
			  " -optimizer="            + 'Adam' + \
			  " -lr="               	+ str(0.001) + \
			  " -training_schedule="	+ 'epochs' + \
			  " -af_set="           	+ '1_relu' +\
			  " -af_weights_init="  	+ 'default' + \
			  " -blend_trainable="  	+ str(False) + \
			  " -blend_mode="       	+ 'unrestricted' + \
			  " -swish_beta_trainable=" + str(False)
	subprocess.run(command, shell=True)

elif int(TASK_ID) < 101:
	RUN = int(TASK_ID)-90
	os.system("nvidia-smi")
	command = "python3 "          		+ 'deepnet_main.py' + \
			  " -experiment_name="  	+ 'SBF_0' + \
			  " -spec_name="        	+ 'Max_relu' + \
			  " -run="              	+ str(RUN) + \
			  " -task="             	+ 'cifar10' + \
			  " -network="          	+ 'smcnMax' + \
			  " -mode="             	+ 'training' + \
			  " -n_minibatches="    	+ str(6000) + \
			  " -minibatch_size="   	+ str(256) + \
			  " -optimizer="            + 'Adam' + \
			  " -lr="               	+ str(0.001) + \
			  " -training_schedule="	+ 'epochs' + \
			  " -af_set="           	+ '1_relu' +\
			  " -af_weights_init="  	+ 'default' + \
			  " -blend_trainable="  	+ str(False) + \
			  " -blend_mode="       	+ 'unrestricted' + \
			  " -swish_beta_trainable=" + str(False)
	subprocess.run(command, shell=True)

elif int(TASK_ID) < 111:
	RUN = int(TASK_ID)-100
	os.system("nvidia-smi")
	command = "python3 "          		+ 'deepnet_main.py' + \
			  " -experiment_name="  	+ 'SBF_0' + \
			  " -spec_name="        	+ 'smcn_relu' + \
			  " -run="              	+ str(RUN) + \
			  " -task="             	+ 'cifar10' + \
			  " -network="          	+ 'smcn' + \
			  " -mode="             	+ 'training' + \
			  " -n_minibatches="    	+ str(6000) + \
			  " -minibatch_size="   	+ str(256) + \
			  " -optimizer="            + 'Adam' + \
			  " -lr="               	+ str(0.001) + \
			  " -training_schedule="	+ 'epochs' + \
			  " -af_set="           	+ '1_relu' +\
			  " -af_weights_init="  	+ 'default' + \
			  " -blend_trainable="  	+ str(False) + \
			  " -blend_mode="       	+ 'unrestricted' + \
			  " -swish_beta_trainable=" + str(False)
	subprocess.run(command, shell=True)

# ##############################################################################

elif int(TASK_ID) < 121:
	RUN = int(TASK_ID)-110
	os.system("nvidia-smi")
	command = "python3 "          		+ 'deepnet_main.py' + \
			  " -experiment_name="  	+ 'SBF_0' + \
			  " -spec_name="        	+ 'MaxLrn_elu' + \
			  " -run="              	+ str(RUN) + \
			  " -task="             	+ 'cifar10' + \
			  " -network="          	+ 'smcnMaxLrn' + \
			  " -mode="             	+ 'training' + \
			  " -n_minibatches="    	+ str(6000) + \
			  " -minibatch_size="   	+ str(256) + \
			  " -optimizer="            + 'Adam' + \
			  " -lr="               	+ str(0.001) + \
			  " -training_schedule="	+ 'epochs' + \
			  " -af_set="           	+ '1_jelu' +\
			  " -af_weights_init="  	+ 'default' + \
			  " -blend_trainable="  	+ str(False) + \
			  " -blend_mode="       	+ 'unrestricted' + \
			  " -swish_beta_trainable=" + str(False)
	subprocess.run(command, shell=True)

elif int(TASK_ID) < 131:
	RUN = int(TASK_ID)-120
	os.system("nvidia-smi")
	command = "python3 "          		+ 'deepnet_main.py' + \
			  " -experiment_name="  	+ 'SBF_0' + \
			  " -spec_name="        	+ 'Max_elu' + \
			  " -run="              	+ str(RUN) + \
			  " -task="             	+ 'cifar10' + \
			  " -network="          	+ 'smcnMax' + \
			  " -mode="             	+ 'training' + \
			  " -n_minibatches="    	+ str(6000) + \
			  " -minibatch_size="   	+ str(256) + \
			  " -optimizer="            + 'Adam' + \
			  " -lr="               	+ str(0.001) + \
			  " -training_schedule="	+ 'epochs' + \
			  " -af_set="           	+ '1_jelu' +\
			  " -af_weights_init="  	+ 'default' + \
			  " -blend_trainable="  	+ str(False) + \
			  " -blend_mode="       	+ 'unrestricted' + \
			  " -swish_beta_trainable=" + str(False)
	subprocess.run(command, shell=True)

elif int(TASK_ID) < 141:
	RUN = int(TASK_ID)-130
	os.system("nvidia-smi")
	command = "python3 "          		+ 'deepnet_main.py' + \
			  " -experiment_name="  	+ 'SBF_0' + \
			  " -spec_name="        	+ 'smcn_elu' + \
			  " -run="              	+ str(RUN) + \
			  " -task="             	+ 'cifar10' + \
			  " -network="          	+ 'smcn' + \
			  " -mode="             	+ 'training' + \
			  " -n_minibatches="    	+ str(6000) + \
			  " -minibatch_size="   	+ str(256) + \
			  " -optimizer="            + 'Adam' + \
			  " -lr="               	+ str(0.001) + \
			  " -training_schedule="	+ 'epochs' + \
			  " -af_set="           	+ '1_jelu' +\
			  " -af_weights_init="  	+ 'default' + \
			  " -blend_trainable="  	+ str(False) + \
			  " -blend_mode="       	+ 'unrestricted' + \
			  " -swish_beta_trainable=" + str(False)
	subprocess.run(command, shell=True)
