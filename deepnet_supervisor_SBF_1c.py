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
			  " -experiment_name="  	+ 'SBF_1c' + \
			  " -spec_name="        	+ 'adaptive_unscaled_relu' + \
			  " -run="              	+ str(RUN) + \
			  " -task="             	+ 'cifar10' + \
			  " -network="          	+ 'smcn' + \
			  " -mode="             	+ 'training' + \
			  " -n_minibatches="    	+ str(10000) + \
			  " -minibatch_size="   	+ str(256) + \
			  " -optimizer="            + 'Adam' + \
			  " -lr="               	+ str(0.001) + \
			  " -training_schedule="	+ 'epochs' + \
			  " -af_set="           	+ '1_relu' +\
			  " -af_weights_init="  	+ 'default' + \
			  " -blend_trainable="  	+ str(True) + \
			  " -blend_mode="       	+ 'unrestricted' + \
			  " -swish_beta_trainable=" + str(False)
	subprocess.run(command, shell=True)

############################################################### 11-20 SELU #####
elif int(TASK_ID) < 21:
	RUN = int(TASK_ID)-10
	os.system("nvidia-smi")
	command = "python3 "          		+ 'deepnet_main.py' + \
			  " -experiment_name="  	+ 'SBF_1c' + \
			  " -spec_name="        	+ 'adaptive_unscaled_selu' + \
			  " -run="              	+ str(RUN) + \
			  " -task="             	+ 'cifar10' + \
			  " -network="          	+ 'smcn' + \
			  " -mode="             	+ 'training' + \
			  " -n_minibatches="    	+ str(10000) + \
			  " -minibatch_size="   	+ str(256) + \
			  " -optimizer="            + 'Adam' + \
			  " -lr="               	+ str(0.001) + \
			  " -training_schedule="	+ 'epochs' + \
			  " -af_set="           	+ '1_selu' +\
			  " -af_weights_init="  	+ 'default' + \
			  " -blend_trainable="  	+ str(True) + \
			  " -blend_mode="       	+ 'unrestricted' + \
			  " -swish_beta_trainable=" + str(False)
	subprocess.run(command, shell=True)

################################################################ 21-30 ELU #####
elif int(TASK_ID) < 31:
	RUN = int(TASK_ID)-20
	os.system("nvidia-smi")
	command = "python3 "          		+ 'deepnet_main.py' + \
			  " -experiment_name="  	+ 'SBF_1c' + \
			  " -spec_name="        	+ 'adaptive_unscaled_elu' + \
			  " -run="              	+ str(RUN) + \
			  " -task="             	+ 'cifar10' + \
			  " -network="          	+ 'smcn' + \
			  " -mode="             	+ 'training' + \
			  " -n_minibatches="    	+ str(10000) + \
			  " -minibatch_size="   	+ str(256) + \
			  " -optimizer="            + 'Adam' + \
			  " -lr="               	+ str(0.001) + \
			  " -training_schedule="	+ 'epochs' + \
			  " -af_set="           	+ '1_jelu' +\
			  " -af_weights_init="  	+ 'default' + \
			  " -blend_trainable="  	+ str(True) + \
			  " -blend_mode="       	+ 'unrestricted' + \
			  " -swish_beta_trainable=" + str(False)
	subprocess.run(command, shell=True)

########################################################### 31-40 SOFTPLUS #####
elif int(TASK_ID) < 41:
	RUN = int(TASK_ID)-30
	os.system("nvidia-smi")
	command = "python3 "          		+ 'deepnet_main.py' + \
			  " -experiment_name="  	+ 'SBF_1c' + \
			  " -spec_name="        	+ 'adaptive_unscaled_softplus' + \
			  " -run="              	+ str(RUN) + \
			  " -task="             	+ 'cifar10' + \
			  " -network="          	+ 'smcn' + \
			  " -mode="             	+ 'training' + \
			  " -n_minibatches="    	+ str(10000) + \
			  " -minibatch_size="   	+ str(256) + \
			  " -optimizer="            + 'Adam' + \
			  " -lr="               	+ str(0.001) + \
			  " -training_schedule="	+ 'epochs' + \
			  " -af_set="           	+ '1_softplus' +\
			  " -af_weights_init="  	+ 'default' + \
			  " -blend_trainable="  	+ str(True) + \
			  " -blend_mode="       	+ 'unrestricted' + \
			  " -swish_beta_trainable=" + str(False)
	subprocess.run(command, shell=True)

########################################################### 41-50 SOFTSIGN #####
elif int(TASK_ID) < 51:
	RUN = int(TASK_ID)-40
	os.system("nvidia-smi")
	command = "python3 "          		+ 'deepnet_main.py' + \
			  " -experiment_name="  	+ 'SBF_1c' + \
			  " -spec_name="        	+ 'adaptive_unscaled_softsign' + \
			  " -run="              	+ str(RUN) + \
			  " -task="             	+ 'cifar10' + \
			  " -network="          	+ 'smcn' + \
			  " -mode="             	+ 'training' + \
			  " -n_minibatches="    	+ str(10000) + \
			  " -minibatch_size="   	+ str(256) + \
			  " -optimizer="            + 'Adam' + \
			  " -lr="               	+ str(0.001) + \
			  " -training_schedule="	+ 'epochs' + \
			  " -af_set="           	+ '1_softsign' +\
			  " -af_weights_init="  	+ 'default' + \
			  " -blend_trainable="  	+ str(True) + \
			  " -blend_mode="       	+ 'unrestricted' + \
			  " -swish_beta_trainable=" + str(False)
	subprocess.run(command, shell=True)

############################################################### 51-60 TANH #####
elif int(TASK_ID) < 61:
	RUN = int(TASK_ID)-50
	os.system("nvidia-smi")
	command = "python3 "          		+ 'deepnet_main.py' + \
			  " -experiment_name="  	+ 'SBF_1c' + \
			  " -spec_name="        	+ 'adaptive_unscaled_tanh' + \
			  " -run="              	+ str(RUN) + \
			  " -task="             	+ 'cifar10' + \
			  " -network="          	+ 'smcn' + \
			  " -mode="             	+ 'training' + \
			  " -n_minibatches="    	+ str(10000) + \
			  " -minibatch_size="   	+ str(256) + \
			  " -optimizer="            + 'Adam' + \
			  " -lr="               	+ str(0.001) + \
			  " -training_schedule="	+ 'epochs' + \
			  " -af_set="           	+ '1_tanh' +\
			  " -af_weights_init="  	+ 'default' + \
			  " -blend_trainable="  	+ str(True) + \
			  " -blend_mode="       	+ 'unrestricted' + \
			  " -swish_beta_trainable=" + str(False)
	subprocess.run(command, shell=True)

############################################################ 61-70 SIGMOID #####
elif int(TASK_ID) < 71:
	RUN = int(TASK_ID)-60
	os.system("nvidia-smi")
	command = "python3 "          		+ 'deepnet_main.py' + \
			  " -experiment_name="  	+ 'SBF_1c' + \
			  " -spec_name="        	+ 'adaptive_unscaled_sigmoid' + \
			  " -run="              	+ str(RUN) + \
			  " -task="             	+ 'cifar10' + \
			  " -network="          	+ 'smcn' + \
			  " -mode="             	+ 'training' + \
			  " -n_minibatches="    	+ str(10000) + \
			  " -minibatch_size="   	+ str(256) + \
			  " -optimizer="            + 'Adam' + \
			  " -lr="               	+ str(0.001) + \
			  " -training_schedule="	+ 'epochs' + \
			  " -af_set="           	+ '1_sigmoid' +\
			  " -af_weights_init="  	+ 'default' + \
			  " -blend_trainable="  	+ str(True) + \
			  " -blend_mode="       	+ 'unrestricted' + \
			  " -swish_beta_trainable=" + str(False)
	subprocess.run(command, shell=True)


############################################################ 71-80 SWISH_1 #####
elif int(TASK_ID) < 81:
	RUN = int(TASK_ID)-70
	os.system("nvidia-smi")
	command = "python3 "          		+ 'deepnet_main.py' + \
			  " -experiment_name="  	+ 'SBF_1c' + \
			  " -spec_name="        	+ 'adaptive_unscaled_swish_1' + \
			  " -run="              	+ str(RUN) + \
			  " -task="             	+ 'cifar10' + \
			  " -network="          	+ 'smcn' + \
			  " -mode="             	+ 'training' + \
			  " -n_minibatches="    	+ str(10000) + \
			  " -minibatch_size="   	+ str(256) + \
			  " -optimizer="            + 'Adam' + \
			  " -lr="               	+ str(0.001) + \
			  " -training_schedule="	+ 'epochs' + \
			  " -af_set="           	+ '1_jswish' +\
			  " -af_weights_init="  	+ 'default' + \
			  " -blend_trainable="  	+ str(True) + \
			  " -blend_mode="       	+ 'unrestricted' + \
			  " -swish_beta_trainable=" + str(False)
	subprocess.run(command, shell=True)

############################################################ 81-90 SWISH_A #####
elif int(TASK_ID) < 91:
	RUN = int(TASK_ID)-80
	os.system("nvidia-smi")
	command = "python3 "          		+ 'deepnet_main.py' + \
			  " -experiment_name="  	+ 'SBF_1c' + \
			  " -spec_name="        	+ 'adaptive_unscaled_swish_adaptive' + \
			  " -run="              	+ str(RUN) + \
			  " -task="             	+ 'cifar10' + \
			  " -network="          	+ 'smcn' + \
			  " -mode="             	+ 'training' + \
			  " -n_minibatches="    	+ str(10000) + \
			  " -minibatch_size="   	+ str(256) + \
			  " -optimizer="            + 'Adam' + \
			  " -lr="               	+ str(0.001) + \
			  " -training_schedule="	+ 'epochs' + \
			  " -af_set="           	+ '1_jswish' +\
			  " -af_weights_init="  	+ 'default' + \
			  " -blend_trainable="  	+ str(True) + \
			  " -blend_mode="       	+ 'unrestricted' + \
			  " -swish_beta_trainable=" + str(True)
	subprocess.run(command, shell=True)

############################################################## 91-100 LINU #####
elif int(TASK_ID) < 101:
	RUN = int(TASK_ID)-90
	os.system("nvidia-smi")
	command = "python3 "          		+ 'deepnet_main.py' + \
			  " -experiment_name="  	+ 'SBF_1c' + \
			  " -spec_name="        	+ 'adaptive_unscaled_linu' + \
			  " -run="              	+ str(RUN) + \
			  " -task="             	+ 'cifar10' + \
			  " -network="          	+ 'smcn' + \
			  " -mode="             	+ 'training' + \
			  " -n_minibatches="    	+ str(10000) + \
			  " -minibatch_size="   	+ str(256) + \
			  " -optimizer="            + 'Adam' + \
			  " -lr="               	+ str(0.001) + \
			  " -training_schedule="	+ 'epochs' + \
			  " -af_set="           	+ '1_linu' +\
			  " -af_weights_init="  	+ 'default' + \
			  " -blend_trainable="  	+ str(True) + \
			  " -blend_mode="       	+ 'unrestricted' + \
			  " -swish_beta_trainable=" + str(False)
	subprocess.run(command, shell=True)
