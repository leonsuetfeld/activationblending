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

TASK_ID = int(sys.argv[1])

################################################################################
if TASK_ID < 101:
	RUN = TASK_ID-0
	os.system("nvidia-smi")
	command = "python3 "          				+ 'deepnet_main.py' + \
			  " -experiment_name "  			+ 'SBF_6a' + \
			  " -spec_name "        			+ 'linu' + \
			  " -run "              			+ str(RUN) + \
			  " -task="             			+ 'cifar100' + \
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
			  " -af_set "           			+ '1_linu' +\
			  " -af_weights_init "  			+ 'default' + \
			  " -blend_trainable "  			+ 'False' + \
			  " -blend_mode "       			+ 'unrestricted' + \
			  " -swish_beta_trainable " 		+ 'False'
	subprocess.run(command, shell=True)

################################################################################
elif TASK_ID < 201:
	RUN = TASK_ID-100
	os.system("nvidia-smi")
	command = "python3 "          				+ 'deepnet_main.py' + \
			  " -experiment_name "  			+ 'SBF_6a' + \
			  " -spec_name "        			+ 'tanh' + \
			  " -run "              			+ str(RUN) + \
			  " -task="             			+ 'cifar100' + \
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
			  " -af_set "           			+ '1_tanh' +\
			  " -af_weights_init "  			+ 'default' + \
			  " -blend_trainable "  			+ 'False' + \
			  " -blend_mode "       			+ 'unrestricted' + \
			  " -swish_beta_trainable " 		+ 'False'
	subprocess.run(command, shell=True)

################################################################################
elif TASK_ID < 301:
	RUN = TASK_ID-200
	os.system("nvidia-smi")
	command = "python3 "          				+ 'deepnet_main.py' + \
			  " -experiment_name "  			+ 'SBF_6a' + \
			  " -spec_name "        			+ 'relu' + \
			  " -run "              			+ str(RUN) + \
			  " -task="             			+ 'cifar100' + \
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
			  " -blend_trainable "  			+ 'False' + \
			  " -blend_mode "       			+ 'unrestricted' + \
			  " -swish_beta_trainable " 		+ 'False'
	subprocess.run(command, shell=True)

################################################################################
elif TASK_ID < 401:
	RUN = TASK_ID-300
	os.system("nvidia-smi")
	command = "python3 "          				+ 'deepnet_main.py' + \
			  " -experiment_name "  			+ 'SBF_6a' + \
			  " -spec_name "        			+ 'elu' + \
			  " -run "              			+ str(RUN) + \
			  " -task="             			+ 'cifar100' + \
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
			  " -af_set "           			+ '1_jelu' +\
			  " -af_weights_init "  			+ 'default' + \
			  " -blend_trainable "  			+ 'False' + \
			  " -blend_mode "       			+ 'unrestricted' + \
			  " -swish_beta_trainable " 		+ 'False'
	subprocess.run(command, shell=True)

################################################################################
elif TASK_ID < 501:
	RUN = TASK_ID-400
	os.system("nvidia-smi")
	command = "python3 "          				+ 'deepnet_main.py' + \
			  " -experiment_name "  			+ 'SBF_6a' + \
			  " -spec_name "        			+ 'selu' + \
			  " -run "              			+ str(RUN) + \
			  " -task="             			+ 'cifar100' + \
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
			  " -af_set "           			+ '1_selu' +\
			  " -af_weights_init "  			+ 'default' + \
			  " -blend_trainable "  			+ 'False' + \
			  " -blend_mode "       			+ 'unrestricted' + \
			  " -swish_beta_trainable " 		+ 'False'
	subprocess.run(command, shell=True)

################################################################################
elif TASK_ID < 601:
	RUN = TASK_ID-500
	os.system("nvidia-smi")
	command = "python3 "          				+ 'deepnet_main.py' + \
			  " -experiment_name "  			+ 'SBF_6a' + \
			  " -spec_name "        			+ 'swish' + \
			  " -run "              			+ str(RUN) + \
			  " -task="             			+ 'cifar100' + \
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
			  " -af_set "           			+ '1_jswish' +\
			  " -af_weights_init "  			+ 'default' + \
			  " -blend_trainable "  			+ 'False' + \
			  " -blend_mode "       			+ 'unrestricted' + \
			  " -swish_beta_trainable " 		+ 'True'
	subprocess.run(command, shell=True)

################################################################################
elif TASK_ID < 701:
	RUN = TASK_ID-600
	os.system("nvidia-smi")
	command = "python3 "          				+ 'deepnet_main.py' + \
			  " -experiment_name "  			+ 'SBF_6a' + \
			  " -spec_name "        			+ 'linu' + \
			  " -run "              			+ str(RUN) + \
			  " -task="             			+ 'cifar100' + \
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
			  " -af_set "           			+ '1_linu' +\
			  " -af_weights_init "  			+ 'default' + \
			  " -blend_trainable "  			+ 'True' + \
			  " -blend_mode "       			+ 'unrestricted' + \
			  " -swish_beta_trainable " 		+ 'False'
	subprocess.run(command, shell=True)

################################################################################
elif TASK_ID < 801:
	RUN = TASK_ID-700
	os.system("nvidia-smi")
	command = "python3 "          				+ 'deepnet_main.py' + \
			  " -experiment_name "  			+ 'SBF_6a' + \
			  " -spec_name "        			+ 'tanh' + \
			  " -run "              			+ str(RUN) + \
			  " -task="             			+ 'cifar100' + \
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
			  " -af_set "           			+ '1_tanh' +\
			  " -af_weights_init "  			+ 'default' + \
			  " -blend_trainable "  			+ 'True' + \
			  " -blend_mode "       			+ 'unrestricted' + \
			  " -swish_beta_trainable " 		+ 'False'
	subprocess.run(command, shell=True)

################################################################################
elif TASK_ID < 901:
	RUN = TASK_ID-800
	os.system("nvidia-smi")
	command = "python3 "          				+ 'deepnet_main.py' + \
			  " -experiment_name "  			+ 'SBF_6a' + \
			  " -spec_name "        			+ 'relu' + \
			  " -run "              			+ str(RUN) + \
			  " -task="             			+ 'cifar100' + \
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
			  " -blend_trainable "  			+ 'True' + \
			  " -blend_mode "       			+ 'unrestricted' + \
			  " -swish_beta_trainable " 		+ 'False'
	subprocess.run(command, shell=True)

################################################################################
elif TASK_ID < 1001:
	RUN = TASK_ID-900
	os.system("nvidia-smi")
	command = "python3 "          				+ 'deepnet_main.py' + \
			  " -experiment_name "  			+ 'SBF_6a' + \
			  " -spec_name "        			+ 'elu' + \
			  " -run "              			+ str(RUN) + \
			  " -task="             			+ 'cifar100' + \
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
			  " -af_set "           			+ '1_jelu' +\
			  " -af_weights_init "  			+ 'default' + \
			  " -blend_trainable "  			+ 'True' + \
			  " -blend_mode "       			+ 'unrestricted' + \
			  " -swish_beta_trainable " 		+ 'False'
	subprocess.run(command, shell=True)

################################################################################
elif TASK_ID < 1101:
	RUN = TASK_ID-1000
	os.system("nvidia-smi")
	command = "python3 "          				+ 'deepnet_main.py' + \
			  " -experiment_name "  			+ 'SBF_6a' + \
			  " -spec_name "        			+ 'selu' + \
			  " -run "              			+ str(RUN) + \
			  " -task="             			+ 'cifar100' + \
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
			  " -af_set "           			+ '1_selu' +\
			  " -af_weights_init "  			+ 'default' + \
			  " -blend_trainable "  			+ 'True' + \
			  " -blend_mode "       			+ 'unrestricted' + \
			  " -swish_beta_trainable " 		+ 'False'
	subprocess.run(command, shell=True)

################################################################################
elif TASK_ID < 1201:
	RUN = TASK_ID-1100
	os.system("nvidia-smi")
	command = "python3 "          				+ 'deepnet_main.py' + \
			  " -experiment_name "  			+ 'SBF_6a' + \
			  " -spec_name "        			+ 'swish' + \
			  " -run "              			+ str(RUN) + \
			  " -task="             			+ 'cifar100' + \
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
			  " -af_set "           			+ '1_jswish' +\
			  " -af_weights_init "  			+ 'default' + \
			  " -blend_trainable "  			+ 'True' + \
			  " -blend_mode "       			+ 'unrestricted' + \
			  " -swish_beta_trainable " 		+ 'True'
	subprocess.run(command, shell=True)

################################################################################
elif TASK_ID < 1301:
	RUN = TASK_ID-1200
	os.system("nvidia-smi")
	command = "python3 "          				+ 'deepnet_main.py' + \
			  " -experiment_name "  			+ 'SBF_6a' + \
			  " -spec_name "        			+ 'blend5_unrest' + \
			  " -run "              			+ str(RUN) + \
			  " -task="             			+ 'cifar100' + \
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
			  " -blend_trainable "  			+ 'True' + \
			  " -blend_mode "       			+ 'unrestricted' + \
			  " -swish_beta_trainable " 		+ 'True'
	subprocess.run(command, shell=True)
