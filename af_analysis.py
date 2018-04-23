import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# relu_i = pickle.load( open( path_finalweights+relu_fw_files[0], "rb" ) )
# for key, value in relu_i.items() :
#     print(key, value)

# ##############################################################################
# ### FUNCTIONS ################################################################
# ##############################################################################

def smcn_extract_weights(folder_path, file):

    # intantiate lists
    conv1_alpha1, conv2_alpha1, conv3_alpha1, conv4_alpha1, dense5_alpha1, dense6_alpha1 = [], [], [], [], [], []
    conv1_alpha2, conv2_alpha2, conv3_alpha2, conv4_alpha2, dense5_alpha2, dense6_alpha2 = [], [], [], [], [], []
    conv1_alpha3, conv2_alpha3, conv3_alpha3, conv4_alpha3, dense5_alpha3, dense6_alpha3 = [], [], [], [], [], []
    conv1_alpha4, conv2_alpha4, conv3_alpha4, conv4_alpha4, dense5_alpha4, dense6_alpha4 = [], [], [], [], [], []
    conv1_alpha5, conv2_alpha5, conv3_alpha5, conv4_alpha5, dense5_alpha5, dense6_alpha5 = [], [], [], [], [], []
    conv1_alpha6, conv2_alpha6, conv3_alpha6, conv4_alpha6, dense5_alpha6, dense6_alpha6 = [], [], [], [], [], []
    conv1_swish, conv2_swish, conv3_swish, conv4_swish, dense5_swish, dense6_swish = [], [], [], [], [], []

    # go through files and put all weights in lists
    for i in range(len(file)):
        run_i = pickle.load( open( folder_path+file[i], "rb" ) )
        for key, value in run_i.items():
            if 'conv1' in key:
                if 'swish_beta' in key: conv1_swish.append(value)
                elif 'blend_weights:0' in key: conv1_alpha1.append(value)
                elif 'blend_weights:1' in key: conv1_alpha2.append(value)
                elif 'blend_weights:2' in key: conv1_alpha3.append(value)
                elif 'blend_weights:3' in key: conv1_alpha4.append(value)
                elif 'blend_weights:4' in key: conv1_alpha5.append(value)
                elif 'blend_weights:5' in key: conv1_alpha6.append(value)
                else: print('[WARNING] Found an extra key:', key)
            elif 'conv2' in key:
                if 'swish_beta' in key: conv2_swish.append(value)
                elif 'blend_weights:0' in key: conv2_alpha1.append(value)
                elif 'blend_weights:1' in key: conv2_alpha2.append(value)
                elif 'blend_weights:2' in key: conv2_alpha3.append(value)
                elif 'blend_weights:3' in key: conv2_alpha4.append(value)
                elif 'blend_weights:4' in key: conv2_alpha5.append(value)
                elif 'blend_weights:5' in key: conv2_alpha6.append(value)
                else: print('[WARNING] Found an extra key:', key)
            elif 'conv3' in key:
                if 'swish_beta' in key: conv3_swish.append(value)
                elif 'blend_weights:0' in key: conv3_alpha1.append(value)
                elif 'blend_weights:1' in key: conv3_alpha2.append(value)
                elif 'blend_weights:2' in key: conv3_alpha3.append(value)
                elif 'blend_weights:3' in key: conv3_alpha4.append(value)
                elif 'blend_weights:4' in key: conv3_alpha5.append(value)
                elif 'blend_weights:5' in key: conv3_alpha6.append(value)
                else: print('[WARNING] Found an extra key:', key)
            elif 'conv4' in key:
                if 'swish_beta' in key: conv4_swish.append(value)
                elif 'blend_weights:0' in key: conv4_alpha1.append(value)
                elif 'blend_weights:1' in key: conv4_alpha2.append(value)
                elif 'blend_weights:2' in key: conv4_alpha3.append(value)
                elif 'blend_weights:3' in key: conv4_alpha4.append(value)
                elif 'blend_weights:4' in key: conv4_alpha5.append(value)
                elif 'blend_weights:5' in key: conv4_alpha6.append(value)
                else: print('[WARNING] Found an extra key:', key)
            elif 'dense5' in key:
                if 'swish_beta' in key: dense5_swish.append(value)
                elif 'blend_weights:0' in key: dense5_alpha1.append(value)
                elif 'blend_weights:1' in key: dense5_alpha2.append(value)
                elif 'blend_weights:2' in key: dense5_alpha3.append(value)
                elif 'blend_weights:3' in key: dense5_alpha4.append(value)
                elif 'blend_weights:4' in key: dense5_alpha5.append(value)
                elif 'blend_weights:5' in key: dense5_alpha6.append(value)
                else: print('[WARNING] Found an extra key:', key)
            elif 'dense6' in key:
                if 'swish_beta' in key: dense6_swish.append(value)
                elif 'blend_weights:0' in key: dense6_alpha1.append(value)
                elif 'blend_weights:1' in key: dense6_alpha2.append(value)
                elif 'blend_weights:2' in key: dense6_alpha3.append(value)
                elif 'blend_weights:3' in key: dense6_alpha4.append(value)
                elif 'blend_weights:4' in key: dense6_alpha5.append(value)
                elif 'blend_weights:5' in key: dense6_alpha6.append(value)
                else: print('[WARNING] Found an extra key:', key)
            else: print('[WARNING] Found an extra key:', key)

    # lists to np arrays
    conv1_1 = np.array(conv1_alpha1)
    conv1_2 = np.array(conv1_alpha2)
    conv1_3 = np.array(conv1_alpha3)
    conv1_4 = np.array(conv1_alpha4)
    conv1_5 = np.array(conv1_alpha5)
    conv1_swish = np.array(conv1_swish)
    conv2_1 = np.array(conv2_alpha1)
    conv2_2 = np.array(conv2_alpha2)
    conv2_3 = np.array(conv2_alpha3)
    conv2_4 = np.array(conv2_alpha4)
    conv2_5 = np.array(conv2_alpha5)
    conv2_swish = np.array(conv2_swish)
    conv3_1 = np.array(conv3_alpha1)
    conv3_2 = np.array(conv3_alpha2)
    conv3_3 = np.array(conv3_alpha3)
    conv3_4 = np.array(conv3_alpha4)
    conv3_5 = np.array(conv3_alpha5)
    conv3_swish = np.array(conv3_swish)
    conv4_1 = np.array(conv4_alpha1)
    conv4_2 = np.array(conv4_alpha2)
    conv4_3 = np.array(conv4_alpha3)
    conv4_4 = np.array(conv4_alpha4)
    conv4_5 = np.array(conv4_alpha5)
    conv4_swish = np.array(conv4_swish)
    dense5_1 = np.array(dense5_alpha1)
    dense5_2 = np.array(dense5_alpha2)
    dense5_3 = np.array(dense5_alpha3)
    dense5_4 = np.array(dense5_alpha4)
    dense5_5 = np.array(dense5_alpha5)
    dense5_swish = np.array(dense5_swish)
    dense6_1 = np.array(dense6_alpha1)
    dense6_2 = np.array(dense6_alpha2)
    dense6_3 = np.array(dense6_alpha3)
    dense6_4 = np.array(dense6_alpha4)
    dense6_5 = np.array(dense6_alpha5)
    dense6_swish = np.array(dense6_swish)

    # put into dicts
    wd_alpha1 = {'conv1': conv1_1, 'conv2': conv2_1, 'conv3': conv3_1, 'conv4': conv4_1, 'dense5': dense5_1, 'dense6': dense6_1 }
    wd_alpha2 = {'conv1': conv1_2, 'conv2': conv2_2, 'conv3': conv3_2, 'conv4': conv4_2, 'dense5': dense5_2, 'dense6': dense6_2 }
    wd_alpha3 = {'conv1': conv1_3, 'conv2': conv2_3, 'conv3': conv3_3, 'conv4': conv4_3, 'dense5': dense5_3, 'dense6': dense6_3 }
    wd_alpha4 = {'conv1': conv1_4, 'conv2': conv2_4, 'conv3': conv3_4, 'conv4': conv4_4, 'dense5': dense5_4, 'dense6': dense6_4 }
    wd_alpha5 = {'conv1': conv1_5, 'conv2': conv2_5, 'conv3': conv3_5, 'conv4': conv4_5, 'dense5': dense5_5, 'dense6': dense6_5 }
    wd_swish = {'conv1': conv1_swish, 'conv2': conv2_swish, 'conv3': conv3_swish, 'conv4': conv4_swish, 'dense5': dense5_swish, 'dense6': dense6_swish }

    # return
    return wd_alpha1, wd_alpha2, wd_alpha3, wd_alpha4, wd_alpha5, wd_swish

def smcn_print_mean_std(input_dict, info):
    print('')
    print(info + '           mean    | std')
    print(info + ' - conv1:  %.5f | %.5f' %(np.mean(input_dict['conv1']), np.std(input_dict['conv1'])))
    print(info + ' - conv2:  %.5f | %.5f' %(np.mean(input_dict['conv2']), np.std(input_dict['conv2'])))
    print(info + ' - conv3:  %.5f | %.5f' %(np.mean(input_dict['conv3']), np.std(input_dict['conv3'])))
    print(info + ' - conv4:  %.5f | %.5f' %(np.mean(input_dict['conv4']), np.std(input_dict['conv4'])))
    print(info + ' - dense5: %.5f | %.5f' %(np.mean(input_dict['dense5']), np.std(input_dict['dense5'])))
    print(info + ' - dense6: %.5f | %.5f' %(np.mean(input_dict['dense6']), np.std(input_dict['dense6'])))
    print('')

# ##############################################################################
# ### SCRIPT ###################################################################
# ##############################################################################

# define folder to find the weights
path_finalweights = './2_output_cifar/SBF_6b/_af_weights/'

# get files from folder
linu_fw_files = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('linu' in f))]
tanh_fw_files = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('tanh' in f))]
relu_fw_files = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('relu' in f))]
elu_fw_files = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('elu' in f))]
selu_fw_files = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('selu' in f))]
swish_fw_files = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('swish' in f))]
blend5_fw_files = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('blend5' in f))]

# extract weights from files
linu_wd, _, _, _, _, _ = smcn_extract_weights(path_finalweights, linu_fw_files)
tanh_wd, _, _, _, _, _ = smcn_extract_weights(path_finalweights, tanh_fw_files)
relu_wd, _, _, _, _, _ = smcn_extract_weights(path_finalweights, relu_fw_files)
elu_wd, _, _, _, _, _ = smcn_extract_weights(path_finalweights, elu_fw_files)
selu_wd, _, _, _, _, _ = smcn_extract_weights(path_finalweights, selu_fw_files)
swish_wd, _, _, _, _, swish_swishbeta = smcn_extract_weights(path_finalweights, swish_fw_files)
blend5_wd_a1, blend5_wd_a2, blend5_wd_a3, blend5_wd_a4, blend5_wd_a5, blend5_wd_swishbeta = smcn_extract_weights(path_finalweights, blend5_fw_files)

# print mean and std for all weights
smcn_print_mean_std(linu_wd, 'identity alpha')
smcn_print_mean_std(tanh_wd, 'tanh alpha')
smcn_print_mean_std(relu_wd, 'relu alpha')
smcn_print_mean_std(elu_wd, 'elu alpha')
smcn_print_mean_std(selu_wd, 'selu alpha')
smcn_print_mean_std(swish_wd, 'swish alpha')
smcn_print_mean_std(swish_swishbeta, 'swish beta')
smcn_print_mean_std(blend5_wd_a1, 'ABU-TERIS alpha AF1')

# smcn_print_mean_std(blend5_wd_a2, 'ABU-TERIS alpha AF2')
# smcn_print_mean_std(blend5_wd_a3, 'ABU-TERIS alpha AF3')
# smcn_print_mean_std(blend5_wd_a4, 'ABU-TERIS alpha AF4')
# smcn_print_mean_std(blend5_wd_a5, 'ABU-TERIS alpha AF5')
smcn_print_mean_std(blend5_wd_swishbeta, 'ABU-TERIS SwB')
