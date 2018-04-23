import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
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
    conv1_alpha, conv2_alpha, conv3_alpha, conv4_alpha, dense5_alpha, dense6_alpha = [], [], [], [], [], []
    conv1_swish, conv2_swish, conv3_swish, conv4_swish, dense5_swish, dense6_swish = [], [], [], [], [], []

    # go through files and put all weights in lists
    for i in range(len(file)):
        run_i = pickle.load( open( folder_path+file[i], "rb" ) )
        for key, value in run_i.items():
            if 'conv1' in key:
                if 'swish_beta' in key: conv1_swish.append(value)
                elif 'blend_weights:0' in key: conv1_alpha.append(value)
                else: print('[WARNING] Found an extra key:', key)
            elif 'conv2' in key:
                if 'swish_beta' in key: conv2_swish.append(value)
                elif 'blend_weights:0' in key: conv2_alpha.append(value)
                else: print('[WARNING] Found an extra key:', key)
            elif 'conv3' in key:
                if 'swish_beta' in key: conv3_swish.append(value)
                elif 'blend_weights:0' in key: conv3_alpha.append(value)
                else: print('[WARNING] Found an extra key:', key)
            elif 'conv4' in key:
                if 'swish_beta' in key: conv4_swish.append(value)
                elif 'blend_weights:0' in key: conv4_alpha.append(value)
                else: print('[WARNING] Found an extra key:', key)
            elif 'dense5' in key:
                if 'swish_beta' in key: dense5_swish.append(value)
                elif 'blend_weights:0' in key: dense5_alpha.append(value)
                else: print('[WARNING] Found an extra key:', key)
            elif 'dense6' in key:
                if 'swish_beta' in key: dense6_swish.append(value)
                elif 'blend_weights:0' in key: dense6_alpha.append(value)
                else: print('[WARNING] Found an extra key:', key)
            else: print('[WARNING] Found an extra key:', key)

    # lists to np arrays
    conv1_alpha = np.array(conv1_alpha)
    conv1_swish = np.array(conv1_swish)
    conv2_alpha = np.array(conv2_alpha)
    conv2_swish = np.array(conv2_swish)
    conv3_alpha = np.array(conv3_alpha)
    conv3_swish = np.array(conv3_swish)
    conv4_alpha = np.array(conv4_alpha)
    conv4_swish = np.array(conv4_swish)
    dense5_alpha = np.array(dense5_alpha)
    dense5_swish = np.array(dense5_swish)
    dense6_alpha = np.array(dense6_alpha)
    dense6_swish = np.array(dense6_swish)

    # put into dicts
    wd_alpha = {'conv1': conv1_alpha, 'conv2': conv2_alpha, 'conv3': conv3_alpha, 'conv4': conv4_alpha, 'dense5': dense5_alpha, 'dense6': dense6_alpha }
    wd_swish = {'conv1': conv1_swish, 'conv2': conv2_swish, 'conv3': conv3_swish, 'conv4': conv4_swish, 'dense5': dense5_swish, 'dense6': dense6_swish }

    # return
    return wd_alpha, wd_swish

def smcn_print_mean_std(input_dict, info):
    print('')
    if input_dict['conv1'].shape[1] > 1:
        print(info + '           [mean] [std]')
        print(info + ' - conv1: ', np.mean(input_dict['conv1'], axis=0), np.std(input_dict['conv1'], axis=0))
        print(info + ' - conv2: ', np.mean(input_dict['conv2'], axis=0), np.std(input_dict['conv2'], axis=0))
        print(info + ' - conv3: ', np.mean(input_dict['conv3'], axis=0), np.std(input_dict['conv3'], axis=0))
        print(info + ' - conv4: ', np.mean(input_dict['conv4'], axis=0), np.std(input_dict['conv4'], axis=0))
        print(info + ' - dense5:', np.mean(input_dict['dense5'], axis=0), np.std(input_dict['dense5'], axis=0))
        print(info + ' - dense6:', np.mean(input_dict['dense6'], axis=0), np.std(input_dict['dense6'], axis=0))
    else:
        print(info + '           mean    | std')
        print(info + ' - conv1:  %.5f | %.5f' %(np.mean(input_dict['conv1'], axis=0), np.std(input_dict['conv1'], axis=0)))
        print(info + ' - conv2:  %.5f | %.5f' %(np.mean(input_dict['conv2'], axis=0), np.std(input_dict['conv2'], axis=0)))
        print(info + ' - conv3:  %.5f | %.5f' %(np.mean(input_dict['conv3'], axis=0), np.std(input_dict['conv3'], axis=0)))
        print(info + ' - conv4:  %.5f | %.5f' %(np.mean(input_dict['conv4'], axis=0), np.std(input_dict['conv4'], axis=0)))
        print(info + ' - dense5: %.5f | %.5f' %(np.mean(input_dict['dense5'], axis=0), np.std(input_dict['dense5'], axis=0)))
        print(info + ' - dense6: %.5f | %.5f' %(np.mean(input_dict['dense6'], axis=0), np.std(input_dict['dense6'], axis=0)))
    print('')

def plot_mean_alpha_over_layers(af_list, name_list, saveplot_path, saveplot_filename, includes_beta=False):
    # extract means from afs
    afs_by_layers_means = np.zeros((len(af_list),6))
    for af in range(len(af_list)):
        af_dict = af_list[af]
        af_name = name_list[af]
        afs_by_layers_means[af,0] = np.mean(af_dict['conv1'])
        afs_by_layers_means[af,1] = np.mean(af_dict['conv2'])
        afs_by_layers_means[af,2] = np.mean(af_dict['conv3'])
        afs_by_layers_means[af,3] = np.mean(af_dict['conv4'])
        afs_by_layers_means[af,4] = np.mean(af_dict['dense5'])
        afs_by_layers_means[af,5] = np.mean(af_dict['dense6'])
    # get number of AFs in list
    n_afs = len(af_list)
    if includes_beta:
        n_afs -= 1
    # figure
    linewidth_default = '3'
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.plot([-1,7],[1.0,1.0], '-', color='#808080', linewidth=linewidth_default, alpha=1.0)
    color_list = ['#66c2a5', '#fc8d62', '#8c9fcb', '#e78ac3', '#a5d853', '#ffd82e', '#e4c494', '#b4b4b4']
    for af in range(n_afs):
        ax.plot(np.arange(6)+1, afs_by_layers_means[af,:], linewidth=linewidth_default, color=color_list[af], label=name_list[af], alpha=1.0)
    if includes_beta:
        ax.plot(np.arange(6)+1, afs_by_layers_means[-1,:], '--', linewidth=linewidth_default, color=color_list[af], label=name_list[n_afs], alpha=0.5)
    plt.ylim([0.0,1.4])
    plt.xlim([0.8,6.2])
    ax.set_xticklabels(['','conv1','conv2','conv3','conv4','dense1','dense2'])
    plt.ylabel('mean alpha')
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='center left',  bbox_to_anchor=(1,0.5), fancybox=True, shadow=True)
    if not os.path.exists(saveplot_path):
        os.makedirs(saveplot_path)
    fig.savefig(saveplot_path+saveplot_filename, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=150)

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
linu_wd, _ = smcn_extract_weights(path_finalweights, linu_fw_files)
tanh_wd, _ = smcn_extract_weights(path_finalweights, tanh_fw_files)
relu_wd, _ = smcn_extract_weights(path_finalweights, relu_fw_files)
elu_wd, _ = smcn_extract_weights(path_finalweights, elu_fw_files)
selu_wd, _ = smcn_extract_weights(path_finalweights, selu_fw_files)
swish_wd, swish_swishbeta = smcn_extract_weights(path_finalweights, swish_fw_files)
blend5_wd, blend5_swishbeta = smcn_extract_weights(path_finalweights, blend5_fw_files)

# print mean and std for all weights
smcn_print_mean_std(linu_wd, 'identity alpha')
smcn_print_mean_std(tanh_wd, 'tanh alpha')
smcn_print_mean_std(relu_wd, 'relu alpha')
smcn_print_mean_std(elu_wd, 'elu alpha')
smcn_print_mean_std(selu_wd, 'selu alpha')
smcn_print_mean_std(swish_wd, 'swish alpha')
smcn_print_mean_std(swish_swishbeta, 'swish beta')
smcn_print_mean_std(blend5_wd, 'ABU-TERIS alpha')
smcn_print_mean_std(blend5_swishbeta, 'ABU-TERIS SwB')

# plot mean alpha over layers for adaptively scaled functions
af_list = [linu_wd, tanh_wd, relu_wd, elu_wd, selu_wd, swish_wd, swish_swishbeta]
name_list = ['identity', 'tanh', 'ReLU', 'ELU', 'SELU', 'Swish', 'Swish beta']
plot_mean_alpha_over_layers(af_list, name_list, './2_result_plots/', 'mean_alpha_over_layers_scaled_afs_cifar100.png', includes_beta=True)
