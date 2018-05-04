import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pickle
import scipy
from numpy import ma
from scipy.stats import norm

# relu_i = pickle.load( open( path_af_weights+relu_fw_files[0], "rb" ) )
# for key, value in relu_i.items() :
#     print(key, value)

# ##############################################################################
# ### FUNCTIONS ################################################################
# ##############################################################################

def smcn_extract_af_weights_over_time(folder_path, keyword, n_runs):

    # init lists
    conv1_alpha_per_run, conv2_alpha_per_run, conv3_alpha_per_run, conv4_alpha_per_run, dense5_alpha_per_run, dense6_alpha_per_run = [], [], [], [], [], []
    conv1_swish_per_run, conv2_swish_per_run, conv3_swish_per_run, conv4_swish_per_run, dense5_swish_per_run, dense6_swish_per_run = [], [], [], [], [], []

    for i in range(n_runs):
        timestep_files_run_i = [f for f in os.listdir(folder_path) if (os.path.isfile(os.path.join(folder_path, f)) and ('.pkl' in f) and (keyword in f) and ('run_'+str(i+1)+'_' in f))]
        n_timesteps = len(timestep_files_run_i)

        # init lists
        timesteps_idcs = []
        conv1_alpha, conv2_alpha, conv3_alpha, conv4_alpha, dense5_alpha, dense6_alpha = [], [], [], [], [], []
        conv1_swish, conv2_swish, conv3_swish, conv4_swish, dense5_swish, dense6_swish = [], [], [], [], [], []

        # go through files and put all weights in lists
        for t in range(n_timesteps):
            timestep_t = pickle.load( open( folder_path+timestep_files_run_i[t], "rb" ) )
            timesteps_idcs.append(timestep_files_run_i[t].split('.')[0].split('mb_')[-1])
            for key, value in timestep_t.items():
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

        # collect lists
        conv1_alpha_per_run.append(conv1_alpha)
        conv1_swish_per_run.append(conv1_swish)
        conv2_alpha_per_run.append(conv2_alpha)
        conv2_swish_per_run.append(conv2_swish)
        conv3_alpha_per_run.append(conv3_alpha)
        conv3_swish_per_run.append(conv3_swish)
        conv4_alpha_per_run.append(conv4_alpha)
        conv4_swish_per_run.append(conv4_swish)
        dense5_alpha_per_run.append(dense5_alpha)
        dense5_swish_per_run.append(dense5_swish)
        dense6_alpha_per_run.append(dense6_alpha)
        dense6_swish_per_run.append(dense6_swish)

    # lists to np arrays

    print('')
    print(type(conv1_alpha_per_run), len(conv1_alpha_per_run))
    for i in range(len(conv1_alpha_per_run)):
        print(type(conv1_alpha_per_run[i]), len(conv1_alpha_per_run[i]))
    print('')


    conv1_alpha_per_run = np.array(conv1_alpha_per_run)
    conv2_alpha_per_run = np.array(conv2_alpha_per_run)
    conv3_alpha_per_run = np.array(conv3_alpha_per_run)
    conv4_alpha_per_run = np.array(conv4_alpha_per_run)
    dense5_alpha_per_run = np.array(dense5_alpha_per_run)
    dense6_alpha_per_run = np.array(dense6_alpha_per_run)
    conv1_swish_per_run = np.array(conv1_swish_per_run)
    conv2_swish_per_run = np.array(conv2_swish_per_run)
    conv3_swish_per_run = np.array(conv3_swish_per_run)
    conv4_swish_per_run = np.array(conv4_swish_per_run)
    dense5_swish_per_run = np.array(dense5_swish_per_run)
    dense6_swish_per_run = np.array(dense6_swish_per_run)

    # put into dicts and return
    wd_alpha = {'conv1': conv1_alpha_per_run, 'conv2': conv2_alpha_per_run, 'conv3': conv3_alpha_per_run, 'conv4': conv4_alpha_per_run, 'dense5': dense5_alpha_per_run, 'dense6': dense6_alpha_per_run }
    wd_swish = {'conv1': conv1_swish_per_run, 'conv2': conv2_swish_per_run, 'conv3': conv3_swish_per_run, 'conv4': conv4_swish_per_run, 'dense5': dense5_swish_per_run, 'dense6': dense6_swish_per_run }
    return timesteps_idcs, wd_alpha, wd_swish

def plot_mean_alpha_by_layers_over_time(alpha_dict, ts_list, layer_list, title, saveplot_path, saveplot_filename):
    n_ts = len(ts_list)
    n_layers = len(layer_list)
    # extract means from alpha_dict
    alphas_by_layers = np.zeros((n_layers,n_ts))
    alphas_by_layers[0,:] = np.squeeze(np.mean(alpha_dict['conv1'], axis=0))
    alphas_by_layers[1,:] = np.squeeze(np.mean(alpha_dict['conv2'], axis=0))
    alphas_by_layers[2,:] = np.squeeze(np.mean(alpha_dict['conv3'], axis=0))
    alphas_by_layers[3,:] = np.squeeze(np.mean(alpha_dict['conv4'], axis=0))
    alphas_by_layers[4,:] = np.squeeze(np.mean(alpha_dict['dense5'], axis=0))
    alphas_by_layers[5,:] = np.squeeze(np.mean(alpha_dict['dense6'], axis=0))
    # figure
    linewidth_default = '3'
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111)
    ax.plot([0,10000],[0.2,0.2], ':', color='#000000', linewidth='1', alpha=0.5)
    ax.plot([0,10000],[0.4,0.4], ':', color='#000000', linewidth='1', alpha=0.5)
    ax.plot([0,10000],[0.6,0.6], ':', color='#000000', linewidth='1', alpha=0.5)
    ax.plot([0,10000],[0.8,0.8], ':', color='#000000', linewidth='1', alpha=0.5)
    ax.plot([0,10000],[1.0,1.0], ':', color='#000000', linewidth='1', alpha=0.5)
    ax.plot([0,10000],[1.2,1.2], ':', color='#000000', linewidth='1', alpha=0.5)
    ax.plot([0,10000],[1.4,1.4], ':', color='#000000', linewidth='1', alpha=0.5)
    color_list = ['#66c2a5', '#fc8d62', '#8c9fcb', '#e78ac3', '#a5d853', '#b4b4b4', '#ffd82e', '#e4c494']
    for layer in range(n_layers):
        ax.plot(np.array(ts_list), alphas_by_layers[layer,:], linewidth=linewidth_default, color=color_list[layer], label=layer_list[layer], alpha=1.0)
    ax.set_ylim([0.0,1.5])
    ax.set_ylabel('mean '+r'$\alpha_i$')
    ax.set_xlabel('iteration')
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='center left',  bbox_to_anchor=(1,0.5), fancybox=False, shadow=False)
    ax.tick_params(axis='x', which='both', bottom='off', top='off')
    # save plot as image
    if not os.path.exists(saveplot_path):
        os.makedirs(saveplot_path)
    fig.savefig(saveplot_path+saveplot_filename, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=150)

def plot_mean_ABU_alphas_over_time(alpha_dict, ts_list, layer_list, af_list, title, saveplot_path, saveplot_filename, ylim=[-.3,.6]):
    n_ts = len(ts_list)
    n_layers = len(layer_list)
    n_AFs = 5
    # extract means from alpha_dict
    alphas_by_layers = np.zeros((n_layers,n_ts,5))
    alphas_by_layers[0,:,:] = np.squeeze(np.mean(alpha_dict['conv1'], axis=0))
    alphas_by_layers[1,:,:] = np.squeeze(np.mean(alpha_dict['conv2'], axis=0))
    alphas_by_layers[2,:,:] = np.squeeze(np.mean(alpha_dict['conv3'], axis=0))
    alphas_by_layers[3,:,:] = np.squeeze(np.mean(alpha_dict['conv4'], axis=0))
    alphas_by_layers[4,:,:] = np.squeeze(np.mean(alpha_dict['dense5'], axis=0))
    alphas_by_layers[5,:,:] = np.squeeze(np.mean(alpha_dict['dense6'], axis=0))

    # figure
    linewidth_default = '3'
    fig = plt.figure(figsize=(20,5))
    for layer in range(n_layers):
        ax = fig.add_subplot(1,n_layers,layer+1)
        ax.plot([0,10000],[-.2,-.2], ':', color='#000000', linewidth='1', alpha=0.5)
        ax.plot([0,10000],[0.0,0.0], '-', color='#000000', linewidth='1', alpha=1.0)
        ax.plot([0,10000],[0.2,0.2], ':', color='#000000', linewidth='1', alpha=0.5)
        ax.plot([0,10000],[0.4,0.4], ':', color='#000000', linewidth='1', alpha=0.5)
        ax.plot([0,10000],[0.6,0.6], ':', color='#000000', linewidth='1', alpha=0.5)
        ax.plot([0,10000],[0.8,0.8], ':', color='#000000', linewidth='1', alpha=0.5)
        color_list = ['#66c2a5', '#fc8d62', '#8c9fcb', '#e78ac3', '#a5d853', '#b4b4b4', '#ffd82e', '#e4c494']
        for af in range(n_AFs):
            ax.plot(np.array(ts_list), alphas_by_layers[layer,:,af], linewidth=linewidth_default, color=color_list[af], label=af_list[af], alpha=1.0)
        ax.set_ylim(ylim)
        ax.set_ylabel('mean '+r'$\alpha_i$')
        ax.set_xlabel('iteration')
        ax.set_title(layer_list[layer])
        # ax.tick_params(axis='x', which='both', bottom='off', top='off')
        ax.set_xticklabels(['0','','','','','10k'])
        if layer+1 == n_layers:
            handles, labels = ax.get_legend_handles_labels()
            lgd = ax.legend(handles, labels, loc='center left',  bbox_to_anchor=(1,0.5), fancybox=False, shadow=False)
    # save plot as image
    if not os.path.exists(saveplot_path):
        os.makedirs(saveplot_path)
    fig.savefig(saveplot_path+saveplot_filename, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=150)

def smcn_extract_wstats_over_time():
    return 0,0

# ##############################################################################
# ### SCRIPT DEFAULT INIT ######################################################
# ##############################################################################

path_af_weights = './2_output_cifar/SBF_9a/_af_weights/'
path_all_weights = './2_output_cifar/SBF_9a/_all_weights/'
n_runs = 10

# extract weights from files
ts_linu, alphas_linu, _ = smcn_extract_af_weights_over_time(path_af_weights, '_linu', n_runs)
ts_relu, alphas_relu, _ = smcn_extract_af_weights_over_time(path_af_weights, '_relu', n_runs)
ts_elu, alphas_elu, _ = smcn_extract_af_weights_over_time(path_af_weights, '_elu', n_runs)
ts_b5u, alphas_b5u, betas_b5u = smcn_extract_af_weights_over_time(path_af_weights, '_blend5_unrest', n_runs)
ts_b5n, alphas_b5n, betas_b5n = smcn_extract_af_weights_over_time(path_af_weights, '_blend5_normalized', n_runs)
ts_b5p, alphas_b5p, betas_b5p = smcn_extract_af_weights_over_time(path_af_weights, '_blend5_posnormed', n_runs)

# plot weights over time
"""
layer_list = ['conv1', 'conv2', 'conv3', 'conv4', 'dense1', 'dense2']
plot_mean_alpha_by_layers_over_time(alphas_linu, ts_linu, layer_list, r'$\alpha_i$ over time ($\alpha$I)', './2_result_plots/', 'over_time_linu_alphas_default_init')
plot_mean_alpha_by_layers_over_time(alphas_relu, ts_relu, layer_list, r'$\alpha_i$ over time ($\alpha$ReLU)', './2_result_plots/', 'over_time_relu_alphas_default_init')
plot_mean_alpha_by_layers_over_time(alphas_elu, ts_elu, layer_list, r'$\alpha_i$ over time ($\alpha$ELU)', './2_result_plots/', 'over_time_elu_alphas_default_init')
af_list = ['ReLU', 'ELU', 'tanh', 'Swish', 'I']
plot_mean_ABU_alphas_over_time(alphas_b5u, ts_b5u, layer_list, af_list, 'needs title', './2_result_plots/', 'over_time_b5u_alphas_default_init', [-.4,.7])
plot_mean_ABU_alphas_over_time(alphas_b5n, ts_b5n, layer_list, af_list, 'needs title', './2_result_plots/', 'over_time_b5n_alphas_default_init', [-.4,.8])
plot_mean_ABU_alphas_over_time(alphas_b5p, ts_b5p, layer_list, af_list, 'needs title', './2_result_plots/', 'over_time_b5p_alphas_default_init', [-.1,.8])
"""

# get regular weights W mean and std
relu_af_weights_final_stats = []
for key, value in alphas_relu.items():
    print(type(alphas_relu[key]))
    print(alphas_relu[key].shape)
    print(alphas_relu[key])
    relu_af_weights_final_stats.append([key, np.mean(np.array(value)), np.std(value)])
tsw_relu, w_relu_stats = smcn_extract_wstats_over_time(path_experiment, 'relu', 'weights', layer_list)

# ##############################################################################
# ### SCRIPT PRE-TRAINED INIT ##################################################
# ##############################################################################
"""
path_af_weights = './2_output_cifar/SBF_9b/_af_weights/'
n_runs = 10

# extract weights from files
ts_linu, alphas_linu, _ = smcn_extract_af_weights_over_time(path_af_weights, '_linu_pretrained', n_runs)
ts_relu, alphas_relu, _ = smcn_extract_af_weights_over_time(path_af_weights, '_relu_pretrained', n_runs)
ts_elu, alphas_elu, _ = smcn_extract_af_weights_over_time(path_af_weights, '_elu_pretrained', n_runs)
ts_b5u, alphas_b5u, betas_b5u = smcn_extract_af_weights_over_time(path_af_weights, '_blend5_unrest_pretrained', n_runs)
ts_b5n, alphas_b5n, betas_b5n = smcn_extract_af_weights_over_time(path_af_weights, '_blend5_normalized_pretrained', n_runs)
ts_b5p, alphas_b5p, betas_b5p = smcn_extract_af_weights_over_time(path_af_weights, '_blend5_posnormed_pretrained', n_runs)

# plot weights over time
layer_list = ['conv1', 'conv2', 'conv3', 'conv4', 'dense1', 'dense2']
plot_mean_alpha_by_layers_over_time(alphas_linu, ts_linu, layer_list, r'$\alpha_i$ over time ($\alpha$I)', './2_result_plots/', 'over_time_linu_alphas_pretrained_init')
plot_mean_alpha_by_layers_over_time(alphas_relu, ts_relu, layer_list, r'$\alpha_i$ over time ($\alpha$ReLU)', './2_result_plots/', 'over_time_relu_alphas_pretrained_init')
plot_mean_alpha_by_layers_over_time(alphas_elu, ts_elu, layer_list, r'$\alpha_i$ over time ($\alpha$ELU)', './2_result_plots/', 'over_time_elu_alphas_pretrained_init')
af_list = ['ReLU', 'ELU', 'tanh', 'Swish', 'I']
plot_mean_ABU_alphas_over_time(alphas_b5u, ts_b5u, layer_list, af_list, 'needs title', './2_result_plots/', 'over_time_b5u_alphas_pretrained_init', [-.4,.7])
plot_mean_ABU_alphas_over_time(alphas_b5n, ts_b5n, layer_list, af_list, 'needs title', './2_result_plots/', 'over_time_b5n_alphas_pretrained_init', [-.4,.8])
plot_mean_ABU_alphas_over_time(alphas_b5p, ts_b5p, layer_list, af_list, 'needs title', './2_result_plots/', 'over_time_b5p_alphas_pretrained_init', [-.1,.8])
"""
