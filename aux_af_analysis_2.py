import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pickle
import scipy
from numpy import ma
from scipy.stats import norm
plt.rcParams["font.family"] = ["FreeSerif"]

# ##############################################################################
# ### FUNCTIONS ################################################################
# ##############################################################################

def sort_files_by_mb(filelist):
    mb_list = []
    for filename in filelist:
        mb_list.append(int(filename.split('mb_')[-1].split('.')[0]))
    order = np.argsort(mb_list).tolist()
    return [filelist[i] for i in order]

def smcn_extract_af_weights_over_time(folder_path, keyword, mbs, n_timesteps_saved):

    # init lists
    conv1_alpha_per_run, conv2_alpha_per_run, conv3_alpha_per_run, conv4_alpha_per_run, dense5_alpha_per_run, dense6_alpha_per_run = [], [], [], [], [], []
    conv1_swish_per_run, conv2_swish_per_run, conv3_swish_per_run, conv4_swish_per_run, dense5_swish_per_run, dense6_swish_per_run = [], [], [], [], [], []

    spec_file_list = [f for f in os.listdir(folder_path) if (os.path.isfile(os.path.join(folder_path, f)) and ('.pkl' in f) and (keyword in f))]
    finished_runs = []
    for spec_file in spec_file_list:
        run = int(spec_file.split('run_')[-1].split('_mb')[0])
        mb = int(spec_file.split('mb_')[-1].split('.')[0])
        if mb == mbs:
            finished_runs.append(run)
    n_runs = len(finished_runs)

    timesteps_idcs = []
    for run in finished_runs:
        timestep_files_run_i = [f for f in os.listdir(folder_path) if (os.path.isfile(os.path.join(folder_path, f)) and ('.pkl' in f) and (keyword in f) and ('run_'+str(run)+'_' in f))]
        timestep_files_run_i = sort_files_by_mb(timestep_files_run_i)
        n_timesteps = len(timestep_files_run_i)

        # init lists
        timesteps_idcs = []
        conv1_alpha, conv2_alpha, conv3_alpha, conv4_alpha, dense5_alpha, dense6_alpha = [], [], [], [], [], []
        conv1_swish, conv2_swish, conv3_swish, conv4_swish, dense5_swish, dense6_swish = [], [], [], [], [], []

        # go through files and put all weights in lists
        last_ts = 0
        for t in range(n_timesteps):
            timestep_t = pickle.load( open( folder_path+timestep_files_run_i[t], "rb" ) )
            ts = int(timestep_files_run_i[t].split('.')[0].split('mb_')[-1])
            if ts == 1 or np.abs(ts - last_ts) > 1000:
                timesteps_idcs.append(ts)
                last_ts = ts
                for key, value in timestep_t.items():
                    if 'conv1' in key:
                        if 'swish_beta' in key: conv1_swish.append(value)
                        elif 'blend_weights' in key: conv1_alpha.append(value)
                        else: print('[WARNING] Found an extra key:', key)
                    elif 'conv2' in key:
                        if 'swish_beta' in key: conv2_swish.append(value)
                        elif 'blend_weights' in key: conv2_alpha.append(value)
                        else: print('[WARNING] Found an extra key:', key)
                    elif 'conv3' in key:
                        if 'swish_beta' in key: conv3_swish.append(value)
                        elif 'blend_weights' in key: conv3_alpha.append(value)
                        else: print('[WARNING] Found an extra key:', key)
                    elif 'conv4' in key:
                        if 'swish_beta' in key: conv4_swish.append(value)
                        elif 'blend_weights' in key: conv4_alpha.append(value)
                        else: print('[WARNING] Found an extra key:', key)
                    elif 'dense5' in key:
                        if 'swish_beta' in key: dense5_swish.append(value)
                        elif 'blend_weights' in key: dense5_alpha.append(value)
                        else: print('[WARNING] Found an extra key:', key)
                    elif 'dense6' in key:
                        if 'swish_beta' in key: dense6_swish.append(value)
                        elif 'blend_weights' in key: dense6_alpha.append(value)
                        else: print('[WARNING] Found an extra key:', key)
                    else: print('[WARNING] Found an extra key:', key)

        if len(conv1_alpha) == n_timesteps_saved:
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

def plot_mean_alpha_by_layers_over_time(alpha_dict, ts_list, layer_list, title, saveplot_path, saveplot_filename, xlim=[0,60000], ylim=[0.0,1.8]):
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
    fig = plt.figure(figsize=(4,6))
    ax = fig.add_subplot(111)
    ax.plot([0,xlim[1]],[0.2,0.2], ':', color='#000000', linewidth='1', alpha=0.5)
    ax.plot([0,xlim[1]],[0.4,0.4], ':', color='#000000', linewidth='1', alpha=0.5)
    ax.plot([0,xlim[1]],[0.6,0.6], ':', color='#000000', linewidth='1', alpha=0.5)
    ax.plot([0,xlim[1]],[0.8,0.8], ':', color='#000000', linewidth='1', alpha=0.5)
    ax.plot([0,xlim[1]],[1.0,1.0], ':', color='#000000', linewidth='1', alpha=0.5)
    ax.plot([0,xlim[1]],[1.2,1.2], ':', color='#000000', linewidth='1', alpha=0.5)
    ax.plot([0,xlim[1]],[1.4,1.4], ':', color='#000000', linewidth='1', alpha=0.5)
    ax.plot([0,xlim[1]],[1.6,1.6], ':', color='#000000', linewidth='1', alpha=0.5)
    ax.plot([0,xlim[1]],[1.8,1.8], ':', color='#000000', linewidth='1', alpha=0.5)
    ax.plot([0,xlim[1]],[2.0,2.0], ':', color='#000000', linewidth='1', alpha=0.5)
    color_list = ['#66c2a5', '#fc8d62', '#8c9fcb', '#e78ac3', '#a5d853', '#b4b4b4', '#ffd82e', '#e4c494']
    for layer in range(n_layers):
        ax.plot(np.array(ts_list), alphas_by_layers[layer,:], linewidth=linewidth_default, color=color_list[layer], label=layer_list[layer], alpha=1.0)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_ylabel('mean '+r'$\alpha_i$')
    ax.set_xlabel('iteration')
    ax.set_title(title)
    ax.tick_params(axis='x', which='both', bottom='on', top='off')
    ax.set_xticklabels(['0','','20k','','40k','','60k'])
    ax.tick_params(axis='y', which='both', left='on', right='off')
    # ax.set_yticklabels(['0','','','','','10k'])
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='center left',  bbox_to_anchor=(1,0.5), fancybox=False, shadow=False)
    ax.tick_params(axis='x', which='both', bottom='off', top='off')
    # save plot as image
    if not os.path.exists(saveplot_path):
        os.makedirs(saveplot_path)
    fig.savefig(saveplot_path+saveplot_filename, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=150)

def plot_mean_alpha_by_layers_over_time_grouped(alpha_list, af_name_list, ts_listoflists, layer_list, saveplot_path, saveplot_filename, xlim=[0,60000], ylim_list=[[0.0,1.8]]):

    fig = plt.figure(figsize=(14.11,3))
    n_layers = len(layer_list)
    n_AFs = len(alpha_list)
    linewidth_default = '2'
    if len(ylim_list) == 1:
        tmp = ylim_list[0]
        for i in range(n_AFs-1):
            ylim_list.append(tmp)

    for i, alpha_dict in enumerate(alpha_list):
        ts_list = ts_listoflists[i]
        n_ts = len(ts_list)
        ax = fig.add_subplot(1,6,i+1)

        # extract means from alpha_dict
        alphas_by_layers = np.zeros((n_layers,n_ts))
        alphas_by_layers[0,:] = np.squeeze(np.mean(alpha_dict['conv1'], axis=0))
        alphas_by_layers[1,:] = np.squeeze(np.mean(alpha_dict['conv2'], axis=0))
        alphas_by_layers[2,:] = np.squeeze(np.mean(alpha_dict['conv3'], axis=0))
        alphas_by_layers[3,:] = np.squeeze(np.mean(alpha_dict['conv4'], axis=0))
        alphas_by_layers[4,:] = np.squeeze(np.mean(alpha_dict['dense5'], axis=0))
        alphas_by_layers[5,:] = np.squeeze(np.mean(alpha_dict['dense6'], axis=0))
        # figure
        ax.plot([0,xlim[1]],[0.2,0.2], ':', color='#000000', linewidth='1', alpha=0.5)
        ax.plot([0,xlim[1]],[0.4,0.4], ':', color='#000000', linewidth='1', alpha=0.5)
        ax.plot([0,xlim[1]],[0.6,0.6], ':', color='#000000', linewidth='1', alpha=0.5)
        ax.plot([0,xlim[1]],[0.8,0.8], ':', color='#000000', linewidth='1', alpha=0.5)
        ax.plot([0,xlim[1]],[1.0,1.0], '-', color='#000000', linewidth='1', alpha=0.5)
        ax.plot([0,xlim[1]],[1.2,1.2], ':', color='#000000', linewidth='1', alpha=0.5)
        ax.plot([0,xlim[1]],[1.4,1.4], ':', color='#000000', linewidth='1', alpha=0.5)
        ax.plot([0,xlim[1]],[1.6,1.6], ':', color='#000000', linewidth='1', alpha=0.5)
        ax.plot([0,xlim[1]],[1.8,1.8], ':', color='#000000', linewidth='1', alpha=0.5)
        ax.plot([0,xlim[1]],[2.0,2.0], ':', color='#000000', linewidth='1', alpha=0.5)
        ax.plot([0,xlim[1]],[2.2,2.2], ':', color='#000000', linewidth='1', alpha=0.5)
        ax.plot([0,xlim[1]],[2.4,2.4], ':', color='#000000', linewidth='1', alpha=0.5)
        color_list = ['#66c2a5', '#fc8d62', '#8c9fcb', '#e78ac3', '#a5d853', '#b4b4b4', '#ffd82e', '#e4c494']
        for layer in range(n_layers):
            ax.plot(np.array(ts_list), alphas_by_layers[layer,:], linewidth=linewidth_default, color=color_list[layer], label=layer_list[layer], alpha=1.0)
        ax.set_ylim(ylim_list[i])
        ax.set_xlim(xlim)
        if i == 0:
            ax.set_ylabel('mean '+r'$\alpha_i$')
        ax.set_yticks(np.linspace(ylim_list[i][0], ylim_list[i][1], num=7))
        ax.set_title(af_name_list[i])
        ax.tick_params(axis='x', which='both', bottom='on', top='off')
        ax.set_xticklabels(['0','','20k','','40k','','60k'])
        ax.tick_params(axis='y', which='both', left='on', right='off')
        if i+1 == n_AFs:
            handles, labels = ax.get_legend_handles_labels()
            lgd = ax.legend(handles, labels, loc='center left',  bbox_to_anchor=(1,0.5), fancybox=False, shadow=False)
        ax.tick_params(axis='x', which='both', bottom='off', top='off')

    # save plot as image
    plt.tight_layout()
    if not os.path.exists(saveplot_path):
        os.makedirs(saveplot_path)
    fig.savefig(saveplot_path+saveplot_filename, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=150)

def plot_mean_ABU_alphas_over_time(alpha_dict, ts_list, layer_list, af_list, title, saveplot_path, saveplot_filename, xlim=[0,60000], ylim=[-.3,.6], norm="None"):
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

    # norm
    if norm == 'N':
        for layer in range(alphas_by_layers.shape[0]):
            for ts in range(alphas_by_layers.shape[1]):
                alphas_by_layers[layer,ts,:] = alphas_by_layers[layer,ts,:]/np.sum(alphas_by_layers[layer,ts,:])
    if norm == 'S':
        for layer in range(alphas_by_layers.shape[0]):
            for ts in range(alphas_by_layers.shape[1]):
                alphas_by_layers[layer,ts,:] = np.exp(alphas_by_layers[layer,ts,:])
                alphas_by_layers[layer,ts,:] = alphas_by_layers[layer,ts,:]/np.sum(alphas_by_layers[layer,ts,:])
    if norm == 'A':
        for layer in range(alphas_by_layers.shape[0]):
            for ts in range(alphas_by_layers.shape[1]):
                alphas_by_layers[layer,ts,:] = alphas_by_layers[layer,ts,:]/np.sum(np.abs(alphas_by_layers[layer,ts,:]))
    if norm == 'P':
        for layer in range(alphas_by_layers.shape[0]):
            for ts in range(alphas_by_layers.shape[1]):
                alphas_by_layers[layer,ts,:] = np.clip(alphas_by_layers[layer,ts,:], 0.0001, 1000.0)
                alphas_by_layers[layer,ts,:] = alphas_by_layers[layer,ts,:]/np.sum(alphas_by_layers[layer,ts,:])

    # figure
    linewidth_default = '2'
    fig = plt.figure(figsize=(14,3))
    for layer in range(n_layers):
        ax = fig.add_subplot(1,n_layers,layer+1)
        y_line = -10.0
        for i in range(100):
            if y_line > ylim[0] and y_line < ylim[1] and not y_line==0.0 and not y_line==0.2:
                ax.plot([0,xlim[1]],[y_line,y_line], ':', color='#000000', linewidth='1', alpha=0.5)
            y_line += 0.2
        ax.plot([0,xlim[1]],[0,0], '-', color='#000000', linewidth='1', alpha=1.0)
        ax.plot([0,xlim[1]],[.2,.2], '-', color='#000000', linewidth='1', alpha=1.0)
        color_list = ['#66c2a5', '#fc8d62', '#8c9fcb', '#e78ac3', '#a5d853', '#b4b4b4', '#ffd82e', '#e4c494']
        for af in range(n_AFs):
            ax.plot(np.array(ts_list), alphas_by_layers[layer,:,af], linewidth=linewidth_default, color=color_list[af], label=af_list[af], alpha=1.0)
        ax.set_ylim(ylim)
        if layer == 0:
            ax.set_ylabel(title)
        # ax.set_xlabel('iteration')
        ax.set_title(layer_list[layer])
        ax.tick_params(axis='x', which='both', bottom='on', top='off')
        ax.set_xticklabels(['0','','20k','','40k','','60k'])
        ax.tick_params(axis='y', which='both', left='on', right='off')
        if layer > 0:
            ax.set_yticklabels([])
        if layer+1 == n_layers:
            handles, labels = ax.get_legend_handles_labels()
            lgd = ax.legend(handles, labels, loc='center left',  bbox_to_anchor=(1,0.5), fancybox=False, shadow=False)
    # save plot as image
    if not os.path.exists(saveplot_path):
        os.makedirs(saveplot_path)
    fig.savefig(saveplot_path+saveplot_filename, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=150)

# ##############################################################################
# ### SCRIPT ###################################################################
# ##############################################################################

sections = [1]

# ##############################################################################
# ### SCRIPT: ASU ##############################################################
# ##############################################################################

if 1 in sections:

    path_af_weights = './3_output_cifar/ASC_main/0_af_weights/'
    path_all_weights = './3_output_cifar/ASC_main/0_all_weights/'
    layer_list = ['c1', 'c2', 'c3', 'c4', 'd1', 'd2']
    abu_af_list = ['ReLU', 'ELU', 'tanh', 'Swish', 'I']

    # extract weights from files
    ts_linu, alphas_linu, _ = smcn_extract_af_weights_over_time(path_af_weights, 'alpha_I_', 60000, 60)
    print('AF weights extracted for I')
    ts_tanh, alphas_tanh, _ = smcn_extract_af_weights_over_time(path_af_weights, 'alpha_tanh_', 60000, 60)
    print('AF weights extracted for tanh')
    ts_relu, alphas_relu, _ = smcn_extract_af_weights_over_time(path_af_weights, 'alpha_relu_', 60000, 60)
    print('AF weights extracted for relu')
    ts_elu, alphas_elu, _ = smcn_extract_af_weights_over_time(path_af_weights, 'alpha_elu_', 60000, 60)
    print('AF weights extracted for elu')
    ts_selu, alphas_selu, _ = smcn_extract_af_weights_over_time(path_af_weights, 'alpha_selu_', 60000, 60)
    print('AF weights extracted for selu')
    ts_swish, alphas_swish, _ = smcn_extract_af_weights_over_time(path_af_weights, 'alpha_swish_', 60000, 60)
    print('AF weights extracted for swish')

    # plot weights over time
    alphas_list = [alphas_linu, alphas_tanh, alphas_relu, alphas_elu, alphas_selu, alphas_swish]
    af_name_list = [r'$\alpha I$', r'$\alpha tanh$', r'$\alpha ReLU$', r'$\alpha ELU$', r'$\alpha SELU$', r'$\alpha Swish$']
    ts_list = [ts_linu, ts_tanh, ts_relu, ts_elu, ts_selu, ts_swish]
    ylim_list = [[0.,1.2], [0.,2.4], [0.,1.2], [0.,1.2], [0.,1.2], [0.,1.2]]
    plot_mean_alpha_by_layers_over_time_grouped(alphas_list, af_name_list, ts_list, layer_list, './3_result_plots/', 'MAIN_ASU_over_time', ylim_list=ylim_list)

    # plot_mean_alpha_by_layers_over_time(alphas_linu, ts_linu, layer_list, r'$\alpha_i$ over time ($\alpha$I)', './3_result_plots/', 'over_time_linu_alphas_default_init', ylim=[0.,1.2])
    # plot_mean_alpha_by_layers_over_time(alphas_relu, ts_relu, layer_list, r'$\alpha_i$ over time ($\alpha$ReLU)', './3_result_plots/', 'over_time_relu_alphas_default_init', ylim=[0.,1.2])
    # plot_mean_alpha_by_layers_over_time(alphas_tanh, ts_tanh, layer_list, r'$\alpha_i$ over time ($\alpha$tanh)', './3_result_plots/', 'over_time_tanh_alphas_default_init', ylim=[0.,2.4])
    # plot_mean_alpha_by_layers_over_time(alphas_elu, ts_elu, layer_list, r'$\alpha_i$ over time ($\alpha$ELU)', './3_result_plots/', 'over_time_elu_alphas_default_init', ylim=[0.,1.2])
    # plot_mean_alpha_by_layers_over_time(alphas_selu, ts_selu, layer_list, r'$\alpha_i$ over time ($\alpha$SELU)', './3_result_plots/', 'over_time_selu_alphas_default_init', ylim=[0.,1.2])
    # plot_mean_alpha_by_layers_over_time(alphas_swish, ts_swish, layer_list, r'$\alpha_i$ over time ($\alpha$Swish)', './3_result_plots/', 'over_time_swish_alphas_default_init', ylim=[0.,1.2])

# ##############################################################################
# ### SCRIPT: ABU ##############################################################
# ##############################################################################

if 2 in sections:

    path_af_weights = './3_output_cifar/ASC_main/0_af_weights/'
    path_all_weights = './3_output_cifar/ASC_main/0_all_weights/'
    layer_list = ['conv1', 'conv2', 'conv3', 'conv4', 'dense1', 'dense2']
    abu_af_list = ['ReLU', 'ELU', 'tanh', 'Swish', 'I']

    ts_b5u, alphas_b5u, betas_b5u = smcn_extract_af_weights_over_time(path_af_weights, 'ABU_T', 60000, 60)
    print('AF weights extracted for ABU')
    ts_b5n, alphas_b5n, betas_b5n = smcn_extract_af_weights_over_time(path_af_weights, 'ABU_N_', 60000, 60)
    print('AF weights extracted for ABU_N')
    ts_b5p, alphas_b5p, betas_b5p = smcn_extract_af_weights_over_time(path_af_weights, 'ABU_P_', 60000, 60)
    print('AF weights extracted for ABU_P')
    ts_b5a, alphas_b5a, betas_b5a = smcn_extract_af_weights_over_time(path_af_weights, 'ABU_A_', 60000, 60)
    print('AF weights extracted for ABU_A')
    ts_b5s, alphas_b5s, betas_b5s = smcn_extract_af_weights_over_time(path_af_weights, 'ABU_S_', 60000, 60)
    print('AF weights extracted for ABU_S')

    plot_mean_ABU_alphas_over_time(alphas_b5u, ts_b5u, layer_list, abu_af_list, r'$ABU$'+': mean '+r'$\alpha_{ij}$', './3_result_plots/', 'over_time_ABU_alphas_default_init', ylim=[-.4,.6], norm="None")
    plot_mean_ABU_alphas_over_time(alphas_b5n, ts_b5n, layer_list, abu_af_list, r'$ABU_N$'+': mean '+r'$\alpha_{ij}$', './3_result_plots/', 'over_time_ABU_N_alphas_default_init', ylim=[-1.6,2.4], norm="N")
    plot_mean_ABU_alphas_over_time(alphas_b5p, ts_b5p, layer_list, abu_af_list, r'$ABU_P$'+': mean '+r'$\alpha_{ij}$', './3_result_plots/', 'over_time_ABU_P_alphas_default_init', ylim=[-.1,0.9], norm="P")
    plot_mean_ABU_alphas_over_time(alphas_b5a, ts_b5a, layer_list, abu_af_list, r'$ABU_A$'+': mean '+r'$\alpha_{ij}$', './3_result_plots/', 'over_time_ABU_A_alphas_default_init', ylim=[-.4,.7], norm="A")
    plot_mean_ABU_alphas_over_time(alphas_b5s, ts_b5s, layer_list, abu_af_list, r'$ABU_S$'+': mean '+r'$\alpha_{ij}$', './3_result_plots/', 'over_time_ABU_S_alphas_default_init', ylim=[-.1,1.1], norm="S")
