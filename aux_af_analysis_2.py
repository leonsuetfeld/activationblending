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

def plot_mean_ABU_alphas_over_time(alpha_dict, ts_list, layer_list, af_list, title, saveplot_path, saveplot_filename, xlim=[0,60000], ylim=[-.3,.6]):
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
    fig = plt.figure(figsize=(25,5))
    for layer in range(n_layers):
        ax = fig.add_subplot(1,n_layers,layer+1)
        ax.plot([0,xlim[1]],[-.2,-.2], ':', color='#000000', linewidth='1', alpha=0.5)
        ax.plot([0,xlim[1]],[0.0,0.0], '-', color='#000000', linewidth='1', alpha=1.0)
        ax.plot([0,xlim[1]],[0.2,0.2], '-', color='#000000', linewidth='1', alpha=0.5)
        ax.plot([0,xlim[1]],[0.4,0.4], ':', color='#000000', linewidth='1', alpha=0.5)
        ax.plot([0,xlim[1]],[0.6,0.6], ':', color='#000000', linewidth='1', alpha=0.5)
        ax.plot([0,xlim[1]],[0.8,0.8], ':', color='#000000', linewidth='1', alpha=0.5)
        color_list = ['#66c2a5', '#fc8d62', '#8c9fcb', '#e78ac3', '#a5d853', '#b4b4b4', '#ffd82e', '#e4c494']
        for af in range(n_AFs):
            ax.plot(np.array(ts_list), alphas_by_layers[layer,:,af], linewidth=linewidth_default, color=color_list[af], label=af_list[af], alpha=1.0)
        ax.set_ylim(ylim)
        if layer == 0:
            ax.set_ylabel('mean '+r'$\alpha_i$')
        ax.set_xlabel('iteration')
        ax.set_title(layer_list[layer])
        ax.tick_params(axis='x', which='both', bottom='on', top='off')
        ax.set_xticklabels(['0','','20k','','40k','','60k'])
        ax.tick_params(axis='y', which='both', left='on', right='off')
        # ax.set_yticklabels(['0','','','','','10k'])
        if layer+1 == n_layers:
            handles, labels = ax.get_legend_handles_labels()
            lgd = ax.legend(handles, labels, loc='center left',  bbox_to_anchor=(1,0.5), fancybox=False, shadow=False)
    # save plot as image
    if not os.path.exists(saveplot_path):
        os.makedirs(saveplot_path)
    fig.savefig(saveplot_path+saveplot_filename, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=150)

def smcn_extract_wstats_1_and_10k(path_all_weights, spec_name, layer_list):

    # get list of files for spec
    files_of_spec = [f for f in os.listdir(path_all_weights) if (os.path.isfile(os.path.join(path_all_weights, f)) and ('.pkl' in f) and (spec_name in f))]
    files_of_spec.sort()

    # get number of runs and timeteps
    runs, timesteps = [], []
    for filename in files_of_spec:
        run = int(filename.split('_mb')[0].split('run_')[-1])
        if run not in runs:
            runs.append(run)
        timestep = int(filename.split('.')[0].split('mb_')[-1])
        if timestep not in timesteps:
            timesteps.append(timestep)
    n_runs, n_timesteps = len(runs), len(timesteps)

    # get relevant stats from files
    w_stats_t1, w_stats_t10000 = [], []
    for filename in files_of_spec:
        w_dict = pickle.load( open( path_all_weights+filename, "rb" ) )
        run = int(filename.split('_mb')[0].split('run_')[-1])
        timestep = int(filename.split('.')[0].split('mb_')[-1])
        # weight stats c1
        c1_means_per_fm = np.mean(w_dict['smcn/conv1/weights:0'], axis=(0,1,2))
        c1_stds_per_fm = np.std(w_dict['smcn/conv1/weights:0'], axis=(0,1,2))
        c1 = [np.mean(c1_means_per_fm), np.mean(c1_stds_per_fm)]
        c1_he_std = he_init_std(w_dict['smcn/conv1/weights:0'].shape)
        # weight stats c2
        c2_means_per_fm = np.mean(w_dict['smcn/conv2/weights:0'], axis=(0,1,2))
        c2_stds_per_fm = np.std(w_dict['smcn/conv2/weights:0'], axis=(0,1,2))
        c2 = [np.mean(c2_means_per_fm), np.mean(c2_stds_per_fm)]
        c2_he_std = he_init_std(w_dict['smcn/conv2/weights:0'].shape)
        # weight stats c3
        c3_means_per_fm = np.mean(w_dict['smcn/conv3/weights:0'], axis=(0,1,2))
        c3_stds_per_fm = np.std(w_dict['smcn/conv3/weights:0'], axis=(0,1,2))
        c3 = [np.mean(c3_means_per_fm), np.mean(c3_stds_per_fm)]
        c3_he_std = he_init_std(w_dict['smcn/conv3/weights:0'].shape)
        # weight stats c4
        c4_means_per_fm = np.mean(w_dict['smcn/conv4/weights:0'], axis=(0,1,2))
        c4_stds_per_fm = np.std(w_dict['smcn/conv4/weights:0'], axis=(0,1,2))
        c4 = [np.mean(c4_means_per_fm), np.mean(c4_stds_per_fm)]
        c4_he_std = he_init_std(w_dict['smcn/conv4/weights:0'].shape)
        # weight stats dense layers
        d5 = [np.mean(w_dict['smcn/dense5/weights:0']), np.std(w_dict['smcn/dense5/weights:0'])]
        d5_he_std = he_init_std(w_dict['smcn/dense5/weights:0'].shape)
        d6 = [np.mean(w_dict['smcn/dense6/weights:0']), np.std(w_dict['smcn/dense6/weights:0'])]
        d6_he_std = he_init_std(w_dict['smcn/dense6/weights:0'].shape)
        do = [np.mean(w_dict['smcn/denseout/weights:0']), np.std(w_dict['smcn/denseout/weights:0'])]
        do_he_std = he_init_std(w_dict['smcn/denseout/weights:0'].shape)
        # keep lists
        w_stats_in_file = [c1, c2, c3, c4, d5, d6, do]
        if timestep == 1:
            w_stats_t1.append(w_stats_in_file)
        elif timestep == 10000:
            w_stats_t10000.append(w_stats_in_file)
        he_stds = [c1_he_std, c2_he_std, c3_he_std, c4_he_std, d5_he_std, d6_he_std, do_he_std]
    # convert to np array
    w_stats_t1 = np.array(w_stats_t1)
    w_stats_t10000 = np.array(w_stats_t10000)
    he_stds = np.array(he_stds)
    # return mean over runs
    return np.mean(w_stats_t1, axis=0), np.mean(w_stats_t10000, axis=0), he_stds

def he_init_std(a_shape):
    N = 1
    if len(a_shape) == 4:
        a_shape = a_shape[:-1]
    for i in range(len(a_shape)):
        N *= a_shape[i]
    return np.sqrt(2./N)

def get_alpha_means(alpha_dict):
    c1 = np.mean(alpha_dict['conv1'][:,-1,0])
    c2 = np.mean(alpha_dict['conv2'][:,-1,0])
    c3 = np.mean(alpha_dict['conv3'][:,-1,0])
    c4 = np.mean(alpha_dict['conv4'][:,-1,0])
    d5 = np.mean(alpha_dict['dense5'][:,-1,0])
    d6 = np.mean(alpha_dict['dense6'][:,-1,0])
    return np.array([c1, c2, c3, c4, d5, d6])

# ##############################################################################
# ### SCRIPT DEFAULT INIT ######################################################
# ##############################################################################

path_af_weights = './3_output_cifar/ASC_main/0_af_weights/'
path_all_weights = './3_output_cifar/ASC_main/0_all_weights/'
layer_list = ['conv1', 'conv2', 'conv3', 'conv4', 'dense1', 'dense2']
abu_af_list = ['ReLU', 'ELU', 'tanh', 'Swish', 'I']

# extract weights from files
ts_linu, alphas_linu, _ = smcn_extract_af_weights_over_time(path_af_weights, 'alpha_I_', 60000, 60)
ts_tanh, alphas_tanh, _ = smcn_extract_af_weights_over_time(path_af_weights, 'alpha_tanh_', 60000, 60)
ts_relu, alphas_relu, _ = smcn_extract_af_weights_over_time(path_af_weights, 'alpha_relu_', 60000, 60)
ts_elu, alphas_elu, _ = smcn_extract_af_weights_over_time(path_af_weights, 'alpha_elu_', 60000, 60)
ts_selu, alphas_selu, _ = smcn_extract_af_weights_over_time(path_af_weights, 'alpha_selu_', 60000, 60)
ts_swish, alphas_swish, _ = smcn_extract_af_weights_over_time(path_af_weights, 'alpha_swish_', 60000, 60)

ts_b5u, alphas_b5u, betas_b5u = smcn_extract_af_weights_over_time(path_af_weights, 'ABU_T', 60000, 60)
ts_b5n, alphas_b5n, betas_b5n = smcn_extract_af_weights_over_time(path_af_weights, 'ABU_N_', 60000, 60)
ts_b5p, alphas_b5p, betas_b5p = smcn_extract_af_weights_over_time(path_af_weights, 'ABU_P_', 60000, 60)

# plot weights over time
plot_mean_alpha_by_layers_over_time(alphas_linu, ts_linu, layer_list, r'$\alpha_i$ over time ($\alpha$I)', './3_result_plots/', 'over_time_linu_alphas_default_init', ylim=[0.,1.2])
plot_mean_alpha_by_layers_over_time(alphas_relu, ts_relu, layer_list, r'$\alpha_i$ over time ($\alpha$ReLU)', './3_result_plots/', 'over_time_relu_alphas_default_init', ylim=[0.,1.2])
plot_mean_alpha_by_layers_over_time(alphas_tanh, ts_tanh, layer_list, r'$\alpha_i$ over time ($\alpha$tanh)', './3_result_plots/', 'over_time_tanh_alphas_default_init', ylim=[0.,2.4])
plot_mean_alpha_by_layers_over_time(alphas_elu, ts_elu, layer_list, r'$\alpha_i$ over time ($\alpha$ELU)', './3_result_plots/', 'over_time_elu_alphas_default_init', ylim=[0.,1.2])
plot_mean_alpha_by_layers_over_time(alphas_selu, ts_selu, layer_list, r'$\alpha_i$ over time ($\alpha$SELU)', './3_result_plots/', 'over_time_selu_alphas_default_init', ylim=[0.,1.2])
plot_mean_alpha_by_layers_over_time(alphas_swish, ts_swish, layer_list, r'$\alpha_i$ over time ($\alpha$Swish)', './3_result_plots/', 'over_time_swish_alphas_default_init', ylim=[0.,1.2])

plot_mean_ABU_alphas_over_time(alphas_b5u, ts_b5u, layer_list, abu_af_list, 'ABU', './3_result_plots/', 'over_time_ABU_alphas_default_init', ylim=[-.4,.6])
plot_mean_ABU_alphas_over_time(alphas_b5n, ts_b5n, layer_list, abu_af_list, 'ABU_N', './3_result_plots/', 'over_time_ABU_N_alphas_default_init', ylim=[-.4,.8])
plot_mean_ABU_alphas_over_time(alphas_b5p, ts_b5p, layer_list, abu_af_list, 'ABU_P', './3_result_plots/', 'over_time_ABU_P_alphas_default_init', ylim=[-.1,1.2])

"""
print('')
# identity: Do the regular weights compensate for the alphas?
linu_af_weights_final_stats = []
for key, value in alphas_linu.items():
    linu_af_weights_final_stats.append([key, np.mean(value[:,-1,0]), np.std(value[:,-1,0])])
w_stats_linu_start, w_stats_linu_end, he_stds = smcn_extract_wstats_1_and_10k(path_all_weights, '_linu_pretrain', layer_list)
std_linu_start = w_stats_linu_start[:,-1]
std_linu_end = w_stats_linu_end[:,-1]
alpha_means_linu_start = np.ones((6,1))
alpha_means_linu_end = get_alpha_means(alphas_linu)
print('identity:')
print('stds (he):         ', he_stds[1:])
print('std (start):       ', np.squeeze(std_linu_start[1:]))
print('alphas*std (end):  ', np.squeeze(alpha_means_linu_end)*std_linu_end[1:])
print('std (end):         ', np.squeeze(std_linu_end[1:]))
print('alphas (end):      ', np.squeeze(alpha_means_linu_end))
print('')

# tanh: Do the regular weights compensate for the alphas?
tanh_af_weights_final_stats = []
for key, value in alphas_tanh.items():
    tanh_af_weights_final_stats.append([key, np.mean(value[:,-1,0]), np.std(value[:,-1,0])])
w_stats_tanh_start, w_stats_tanh_end, he_stds = smcn_extract_wstats_1_and_10k(path_all_weights, '_tanh_pretrain', layer_list)
std_tanh_start = w_stats_tanh_start[:,-1]
std_tanh_end = w_stats_tanh_end[:,-1]
alpha_means_tanh_start = np.ones((6,1))
alpha_means_tanh_end = get_alpha_means(alphas_tanh)
print('tanh:')
print('stds (he):         ', he_stds[1:])
print('std (start):       ', np.squeeze(std_tanh_start[1:]))
print('alphas*std (end):  ', np.squeeze(alpha_means_tanh_end)*std_tanh_end[1:])
print('std (end):         ', np.squeeze(std_tanh_end[1:]))
print('alphas (end):      ', np.squeeze(alpha_means_tanh_end))
print('')

# ReLU: Do the regular weights compensate for the alphas?
relu_af_weights_final_stats = []
for key, value in alphas_relu.items():
    relu_af_weights_final_stats.append([key, np.mean(value[:,-1,0]), np.std(value[:,-1,0])])
w_stats_relu_start, w_stats_relu_end, he_stds = smcn_extract_wstats_1_and_10k(path_all_weights, '_relu_pretrain', layer_list)
std_relu_start = w_stats_relu_start[:,-1]
std_relu_end = w_stats_relu_end[:,-1]
alpha_means_relu_start = np.ones((6,1))
alpha_means_relu_end = get_alpha_means(alphas_relu)
print('ReLU:')
print('stds (he):         ', he_stds[1:])
print('std (start):       ', np.squeeze(std_relu_start[1:]))
print('alphas*std (end):  ', np.squeeze(alpha_means_relu_end)*std_relu_end[1:])
print('std (end):         ', np.squeeze(std_relu_end[1:]))
print('alphas (end):      ', np.squeeze(alpha_means_relu_end))
print('')

# ELU: Do the regular weights compensate for the alphas?
elu_af_weights_final_stats = []
for key, value in alphas_elu.items():
    elu_af_weights_final_stats.append([key, np.mean(value[:,-1,0]), np.std(value[:,-1,0])])
w_stats_elu_start, w_stats_elu_end, he_stds = smcn_extract_wstats_1_and_10k(path_all_weights, '_elu_pretrain', layer_list)
std_elu_start = w_stats_elu_start[:,-1]
std_elu_end = w_stats_elu_end[:,-1]
alpha_means_elu_start = np.ones((6,1))
alpha_means_elu_end = get_alpha_means(alphas_elu)
print('ELU:')
print('stds (he):         ', he_stds[1:])
print('std (start):       ', np.squeeze(std_elu_start[1:]))
print('alphas*std (end):  ', np.squeeze(alpha_means_elu_end)*std_elu_end[1:])
print('std (end):         ', np.squeeze(std_elu_end[1:]))
print('alphas (end):      ', np.squeeze(alpha_means_elu_end))
print('')
"""
# ##############################################################################
# ### SCRIPT PRE-TRAINED INIT ##################################################
# ##############################################################################
"""
path_af_weights = './2_output_cifar/SBF_9b/_af_weights/'
n_runs = 10

# extract weights from files
ts_linu, alphas_linu, _ = smcn_extract_af_weights_over_time(path_af_weights, '_linu_pretrained', n_runs)
ts_tanh, alphas_tanh, _ = smcn_extract_af_weights_over_time(path_af_weights, '_tanh_pretrained', n_runs)
ts_relu, alphas_relu, _ = smcn_extract_af_weights_over_time(path_af_weights, '_relu_pretrained', n_runs)
ts_elu, alphas_elu, _ = smcn_extract_af_weights_over_time(path_af_weights, '_elu_pretrained', n_runs)
ts_b5u, alphas_b5u, betas_b5u = smcn_extract_af_weights_over_time(path_af_weights, '_blend5_unrest_pretrained', n_runs)
ts_b5n, alphas_b5n, betas_b5n = smcn_extract_af_weights_over_time(path_af_weights, '_blend5_normalized_pretrained', n_runs)
ts_b5p, alphas_b5p, betas_b5p = smcn_extract_af_weights_over_time(path_af_weights, '_blend5_posnormed_pretrained', n_runs)

# plot weights over time
layer_list = ['conv1', 'conv2', 'conv3', 'conv4', 'dense1', 'dense2']
plot_mean_alpha_by_layers_over_time(alphas_linu, ts_linu, layer_list, r'$\alpha_i$ over time ($\alpha$I)', './2_result_plots/', 'over_time_linu_alphas_pretrained_init', ylim=[0.,1.2])
plot_mean_alpha_by_layers_over_time(alphas_relu, ts_relu, layer_list, r'$\alpha_i$ over time ($\alpha$ReLU)', './2_result_plots/', 'over_time_relu_alphas_pretrained_init', ylim=[0.,1.5])
plot_mean_alpha_by_layers_over_time(alphas_tanh, ts_tanh, layer_list, r'$\alpha_i$ over time ($\alpha$tanh)', './2_result_plots/', 'over_time_tanh_alphas_pretrained_init', ylim=[0.,2.0])
plot_mean_alpha_by_layers_over_time(alphas_elu, ts_elu, layer_list, r'$\alpha_i$ over time ($\alpha$ELU)', './2_result_plots/', 'over_time_elu_alphas_pretrained_init', ylim=[0.,1.2])
af_list = ['ReLU', 'ELU', 'tanh', 'Swish', 'I']
plot_mean_ABU_alphas_over_time(alphas_b5u, ts_b5u, layer_list, af_list, 'needs title', './2_result_plots/', 'over_time_b5u_alphas_pretrained_init', ylim=[-.4,.7])
plot_mean_ABU_alphas_over_time(alphas_b5n, ts_b5n, layer_list, af_list, 'needs title', './2_result_plots/', 'over_time_b5n_alphas_pretrained_init', ylim=[-.4,.8])
plot_mean_ABU_alphas_over_time(alphas_b5p, ts_b5p, layer_list, af_list, 'needs title', './2_result_plots/', 'over_time_b5p_alphas_pretrained_init', ylim=[-.2,.8])
"""
