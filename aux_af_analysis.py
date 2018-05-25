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

def smcn_extract_weights(folder_path, filenames):

    # initialize lists
    conv1_alpha, conv2_alpha, conv3_alpha, conv4_alpha, dense5_alpha, dense6_alpha = [], [], [], [], [], []
    conv1_swish, conv2_swish, conv3_swish, conv4_swish, dense5_swish, dense6_swish = [], [], [], [], [], []

    # go through files and put all weights in lists
    for i in range(len(filenames)):
        run_i = pickle.load( open( folder_path+filenames[i], "rb" ) )
        for key, value in run_i.items():
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

def af(xx, AF, swish_beta=1.):
    if AF == 'identity':
        x = np.copy(xx)
        return x
    elif AF == 'relu':
        x = np.copy(xx)
        x[x<0] = 0
        return x
    elif AF == 'elu':
        a = 1.
        x = np.copy(xx)
        x[x<0] = a*(np.exp(x[x<0])-1.)
        return x
    elif AF == 'selu':
        x = np.copy(xx)
        _alpha = 1.673263242354377284817042991671
        _lambda = 1.0507009873554804934193349852946
        x[x<0] = _alpha * np.exp(x[x<0]) - _alpha
        return x*_lambda
    elif AF == 'tanh':
        x = np.copy(xx)
        return np.tanh(x)
    elif AF == 'swish':
        x = np.copy(xx)
        return x/(1.+(np.exp(-(x*swish_beta))))

def smcn_print_mean_std(input_dict, info):
    print('')
    if input_dict['conv1'].shape[1] > 1:
        print(info + '           [mean] (sum of means) [std]')
        print(info + ' - conv1:  %s (sum: %.4f) | %s' %( str(np.mean(input_dict['conv1'], axis=0)), float(np.sum(np.mean(input_dict['conv1'], axis=0))), str(np.std(input_dict['conv1'], axis=0)) ))
        print(info + ' - conv2:  %s (sum: %.4f) | %s' %( str(np.mean(input_dict['conv2'], axis=0)), float(np.sum(np.mean(input_dict['conv2'], axis=0))), str(np.std(input_dict['conv2'], axis=0)) ))
        print(info + ' - conv3:  %s (sum: %.4f) | %s' %( str(np.mean(input_dict['conv3'], axis=0)), float(np.sum(np.mean(input_dict['conv3'], axis=0))), str(np.std(input_dict['conv3'], axis=0)) ))
        print(info + ' - conv4:  %s (sum: %.4f) | %s' %( str(np.mean(input_dict['conv4'], axis=0)), float(np.sum(np.mean(input_dict['conv4'], axis=0))), str(np.std(input_dict['conv4'], axis=0)) ))
        print(info + ' - dense5: %s (sum: %.4f) | %s' %( str(np.mean(input_dict['dense5'], axis=0)), float(np.sum(np.mean(input_dict['dense5'], axis=0))), str(np.std(input_dict['dense5'], axis=0)) ))
        print(info + ' - dense6: %s (sum: %.4f) | %s' %( str(np.mean(input_dict['dense6'], axis=0)), float(np.sum(np.mean(input_dict['dense6'], axis=0))), str(np.std(input_dict['dense6'], axis=0)) ))
    else:
        print(info + '           mean    | std')
        print(info + ' - conv1:  %.5f | %.5f' %(np.mean(input_dict['conv1'], axis=0), np.std(input_dict['conv1'], axis=0)))
        print(info + ' - conv2:  %.5f | %.5f' %(np.mean(input_dict['conv2'], axis=0), np.std(input_dict['conv2'], axis=0)))
        print(info + ' - conv3:  %.5f | %.5f' %(np.mean(input_dict['conv3'], axis=0), np.std(input_dict['conv3'], axis=0)))
        print(info + ' - conv4:  %.5f | %.5f' %(np.mean(input_dict['conv4'], axis=0), np.std(input_dict['conv4'], axis=0)))
        print(info + ' - dense5: %.5f | %.5f' %(np.mean(input_dict['dense5'], axis=0), np.std(input_dict['dense5'], axis=0)))
        print(info + ' - dense6: %.5f | %.5f' %(np.mean(input_dict['dense6'], axis=0), np.std(input_dict['dense6'], axis=0)))
    print('')

def plot_mean_alpha_by_layers(af_list, name_list, title, saveplot_path, saveplot_filename, includes_beta=False):
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
    linewidth_default = '2'
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111)
    ax.plot([0,7],[1,1], '-', color='#000000', linewidth='1', alpha=0.5)
    ax.plot([1,1],[-9,9], ':', color='#000000', linewidth='1', alpha=0.3)
    ax.plot([2,2],[-9,9], ':', color='#000000', linewidth='1', alpha=0.3)
    ax.plot([3,3],[-9,9], ':', color='#000000', linewidth='1', alpha=0.3)
    ax.plot([4,4],[-9,9], ':', color='#000000', linewidth='1', alpha=0.3)
    ax.plot([5,5],[-9,9], ':', color='#000000', linewidth='1', alpha=0.3)
    ax.plot([6,6],[-9,9], ':', color='#000000', linewidth='1', alpha=0.3)
    color_list = ['#66c2a5', '#fc8d62', '#8c9fcb', '#e78ac3', '#a5d853', '#b4b4b4', '#ffd82e', '#e4c494']
    x = np.arange(6)+1
    for af in range(n_afs):
        ax.plot(x, afs_by_layers_means[af,:], linewidth=linewidth_default, color=color_list[af], label=name_list[af], alpha=1.0, marker='.', markersize=12)
    if includes_beta:
        ax.plot(x, afs_by_layers_means[-1,:], '--', linewidth=linewidth_default, color=color_list[af], label=name_list[n_afs], alpha=1.0, marker='.', markersize=12)
    ax.set_ylim([0.0,1.4])
    ax.set_xlim([0.8,6.2])
    ax.set_xticklabels(['','conv1','conv2','conv3','conv4','dense1','dense2'])
    ax.set_ylabel('mean '+r'$\alpha_i$')
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='center left',  bbox_to_anchor=(1,0.5), fancybox=False, shadow=False)
    ax.tick_params(axis='x', which='both', bottom='off', top='off')
    # save plot as image
    if not os.path.exists(saveplot_path):
        os.makedirs(saveplot_path)
    fig.savefig(saveplot_path+saveplot_filename, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=150)
    plt.close()

def plot_all_runs_alphas(af_dict, title, saveplot_path, saveplot_filename, af_dict_2={}, af_dict_3={}, beta={}, beta_2={}, beta_3={}, ylim=[0.0,1.4]):
    n_runs = len(af_dict['conv1'])
    # extract means from afs
    alphas_by_layers = np.zeros((n_runs,6))
    if af_dict_2:
        alphas_by_layers_2 = np.zeros((n_runs,6))
    if af_dict_3:
        alphas_by_layers_3 = np.zeros((n_runs,6))
    for run in range(n_runs):
        alphas_by_layers[run,0] = af_dict['conv1'][run]
        alphas_by_layers[run,1] = af_dict['conv2'][run]
        alphas_by_layers[run,2] = af_dict['conv3'][run]
        alphas_by_layers[run,3] = af_dict['conv4'][run]
        alphas_by_layers[run,4] = af_dict['dense5'][run]
        alphas_by_layers[run,5] = af_dict['dense6'][run]
        if af_dict_2:
            alphas_by_layers_2[run,0] = af_dict_2['conv1'][run]
            alphas_by_layers_2[run,1] = af_dict_2['conv2'][run]
            alphas_by_layers_2[run,2] = af_dict_2['conv3'][run]
            alphas_by_layers_2[run,3] = af_dict_2['conv4'][run]
            alphas_by_layers_2[run,4] = af_dict_2['dense5'][run]
            alphas_by_layers_2[run,5] = af_dict_2['dense6'][run]
        if af_dict_3:
            alphas_by_layers_3[run,0] = af_dict_3['conv1'][run]
            alphas_by_layers_3[run,1] = af_dict_3['conv2'][run]
            alphas_by_layers_3[run,2] = af_dict_3['conv3'][run]
            alphas_by_layers_3[run,3] = af_dict_3['conv4'][run]
            alphas_by_layers_3[run,4] = af_dict_3['dense5'][run]
            alphas_by_layers_3[run,5] = af_dict_3['dense6'][run]
    if beta:
        betas_by_layers = np.zeros((n_runs,6))
        if beta_2:
            betas_by_layers_2 = np.zeros((n_runs,6))
        if beta_3:
            betas_by_layers_3 = np.zeros((n_runs,6))
        for run in range(n_runs):
            betas_by_layers[run,0] = beta['conv1'][run]
            betas_by_layers[run,1] = beta['conv2'][run]
            betas_by_layers[run,2] = beta['conv3'][run]
            betas_by_layers[run,3] = beta['conv4'][run]
            betas_by_layers[run,4] = beta['dense5'][run]
            betas_by_layers[run,5] = beta['dense6'][run]
            if beta_2:
                betas_by_layers_2[run,0] = beta_2['conv1'][run]
                betas_by_layers_2[run,1] = beta_2['conv2'][run]
                betas_by_layers_2[run,2] = beta_2['conv3'][run]
                betas_by_layers_2[run,4] = beta_2['dense5'][run]
                betas_by_layers_2[run,3] = beta_2['conv4'][run]
                betas_by_layers_2[run,5] = beta_2['dense6'][run]
            if beta_3:
                betas_by_layers_3[run,0] = beta_3['conv1'][run]
                betas_by_layers_3[run,1] = beta_3['conv2'][run]
                betas_by_layers_3[run,2] = beta_3['conv3'][run]
                betas_by_layers_3[run,4] = beta_3['dense5'][run]
                betas_by_layers_3[run,3] = beta_3['conv4'][run]
                betas_by_layers_3[run,5] = beta_3['dense6'][run]
    # figure
    linewidth_default = '1'
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111)
    ax.plot([0,7],[1,1], '-', color='#000000', linewidth='1', alpha=0.5)
    ax.plot([1,1],[-9,9], ':', color='#000000', linewidth='1', alpha=0.3)
    ax.plot([2,2],[-9,9], ':', color='#000000', linewidth='1', alpha=0.3)
    ax.plot([3,3],[-9,9], ':', color='#000000', linewidth='1', alpha=0.3)
    ax.plot([4,4],[-9,9], ':', color='#000000', linewidth='1', alpha=0.3)
    ax.plot([5,5],[-9,9], ':', color='#000000', linewidth='1', alpha=0.3)
    ax.plot([6,6],[-9,9], ':', color='#000000', linewidth='1', alpha=0.3)
    x = np.arange(6)+1
    if beta:
        for run in range(n_runs):
            ax.plot(x, betas_by_layers[run,:], linewidth=linewidth_default, color='orange', alpha=0.3)
    if beta_2:
        for run in range(n_runs):
            ax.plot(x, betas_by_layers_2[run,:], linewidth=linewidth_default, color='red', alpha=0.3)
    if beta_3:
        for run in range(n_runs):
            ax.plot(x, betas_by_layers_3[run,:], linewidth=linewidth_default, color='pink', alpha=0.3)
    for run in range(n_runs):
        ax.plot(x, alphas_by_layers[run,:], linewidth=linewidth_default, color='blue', alpha=0.3)
    if af_dict_2:
        for run in range(n_runs):
            ax.plot(x, alphas_by_layers_2[run,:], linewidth=linewidth_default, color='green', alpha=0.3)
    if af_dict_3:
        for run in range(n_runs):
            ax.plot(x, alphas_by_layers_3[run,:], linewidth=linewidth_default, color='black', alpha=0.3)
    ax.set_ylim(ylim)
    ax.set_xlim([0.8,6.2])
    ax.set_xticklabels(['','c1','c2','c3','c4','d1','d2'])
    ax.set_ylabel('mean '+r'$\alpha_i$')
    ax.set_title(title)
    ax.tick_params(axis='x', which='both', bottom='off', top='off')
    # save plot as image
    if not os.path.exists(saveplot_path):
        os.makedirs(saveplot_path)
    fig.savefig(saveplot_path+saveplot_filename, bbox_inches='tight', dpi=150)
    plt.close()

def plot_all_runs_alphas_multi(af_dict_list, title_list, saveplot_path, saveplot_filename, af_dict_list_2=[], af_dict_list_3=[], beta_list=[], beta_list_2=[], beta_list_3=[], ylim_list=[[0.0,1.4]]):
    n_runs = len(af_dict_list[0]['conv1'])
    while len(ylim_list) < n_runs:
        ylim_list.append(ylim_list[0])
    linewidth_default = '1'
    fig = plt.figure(figsize=(14,3))

    for af_num, af_dict in enumerate(af_dict_list):

        # extract other dicts
        if len(af_dict_list_2) > 0:
            af_dict_2 = af_dict_list_2[af_num]
        if len(af_dict_list_3) > 0:
            af_dict_3 = af_dict_list_3[af_num]
        if len(beta_list) > 0:
            beta = beta_list[af_num]
        if len(beta_list_2) > 0:
            beta_2 = beta_list_2[af_num]
        if len(beta_list_3) > 0:
            beta_3 = beta_list_3[af_num]

        # extract means from afs
        alphas_by_layers = np.zeros((n_runs,6))
        if len(af_dict_list_2) > 0:
            alphas_by_layers_2 = np.zeros((n_runs,6))
        if len(af_dict_list_3) > 0:
            alphas_by_layers_3 = np.zeros((n_runs,6))
        for run in range(n_runs):
            alphas_by_layers[run,0] = af_dict['conv1'][run]
            alphas_by_layers[run,1] = af_dict['conv2'][run]
            alphas_by_layers[run,2] = af_dict['conv3'][run]
            alphas_by_layers[run,3] = af_dict['conv4'][run]
            alphas_by_layers[run,4] = af_dict['dense5'][run]
            alphas_by_layers[run,5] = af_dict['dense6'][run]
            if len(af_dict_list_2) > 0:
                alphas_by_layers_2[run,0] = af_dict_2['conv1'][run]
                alphas_by_layers_2[run,1] = af_dict_2['conv2'][run]
                alphas_by_layers_2[run,2] = af_dict_2['conv3'][run]
                alphas_by_layers_2[run,3] = af_dict_2['conv4'][run]
                alphas_by_layers_2[run,4] = af_dict_2['dense5'][run]
                alphas_by_layers_2[run,5] = af_dict_2['dense6'][run]
            if len(af_dict_list_3) > 0:
                alphas_by_layers_3[run,0] = af_dict_3['conv1'][run]
                alphas_by_layers_3[run,1] = af_dict_3['conv2'][run]
                alphas_by_layers_3[run,2] = af_dict_3['conv3'][run]
                alphas_by_layers_3[run,3] = af_dict_3['conv4'][run]
                alphas_by_layers_3[run,4] = af_dict_3['dense5'][run]
                alphas_by_layers_3[run,5] = af_dict_3['dense6'][run]
        if len(beta_list) > 0 and not beta is None:
            betas_by_layers = np.zeros((n_runs,6))
            if len(beta_list_2) > 0 and not beta_2 is None:
                betas_by_layers_2 = np.zeros((n_runs,6))
            if len(beta_list_3) > 0 and not beta_3 is None:
                betas_by_layers_3 = np.zeros((n_runs,6))
            for run in range(n_runs):
                betas_by_layers[run,0] = beta['conv1'][run]
                betas_by_layers[run,1] = beta['conv2'][run]
                betas_by_layers[run,2] = beta['conv3'][run]
                betas_by_layers[run,3] = beta['conv4'][run]
                betas_by_layers[run,4] = beta['dense5'][run]
                betas_by_layers[run,5] = beta['dense6'][run]
                if len(beta_list_2) > 0 and not beta_2 is None:
                    betas_by_layers_2[run,0] = beta_2['conv1'][run]
                    betas_by_layers_2[run,1] = beta_2['conv2'][run]
                    betas_by_layers_2[run,2] = beta_2['conv3'][run]
                    betas_by_layers_2[run,4] = beta_2['dense5'][run]
                    betas_by_layers_2[run,3] = beta_2['conv4'][run]
                    betas_by_layers_2[run,5] = beta_2['dense6'][run]
                if len(beta_list_3) > 0 and not beta_2 is None:
                    betas_by_layers_3[run,0] = beta_3['conv1'][run]
                    betas_by_layers_3[run,1] = beta_3['conv2'][run]
                    betas_by_layers_3[run,2] = beta_3['conv3'][run]
                    betas_by_layers_3[run,4] = beta_3['dense5'][run]
                    betas_by_layers_3[run,3] = beta_3['conv4'][run]
                    betas_by_layers_3[run,5] = beta_3['dense6'][run]
        # figure
        ax = fig.add_subplot(1,6,af_num+1)
        ax.plot([0,7],[1,1], '-', color='#000000', linewidth='1', alpha=0.5)
        ax.plot([1,1],[-9,9], ':', color='#000000', linewidth='1', alpha=0.3)
        ax.plot([2,2],[-9,9], ':', color='#000000', linewidth='1', alpha=0.3)
        ax.plot([3,3],[-9,9], ':', color='#000000', linewidth='1', alpha=0.3)
        ax.plot([4,4],[-9,9], ':', color='#000000', linewidth='1', alpha=0.3)
        ax.plot([5,5],[-9,9], ':', color='#000000', linewidth='1', alpha=0.3)
        ax.plot([6,6],[-9,9], ':', color='#000000', linewidth='1', alpha=0.3)
        x = np.arange(6)+1
        if len(beta_list) > 0 and not beta is None:
            for run in range(n_runs):
                ax.plot(x, betas_by_layers[run,:], linewidth=linewidth_default, color='orange', alpha=0.3)
        if len(beta_list_2) > 0 and not beta_2 is None:
            for run in range(n_runs):
                ax.plot(x, betas_by_layers_2[run,:], linewidth=linewidth_default, color='red', alpha=0.3)
        if len(beta_list_3) > 0 and not beta_3 is None:
            for run in range(n_runs):
                ax.plot(x, betas_by_layers_3[run,:], linewidth=linewidth_default, color='pink', alpha=0.3)
        for run in range(n_runs):
            ax.plot(x, alphas_by_layers[run,:], linewidth=linewidth_default, color='blue', alpha=0.3)
        if len(af_dict_list_2) > 0:
            for run in range(n_runs):
                ax.plot(x, alphas_by_layers_2[run,:], linewidth=linewidth_default, color='green', alpha=0.3)
        if len(af_dict_list_3) > 0:
            for run in range(n_runs):
                ax.plot(x, alphas_by_layers_3[run,:], linewidth=linewidth_default, color='black', alpha=0.3)
        ax.set_ylim(ylim_list[af_num])
        ax.set_xlim([0.8,6.2])
        ax.set_xticklabels(['','c1','c2','c3','c4','d1','d2'])
        if af_num == 0:
            ax.set_ylabel('mean '+r'$\alpha_i$')
        # else:
        #     ax.set_yticklabels([])
        ax.set_title(title_list[af_num])
        ax.tick_params(axis='x', which='both', bottom='off', top='off')
    # save plot as image
    plt.tight_layout()
    if not os.path.exists(saveplot_path):
        os.makedirs(saveplot_path)
    fig.savefig(saveplot_path+saveplot_filename, bbox_inches='tight', dpi=150)
    plt.close()

def plot_mean_alpha_by_layers_ABU(AF_names, afs_by_layers_means, title, saveplot_path, saveplot_filename, ylim=[-0.4,0.8], swish_betas_by_layers_means=np.array([0])):
    linewidth_default = '2'
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.plot([0,7],[1./5.,1./5.], '-', color='#000000', linewidth='1', alpha=0.5)
    ax.plot([0,7],[0.,0.], '-', color='#000000', linewidth='1', alpha=0.5)
    ax.plot([1,1],[-9,9], ':', color='#000000', linewidth='1', alpha=0.3)
    ax.plot([2,2],[-9,9], ':', color='#000000', linewidth='1', alpha=0.3)
    ax.plot([3,3],[-9,9], ':', color='#000000', linewidth='1', alpha=0.3)
    ax.plot([4,4],[-9,9], ':', color='#000000', linewidth='1', alpha=0.3)
    ax.plot([5,5],[-9,9], ':', color='#000000', linewidth='1', alpha=0.3)
    ax.plot([6,6],[-9,9], ':', color='#000000', linewidth='1', alpha=0.3)
    color_list = ['#8c9fcb', '#e78ac3', '#fc8d62', '#b4b4b4', '#66c2a5', '#a5d853', '#ffd82e', '#e4c494']
    x = np.arange(6)+1
    if swish_betas_by_layers_means.shape[0]>1:
        ax2 = ax.twinx()
        ax2.plot(x, swish_betas_by_layers_means, '--', linewidth=linewidth_default, color=color_list[3], label=r'$Swish\ \beta$', marker='.', markersize=12, alpha=1.0)
        ax2.set_ylabel(r'$Swish\ \beta$')
        ax2.set_ylim([0.0,1.3])
        ax2.set_xlim([0.8,6.2])
    for af in range(len(AF_names)):
        ax.plot(x, afs_by_layers_means[:,af], linewidth=linewidth_default, color=color_list[af], label=AF_names[af], alpha=1.0, marker='.', markersize=14)
    ax.set_ylim(ylim)
    ax.set_xlim([0.8,6.2])
    ax.set_xticklabels(['','conv1','conv2','conv3','conv4','dense1','dense2'])
    ax.set_ylabel('mean '+r'$\alpha_i$')
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    lgd = ax.legend(handles+handles2, labels+labels2, loc='lower left',  bbox_to_anchor=(1.05,0.0), fancybox=False, shadow=False)
    ax.tick_params(axis='x', which='both', bottom='off', top='off')
    # save plot as image
    if not os.path.exists(saveplot_path):
        os.makedirs(saveplot_path)
    fig.savefig(saveplot_path+saveplot_filename, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=150)
    plt.close()

def get_mean_alphas_by_layers_ABU(weight_dict, swish_beta={}):
    AF_names = [r'$ReLU$', r'$ELU$', r'$tanh$', r'$Swish$', r'$I$'] # this order depends on the allocation of weights to AFs in the network. look it up from the networks blend-AF definition in activate().
    n_layers = 6
    n_afs = len(AF_names)
    # extracting alpha means
    afs_by_layers_means = np.zeros((n_layers,n_afs))
    afs_by_layers_means[0,:] = np.mean(weight_dict['conv1'], axis=0)
    afs_by_layers_means[1,:] = np.mean(weight_dict['conv2'], axis=0)
    afs_by_layers_means[2,:] = np.mean(weight_dict['conv3'], axis=0)
    afs_by_layers_means[3,:] = np.mean(weight_dict['conv4'], axis=0)
    afs_by_layers_means[4,:] = np.mean(weight_dict['dense5'], axis=0)
    afs_by_layers_means[5,:] = np.mean(weight_dict['dense6'], axis=0)
    # extracting swish_beta means
    swish_betas_by_layers_means = np.zeros((n_layers,))
    if swish_beta:
        swish_betas_by_layers_means = np.zeros((n_layers,))
        swish_betas_by_layers_means[0] = np.mean(swish_beta['conv1'], axis=0)[0]
        swish_betas_by_layers_means[1] = np.mean(swish_beta['conv2'], axis=0)[0]
        swish_betas_by_layers_means[2] = np.mean(swish_beta['conv3'], axis=0)[0]
        swish_betas_by_layers_means[3] = np.mean(swish_beta['conv4'], axis=0)[0]
        swish_betas_by_layers_means[4] = np.mean(swish_beta['dense5'], axis=0)[0]
        swish_betas_by_layers_means[5] = np.mean(swish_beta['dense6'], axis=0)[0]
    return AF_names, afs_by_layers_means, swish_betas_by_layers_means

def plot_ABU(mean_alphas, mean_swish_beta, saveplot_path, saveplot_filename, ylim=[-2.0, 2.0], col='black', norm="None"):
    mean_alphas = np.copy(mean_alphas)
    mean_swish_beta = np.copy(mean_swish_beta)
    assert mean_alphas.shape[0] == 6 and mean_alphas.shape[1] == 5, 'got wrong mean_alphas matrix. expected [6 layers x 5 AFs].'
    fig = plt.figure(figsize=(11.44,2.3))
    show_range = [-2.,2.,.01]
    layer_names=['conv1','conv2','conv3','conv4','dense1','dense2']
    for layer_i in range(6):
        # calculate AF
        vec_alpha_layer_i = mean_alphas[layer_i,:]
        x = np.arange(show_range[0], show_range[1], show_range[2])
        if norm == 'N':
            vec_alpha_layer_i = vec_alpha_layer_i/np.sum(vec_alpha_layer_i)
        elif norm == 'P':
            vec_alpha_layer_i = np.clip(vec_alpha_layer_i, 0.0001, 1000.0)
            vec_alpha_layer_i = vec_alpha_layer_i/np.sum(vec_alpha_layer_i)
        if norm == 'A':
            vec_alpha_layer_i = vec_alpha_layer_i/np.sum(np.abs(vec_alpha_layer_i))
        if norm == 'S':
            vec_alpha_layer_i = np.exp(vec_alpha_layer_i)
            vec_alpha_layer_i = vec_alpha_layer_i/np.sum(vec_alpha_layer_i)
        resulting_AF_layer_i = np.zeros(x.shape)
        resulting_AF_layer_i += af(x, 'relu')*vec_alpha_layer_i[0]
        resulting_AF_layer_i += af(x, 'elu')*vec_alpha_layer_i[1]
        resulting_AF_layer_i += af(x, 'tanh')*vec_alpha_layer_i[2]
        resulting_AF_layer_i += af(x, 'swish', mean_swish_beta[layer_i])*vec_alpha_layer_i[3]
        resulting_AF_layer_i += af(x, 'identity')*vec_alpha_layer_i[4]
        # plot AF
        ax = fig.add_subplot(1,6,layer_i+1)
        plt.plot([-6.,6.],[0,0], '--', linewidth=1, color='black', alpha=0.5)
        plt.plot([0,0],[-100,100], '--', linewidth=1, color='black', alpha=0.5)
        plt.plot(x,resulting_AF_layer_i, linewidth='2', color=col)
        plt.xlim(show_range[0], show_range[1])
        plt.ylim(ylim)
        plt.title(layer_names[layer_i])
        if layer_i > 0:
            plt.tick_params(axis='y', which='both', labelleft='off') # labels along the left and right off
        # ax.set_xticklabels(['-2','','-1','','0','','1','','2'])
        plt.locator_params(axis='y', nbins=5)
        plt.locator_params(axis='x', nbins=5)
    plt.tight_layout()
    # save plot as image
    if not os.path.exists(saveplot_path):
        os.makedirs(saveplot_path)
    fig.savefig(saveplot_path+saveplot_filename, dpi=150)
    plt.close()

def plot_ABU2(mean_alphas_C10, mean_swish_beta_C10, mean_alphas_C100, mean_swish_beta_C100, saveplot_path, saveplot_filename, normalize_weights=False):
    mean_alphas_C10 = np.copy(mean_alphas_C10)
    mean_swish_beta_C10 = np.copy(mean_swish_beta_C10)
    mean_alphas_C100 = np.copy(mean_alphas_C100)
    mean_swish_beta_C100 = np.copy(mean_swish_beta_C100)
    assert mean_alphas_C10.shape[0] == 6 and mean_alphas_C10.shape[1] == 5, 'got wrong mean_alphas matrix. expected [6 layers x 5 AFs].'
    assert mean_alphas_C100.shape[0] == 6 and mean_alphas_C100.shape[1] == 5, 'got wrong mean_alphas matrix. expected [6 layers x 5 AFs].'
    fig = plt.figure(figsize=(18,3.3))
    show_range = [-4.,4.,.01]
    layer_names=['conv1','conv2','conv3','conv4','dense1','dense2']
    for layer_i in range(6):
        x = np.arange(show_range[0], show_range[1], show_range[2])
        # calculate AF C10
        vec_alpha_layer_i_C10 = mean_alphas_C10[layer_i,:]
        if normalize_weights:
            vec_alpha_layer_i_C10 /= np.sum(vec_alpha_layer_i_C10)
        resulting_AF_layer_i_C10 = np.zeros(x.shape)
        resulting_AF_layer_i_C10 += af(x, 'relu')*vec_alpha_layer_i_C10[0]
        resulting_AF_layer_i_C10 += af(x, 'elu')*vec_alpha_layer_i_C10[1]
        resulting_AF_layer_i_C10 += af(x, 'tanh')*vec_alpha_layer_i_C10[2]
        resulting_AF_layer_i_C10 += af(x, 'swish', mean_swish_beta_C10[layer_i])*vec_alpha_layer_i_C10[3]
        resulting_AF_layer_i_C10 += af(x, 'identity')*vec_alpha_layer_i_C10[4]
        # calculate AF C100
        vec_alpha_layer_i_C100 = mean_alphas_C100[layer_i,:]
        if normalize_weights:
            vec_alpha_layer_i_C100 /= np.sum(vec_alpha_layer_i_C100)
        resulting_AF_layer_i_C100 = np.zeros(x.shape)
        resulting_AF_layer_i_C100 += af(x, 'relu')*vec_alpha_layer_i_C100[0]
        resulting_AF_layer_i_C100 += af(x, 'elu')*vec_alpha_layer_i_C100[1]
        resulting_AF_layer_i_C100 += af(x, 'tanh')*vec_alpha_layer_i_C100[2]
        resulting_AF_layer_i_C100 += af(x, 'swish', mean_swish_beta_C100[layer_i])*vec_alpha_layer_i_C100[3]
        resulting_AF_layer_i_C100 += af(x, 'identity')*vec_alpha_layer_i_C100[4]
        # plot AF
        ax = fig.add_subplot(1,6,layer_i+1)
        plt.plot([-6.,6.],[0,0], '--', linewidth=1, color='black', alpha=0.5)
        plt.plot([0,0],[-100,100], '--', linewidth=1, color='black', alpha=0.5)
        plt.plot(x,resulting_AF_layer_i_C100, linewidth='3', color='blue', alpha=0.5, label='CIFAR100')
        plt.plot(x,resulting_AF_layer_i_C10, linewidth='3', color='orange', alpha=1.0, label='CIFAR10')
        plt.xlim(show_range[0], show_range[1])
        plt.ylim(-1.5, 2.5)
        plt.title(layer_names[layer_i])
        if layer_i > 0:
            plt.tick_params(axis='y', which='both', labelleft='off') # labels along the left and right off
        if layer_i == 5:
            handles, labels = ax.get_legend_handles_labels()
            lgd = ax.legend(handles, labels, loc='upper left',  bbox_to_anchor=(1,1), fancybox=False, shadow=False)
    plt.tight_layout()
    # save plot as image
    if not os.path.exists(saveplot_path):
        os.makedirs(saveplot_path)
    fig.savefig(saveplot_path+saveplot_filename, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=150)
    plt.close()

# ##############################################################################
# ### SCRIPT ###################################################################
# ##############################################################################

steps = [2]

# ##############################################################################
# ### SCRIPT ASU ###############################################################
# ##############################################################################

if 1 in steps:

    path_finalweights = './3_output_cifar/ASC_main/0_af_weights/'
    scaled_AFs_figname = 'MAIN_mean_alphas_ASU.png'

    # get files from folder
    mb_step = 60000
    mb_step_2 = 6103
    mb_step_3 = 1018
    linu_fw_files = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('_alpha_I_' in f) and ('_mb_'+str(mb_step) in f))]
    linu_fw_files_2 = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('_alpha_I_' in f) and ('_mb_'+str(mb_step_2) in f))]
    linu_fw_files_3 = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('_alpha_I_' in f) and ('_mb_'+str(mb_step_3) in f))]
    print('- created list of linu final weight files at minibatches %i, %i, and %i (%i runs in total)' %(mb_step, mb_step_2, mb_step_3, len(linu_fw_files)))
    tanh_fw_files = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('alpha_tanh_' in f) and ('_mb_'+str(mb_step) in f))]
    tanh_fw_files_2 = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('alpha_tanh_' in f) and ('_mb_'+str(mb_step_2) in f))]
    tanh_fw_files_3 = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('alpha_tanh_' in f) and ('_mb_'+str(mb_step_3) in f))]
    print('- created list of tanh final weight files at minibatches %i, %i, and %i (%i runs in total)' %(mb_step, mb_step_2, mb_step_3, len(tanh_fw_files)))
    relu_fw_files = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('alpha_relu_' in f) and ('_mb_'+str(mb_step) in f))]
    relu_fw_files_2 = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('alpha_relu_' in f) and ('_mb_'+str(mb_step_2) in f))]
    relu_fw_files_3 = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('alpha_relu_' in f) and ('_mb_'+str(mb_step_3) in f))]
    print('- created list of relu final weight files at minibatches %i, %i, and %i (%i runs in total)' %(mb_step, mb_step_2, mb_step_3, len(relu_fw_files)))
    elu_fw_files = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('alpha_elu_' in f) and ('_mb_'+str(mb_step) in f))]
    elu_fw_files_2 = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('alpha_elu_' in f) and ('_mb_'+str(mb_step_2) in f))]
    elu_fw_files_3 = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('alpha_elu_' in f) and ('_mb_'+str(mb_step_3) in f))]
    print('- created list of elu final weight files at minibatches %i, %i, and %i (%i runs in total)' %(mb_step, mb_step_2, mb_step_3, len(elu_fw_files)))
    selu_fw_files = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('alpha_selu_' in f) and ('_mb_'+str(mb_step) in f))]
    selu_fw_files_2 = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('alpha_selu_' in f) and ('_mb_'+str(mb_step_2) in f))]
    selu_fw_files_3 = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('alpha_selu_' in f) and ('_mb_'+str(mb_step_3) in f))]
    print('- created list of selu final weight files at minibatches %i, %i, and %i (%i runs in total)' %(mb_step, mb_step_2, mb_step_3, len(selu_fw_files)))
    swish_fw_files = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('alpha_swish_' in f) and ('_mb_'+str(mb_step) in f))]
    swish_fw_files_2 = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('alpha_swish_' in f) and ('_mb_'+str(mb_step_2) in f))]
    swish_fw_files_3 = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('alpha_swish_' in f) and ('_mb_'+str(mb_step_3) in f))]
    print('- created list of swish final weight files at minibatches %i, %i, and %i (%i runs in total)' %(mb_step, mb_step_2, mb_step_3, len(swish_fw_files)))

    # extract weights from files
    linu_wd, _ = smcn_extract_weights(path_finalweights, linu_fw_files)
    linu_wd_2, _ = smcn_extract_weights(path_finalweights, linu_fw_files_2)
    linu_wd_3, _ = smcn_extract_weights(path_finalweights, linu_fw_files_3)
    tanh_wd, _ = smcn_extract_weights(path_finalweights, tanh_fw_files)
    tanh_wd_2, _ = smcn_extract_weights(path_finalweights, tanh_fw_files_2)
    tanh_wd_3, _ = smcn_extract_weights(path_finalweights, tanh_fw_files_3)
    relu_wd, _ = smcn_extract_weights(path_finalweights, relu_fw_files)
    relu_wd_2, _ = smcn_extract_weights(path_finalweights, relu_fw_files_2)
    relu_wd_3, _ = smcn_extract_weights(path_finalweights, relu_fw_files_3)
    elu_wd, _ = smcn_extract_weights(path_finalweights, elu_fw_files)
    elu_wd_2, _ = smcn_extract_weights(path_finalweights, elu_fw_files_2)
    elu_wd_3, _ = smcn_extract_weights(path_finalweights, elu_fw_files_3)
    selu_wd, _ = smcn_extract_weights(path_finalweights, selu_fw_files)
    selu_wd_2, _ = smcn_extract_weights(path_finalweights, selu_fw_files_2)
    selu_wd_3, _ = smcn_extract_weights(path_finalweights, selu_fw_files_3)
    swish_wd, swish_swishbeta = smcn_extract_weights(path_finalweights, swish_fw_files)
    swish_wd_2, swish_swishbeta_2 = smcn_extract_weights(path_finalweights, swish_fw_files_2)
    swish_wd_3, swish_swishbeta_3 = smcn_extract_weights(path_finalweights, swish_fw_files_3)

    # print mean and std for all weights
    smcn_print_mean_std(linu_wd, 'alpha-I alpha')
    smcn_print_mean_std(tanh_wd, 'alpha-tanh alpha')
    smcn_print_mean_std(relu_wd, 'alpha-ReLU alpha')
    smcn_print_mean_std(elu_wd, 'alpha-ELU alpha')
    smcn_print_mean_std(selu_wd, 'alpha-SELU alpha')
    smcn_print_mean_std(swish_wd, 'alpha-Swish alpha')
    smcn_print_mean_std(swish_swishbeta, 'alpha-Swish beta')

    # plot all alphas for individual AFs (only plot final states)
    af_dict_list = [linu_wd, tanh_wd, relu_wd, elu_wd, selu_wd, swish_wd]
    beta_list = [None, None, None, None, None, swish_swishbeta]
    title_list = [r'$\alpha I$', r'$\alpha tanh$', r'$\alpha ReLU$', r'$\alpha ELU$', r'$\alpha SELU$', r'$\alpha Swish$']
    plot_all_runs_alphas_multi(af_dict_list, title_list, './3_result_plots/', 'MAIN_final_alpha_all_1.png', beta_list=beta_list)
    # plot_all_runs_alphas(linu_wd, r'$\alpha I$', './3_result_plots/', 'MAIN_final_alpha_I.png')
    # plot_all_runs_alphas(tanh_wd, r'$\alpha tanh$', './3_result_plots/', 'MAIN_final_alpha_tanh.png') # ylim=[0.0,1.8]
    # plot_all_runs_alphas(relu_wd, r'$\alpha ReLU$', './3_result_plots/', 'MAIN_final_alpha_ReLU.png')
    # plot_all_runs_alphas(elu_wd, r'$\alpha ELU$', './3_result_plots/', 'MAIN_final_alpha_ELU.png')
    # plot_all_runs_alphas(selu_wd, r'$\alpha SELU$', './3_result_plots/', 'MAIN_final_alpha_SELU.png')
    # plot_all_runs_alphas(swish_wd, r'$\alpha Swish$', './3_result_plots/', 'MAIN_final_alpha_Swish.png', beta=swish_swishbeta)

    # plot all alphas for individual AFs (plot @ 6k and @60k)
    af_dict_list = [linu_wd, tanh_wd, relu_wd, elu_wd, selu_wd, swish_wd]
    af_dict_list_2 = [linu_wd_2, tanh_wd_2, relu_wd_2, elu_wd_2, selu_wd_2, swish_wd_2]
    beta_list = [None, None, None, None, None, swish_swishbeta]
    beta_list_2 = [None, None, None, None, None, swish_swishbeta_2]
    title_list = [r'$\alpha I$', r'$\alpha tanh$', r'$\alpha ReLU$', r'$\alpha ELU$', r'$\alpha SELU$', r'$\alpha Swish$']
    plot_all_runs_alphas_multi(af_dict_list, title_list, './3_result_plots/', 'MAIN_final_alpha_all_2.png', af_dict_list_2=af_dict_list_2, beta_list=beta_list, beta_list_2=beta_list_2)
    # plot_all_runs_alphas(linu_wd, r'$\alpha I$', './3_result_plots/', 'MAIN_final_alpha_I_2.png', af_dict_2=linu_wd_2)
    # plot_all_runs_alphas(tanh_wd, r'$\alpha tanh$', './3_result_plots/', 'MAIN_final_alpha_tanh_2.png', af_dict_2=tanh_wd_2) # ylim=[0.0,1.8]
    # plot_all_runs_alphas(relu_wd, r'$\alpha ReLU$', './3_result_plots/', 'MAIN_final_alpha_ReLU_2.png', af_dict_2=relu_wd_2)
    # plot_all_runs_alphas(elu_wd, r'$\alpha ELU$', './3_result_plots/', 'MAIN_final_alpha_ELU_2.png', af_dict_2=elu_wd_2)
    # plot_all_runs_alphas(selu_wd, r'$\alpha SELU$', './3_result_plots/', 'MAIN_final_alpha_SELU_2.png', af_dict_2=selu_wd_2)
    # plot_all_runs_alphas(swish_wd, r'$\alpha Swish$', './3_result_plots/', 'MAIN_final_alpha_Swish_2.png', af_dict_2=swish_wd_2, beta=swish_swishbeta, beta_2=swish_swishbeta_2)

    # plot all alphas for individual AFs (plot @ 1k, 6k and @60k)
    af_dict_list = [linu_wd, tanh_wd, relu_wd, elu_wd, selu_wd, swish_wd]
    af_dict_list_2 = [linu_wd_2, tanh_wd_2, relu_wd_2, elu_wd_2, selu_wd_2, swish_wd_2]
    af_dict_list_3 = [linu_wd_3, tanh_wd_3, relu_wd_3, elu_wd_3, selu_wd_3, swish_wd_3]
    beta_list = [None, None, None, None, None, swish_swishbeta]
    beta_list_2 = [None, None, None, None, None, swish_swishbeta_2]
    beta_list_3 = [None, None, None, None, None, swish_swishbeta_3]
    title_list = [r'$\alpha I$', r'$\alpha tanh$', r'$\alpha ReLU$', r'$\alpha ELU$', r'$\alpha SELU$', r'$\alpha Swish$']
    plot_all_runs_alphas_multi(af_dict_list, title_list, './3_result_plots/', 'MAIN_final_alpha_all_3.png', af_dict_list_2=af_dict_list_2, af_dict_list_3=af_dict_list_3, beta_list=beta_list, beta_list_2=beta_list_2, beta_list_3=beta_list_3)
    # plot_all_runs_alphas(linu_wd, r'$\alpha I$', './3_result_plots/', 'MAIN_final_alpha_I_3.png', af_dict_2=linu_wd_2, af_dict_3=linu_wd_3)
    # plot_all_runs_alphas(tanh_wd, r'$\alpha tanh$', './3_result_plots/', 'MAIN_final_alpha_tanh_3.png', af_dict_2=tanh_wd_2, af_dict_3=tanh_wd_3) # ylim=[0.0,1.8]
    # plot_all_runs_alphas(relu_wd, r'$\alpha ReLU$', './3_result_plots/', 'MAIN_final_alpha_ReLU_3.png', af_dict_2=relu_wd_2, af_dict_3=relu_wd_3)
    # plot_all_runs_alphas(elu_wd, r'$\alpha ELU$', './3_result_plots/', 'MAIN_final_alpha_ELU_3.png', af_dict_2=elu_wd_2, af_dict_3=elu_wd_3)
    # plot_all_runs_alphas(selu_wd, r'$\alpha SELU$', './3_result_plots/', 'MAIN_final_alpha_SELU_3.png', af_dict_2=selu_wd_2, af_dict_3=selu_wd_3)
    # plot_all_runs_alphas(swish_wd, r'$\alpha Swish$', './3_result_plots/', 'MAIN_final_alpha_Swish_3.png', af_dict_2=swish_wd_2, af_dict_3=swish_wd_3, beta=swish_swishbeta, beta_2=swish_swishbeta_2, beta_3=swish_swishbeta_3)

    # plot mean alpha over layers for adaptively scaled functions
    af_list = [linu_wd, tanh_wd, relu_wd, elu_wd, selu_wd, swish_wd, swish_swishbeta]
    name_list = [r'$\alpha I$', r'$\alpha tanh$', r'$\alpha ReLU$', r'$\alpha ELU$', r'$\alpha SELU$', r'$\alpha Swish$', r'$\alpha Swish\ \beta$']
    plot_mean_alpha_by_layers(af_list, name_list, 'mean alphas ASU', './3_result_plots/', scaled_AFs_figname, includes_beta=True)

# ##############################################################################
# ### SCRIPT ASU ###############################################################
# ##############################################################################

if 2 in steps:

    path_finalweights = './3_output_cifar/ASC_main/0_af_weights/'

    ABU_figname = 'MAIN_mean_alphas_ABU.png'
    ABU_S_figname = 'MAIN_mean_alphas_ABU_S.png'
    ABU_N_figname = 'MAIN_mean_alphas_ABU_N.png'
    ABU_A_figname = 'MAIN_mean_alphas_ABU_A.png'
    ABU_P_figname = 'MAIN_mean_alphas_ABU_P.png'

    ABU_AF_figname = 'MAIN_resulting_AFs_ABU.png'
    ABU_S_AF_figname = 'MAIN_resulting_AFs_ABU_S.png'
    ABU_N_AF_figname = 'MAIN_resulting_AFs_ABU_N.png'
    ABU_A_AF_figname = 'MAIN_resulting_AFs_ABU_A.png'
    ABU_P_AF_figname = 'MAIN_resulting_AFs_ABU_P.png'

    ABU_AF_norm_figname = 'MAIN_resulting_AFnormed_ABU.png'

    # get files from folder
    mb_step = 60000
    mb_step_2 = 6103
    mb_step_3 = 1018
    blend5u_fw_files = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('_ABU_TERIS_' in f) and ('_mb_'+str(mb_step) in f))]
    blend5u_fw_files_2 = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('_ABU_TERIS_' in f) and ('_mb_'+str(mb_step_2) in f))]
    blend5u_fw_files_3 = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('_ABU_TERIS_' in f) and ('_mb_'+str(mb_step_3) in f))]
    print('- created list of ABU final weight files at minibatches %i, %i, and %i (%i runs in total)' %(mb_step, mb_step_2, mb_step_3, len(blend5u_fw_files)))
    blend5n_fw_files = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('_ABU_N_TERIS_' in f) and ('_mb_'+str(mb_step) in f))]
    blend5n_fw_files_2 = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('_ABU_N_TERIS_' in f) and ('_mb_'+str(mb_step_2) in f))]
    blend5n_fw_files_3 = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('_ABU_N_TERIS_' in f) and ('_mb_'+str(mb_step_3) in f))]
    print('- created list of ABU_N final weight files at minibatches %i, %i, and %i (%i runs in total)' %(mb_step, mb_step_2, mb_step_3, len(blend5n_fw_files)))
    blend5p_fw_files = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('_ABU_P_TERIS_' in f) and ('_mb_'+str(mb_step) in f))]
    blend5p_fw_files_2 = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('_ABU_P_TERIS_' in f) and ('_mb_'+str(mb_step_2) in f))]
    blend5p_fw_files_3 = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('_ABU_P_TERIS_' in f) and ('_mb_'+str(mb_step_3) in f))]
    print('- created list of ABU_P final weight files at minibatches %i, %i, and %i (%i runs in total)' %(mb_step, mb_step_2, mb_step_3, len(blend5p_fw_files)))
    blend5a_fw_files = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('_ABU_A_TERIS_' in f) and ('_mb_'+str(mb_step) in f))]
    blend5a_fw_files_2 = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('_ABU_A_TERIS_' in f) and ('_mb_'+str(mb_step_2) in f))]
    blend5a_fw_files_3 = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('_ABU_A_TERIS_' in f) and ('_mb_'+str(mb_step_3) in f))]
    print('- created list of ABU_A final weight files at minibatches %i, %i, and %i (%i runs in total)' %(mb_step, mb_step_2, mb_step_3, len(blend5p_fw_files)))
    blend5s_fw_files = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('_ABU_S_TERIS_' in f) and ('_mb_'+str(mb_step) in f))]
    blend5s_fw_files_2 = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('_ABU_S_TERIS_' in f) and ('_mb_'+str(mb_step_2) in f))]
    blend5s_fw_files_3 = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('_ABU_S_TERIS_' in f) and ('_mb_'+str(mb_step_3) in f))]
    print('- created list of ABU_S final weight files at minibatches %i, %i, and %i (%i runs in total)' %(mb_step, mb_step_2, mb_step_3, len(blend5p_fw_files)))


    if len(blend5u_fw_files) > 0:
        blend5u_wd, blend5u_swishbeta = smcn_extract_weights(path_finalweights, blend5u_fw_files)
        blend5u_wd_2, blend5u_swishbeta_2 = smcn_extract_weights(path_finalweights, blend5u_fw_files_2)
        blend5u_wd_3, blend5u_swishbeta_3 = smcn_extract_weights(path_finalweights, blend5u_fw_files_3)
    if len(blend5n_fw_files) > 0:
        blend5n_wd, blend5n_swishbeta = smcn_extract_weights(path_finalweights, blend5n_fw_files)
        blend5n_wd_2, blend5n_swishbeta_2 = smcn_extract_weights(path_finalweights, blend5n_fw_files_2)
        blend5n_wd_3, blend5n_swishbeta_3 = smcn_extract_weights(path_finalweights, blend5n_fw_files_3)
    if len(blend5p_fw_files) > 0:
        blend5p_wd, blend5p_swishbeta = smcn_extract_weights(path_finalweights, blend5p_fw_files)
        blend5p_wd_2, blend5p_swishbeta_2 = smcn_extract_weights(path_finalweights, blend5p_fw_files_2)
        blend5p_wd_3, blend5p_swishbeta_3 = smcn_extract_weights(path_finalweights, blend5p_fw_files_3)
    if len(blend5a_fw_files) > 0:
        blend5a_wd, blend5a_swishbeta = smcn_extract_weights(path_finalweights, blend5a_fw_files)
        blend5a_wd_2, blend5a_swishbeta_2 = smcn_extract_weights(path_finalweights, blend5a_fw_files_2)
        blend5a_wd_3, blend5a_swishbeta_3 = smcn_extract_weights(path_finalweights, blend5a_fw_files_3)
    if len(blend5s_fw_files) > 0:
        blend5s_wd, blend5s_swishbeta = smcn_extract_weights(path_finalweights, blend5s_fw_files)
        blend5s_wd_2, blend5s_swishbeta_2 = smcn_extract_weights(path_finalweights, blend5s_fw_files_2)
        blend5s_wd_3, blend5s_swishbeta_3 = smcn_extract_weights(path_finalweights, blend5s_fw_files_3)

    if len(blend5u_fw_files) > 0:
        smcn_print_mean_std(blend5u_wd, 'ABU-TERIS alpha')
        smcn_print_mean_std(blend5u_swishbeta, 'ABU-TERIS SwB')
    if len(blend5n_fw_files) > 0:
        smcn_print_mean_std(blend5n_wd, 'ABU_N-TERIS alpha')
        smcn_print_mean_std(blend5n_swishbeta, 'ABU_N-TERIS SwB')
    if len(blend5p_fw_files) > 0:
        smcn_print_mean_std(blend5p_wd, 'ABU_P-TERIS alpha')
        smcn_print_mean_std(blend5p_swishbeta, 'ABU_P-TERIS SwB')
    if len(blend5a_fw_files) > 0:
        smcn_print_mean_std(blend5a_wd, 'ABU_P-TERIS alpha')
        smcn_print_mean_std(blend5a_swishbeta, 'ABU_P-TERIS SwB')
    if len(blend5s_fw_files) > 0:
        smcn_print_mean_std(blend5s_wd, 'ABU_P-TERIS alpha')
        smcn_print_mean_std(blend5s_swishbeta, 'ABU_P-TERIS SwB')

    # plot mean alpha over layers for adaptively scaled functions & plot resulting AFs:
    if len(blend5u_fw_files) > 0:
        # ABU
        ABU_AF_names, ABU_afs_by_layers_means_C10, ABU_swish_betas_by_layers_means_C10 = get_mean_alphas_by_layers_ABU(blend5u_wd, blend5u_swishbeta)
        plot_mean_alpha_by_layers_ABU(ABU_AF_names, ABU_afs_by_layers_means_C10, 'ABU', './3_result_plots/', ABU_figname, swish_betas_by_layers_means=ABU_swish_betas_by_layers_means_C10, ylim=[-0.4,0.6])
        plot_ABU(ABU_afs_by_layers_means_C10, ABU_swish_betas_by_layers_means_C10, './3_result_plots/', ABU_AF_figname, ylim=[-1.0, 1.0], col='blue', norm="None")
    if len(blend5n_fw_files) > 0:
        # ABU_N
        ABU_N_AF_names, ABU_N_afs_by_layers_means_C10, ABU_N_swish_betas_by_layers_means_C10 = get_mean_alphas_by_layers_ABU(blend5n_wd, blend5n_swishbeta)
        plot_mean_alpha_by_layers_ABU(ABU_N_AF_names, ABU_N_afs_by_layers_means_C10, r'$ABU_{N}$', './3_result_plots/', ABU_N_figname, swish_betas_by_layers_means=ABU_N_swish_betas_by_layers_means_C10, ylim=[-0.6,2.0])
        plot_ABU(ABU_N_afs_by_layers_means_C10, ABU_N_swish_betas_by_layers_means_C10, './3_result_plots/', ABU_N_AF_figname, ylim=[-2.0, 2.0], col='blue', norm="N")
    if len(blend5p_fw_files) > 0:
        # ABU_P
        ABU_P_AF_names, ABU_P_afs_by_layers_means_C10, ABU_P_swish_betas_by_layers_means_C10 = get_mean_alphas_by_layers_ABU(blend5p_wd, blend5p_swishbeta)
        plot_mean_alpha_by_layers_ABU(ABU_P_AF_names, ABU_P_afs_by_layers_means_C10, r'$ABU_{P}$', './3_result_plots/', ABU_P_figname, swish_betas_by_layers_means=ABU_P_swish_betas_by_layers_means_C10, ylim=[-0.1,1.2])
        plot_ABU(ABU_P_afs_by_layers_means_C10, ABU_P_swish_betas_by_layers_means_C10, './3_result_plots/', ABU_P_AF_figname, ylim=[-2.0, 2.0], col='blue', norm="P")
    if len(blend5a_fw_files) > 0:
        # ABU_A
        ABU_A_AF_names, ABU_A_afs_by_layers_means_C10, ABU_A_swish_betas_by_layers_means_C10 = get_mean_alphas_by_layers_ABU(blend5a_wd, blend5a_swishbeta)
        plot_mean_alpha_by_layers_ABU(ABU_A_AF_names, ABU_A_afs_by_layers_means_C10, r'$ABU_{A}$', './3_result_plots/', ABU_A_figname, swish_betas_by_layers_means=ABU_A_swish_betas_by_layers_means_C10, ylim=[-0.1,1.2])
        plot_ABU(ABU_A_afs_by_layers_means_C10, ABU_A_swish_betas_by_layers_means_C10, './3_result_plots/', ABU_A_AF_figname, ylim=[-1.0, 1.0], col='blue', norm="A")
    if len(blend5s_fw_files) > 0:
        # ABU_S
        ABU_S_AF_names, ABU_S_afs_by_layers_means_C10, ABU_S_swish_betas_by_layers_means_C10 = get_mean_alphas_by_layers_ABU(blend5s_wd, blend5s_swishbeta)
        plot_mean_alpha_by_layers_ABU(ABU_S_AF_names, ABU_S_afs_by_layers_means_C10, r'$ABU_{S}$', './3_result_plots/', ABU_S_figname, swish_betas_by_layers_means=ABU_S_swish_betas_by_layers_means_C10, ylim=[-0.1,1.2])
        plot_ABU(ABU_S_afs_by_layers_means_C10, ABU_S_swish_betas_by_layers_means_C10, './3_result_plots/', ABU_S_AF_figname, ylim=[-2.0, 2.0], col='blue', norm="S")
