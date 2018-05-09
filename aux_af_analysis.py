import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pickle
import scipy
from numpy import ma
from scipy.stats import norm

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
    linewidth_default = '3'
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
        ax.plot(x, afs_by_layers_means[af,:], linewidth=linewidth_default, color=color_list[af], label=name_list[af], alpha=1.0)
    if includes_beta:
        ax.plot(x, afs_by_layers_means[-1,:], '--', linewidth=linewidth_default, color=color_list[af], label=name_list[n_afs], alpha=1.0)
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

def plot_all_runs_alphas(af_dict, title, saveplot_path, saveplot_filename, ylim=[0.0,1.4], beta={}):
    n_runs = len(af_dict['conv1'])
    # extract means from afs
    alphas_by_layers = np.zeros((n_runs,6))
    for run in range(n_runs):
        alphas_by_layers[run,0] = af_dict['conv1'][run]
        alphas_by_layers[run,1] = af_dict['conv2'][run]
        alphas_by_layers[run,2] = af_dict['conv3'][run]
        alphas_by_layers[run,3] = af_dict['conv4'][run]
        alphas_by_layers[run,4] = af_dict['dense5'][run]
        alphas_by_layers[run,5] = af_dict['dense6'][run]
    if beta:
        betas_by_layers = np.zeros((n_runs,6))
        for run in range(n_runs):
            betas_by_layers[run,0] = beta['conv1'][run]
            betas_by_layers[run,1] = beta['conv2'][run]
            betas_by_layers[run,2] = beta['conv3'][run]
            betas_by_layers[run,3] = beta['conv4'][run]
            betas_by_layers[run,4] = beta['dense5'][run]
            betas_by_layers[run,5] = beta['dense6'][run]
    # figure
    linewidth_default = '1'
    fig = plt.figure(figsize=(6,6))
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
            ax.plot(x, betas_by_layers[run,:], linewidth=linewidth_default, color='orange', alpha=0.1)
    for run in range(n_runs):
        ax.plot(x, alphas_by_layers[run,:], linewidth=linewidth_default, color='blue', alpha=0.1)
    ax.set_ylim(ylim)
    ax.set_xlim([0.8,6.2])
    ax.set_xticklabels(['','conv1','conv2','conv3','conv4','dense1','dense2'])
    ax.set_ylabel('mean '+r'$\alpha_i$')
    ax.set_title(title)
    ax.tick_params(axis='x', which='both', bottom='off', top='off')
    # save plot as image
    if not os.path.exists(saveplot_path):
        os.makedirs(saveplot_path)
    fig.savefig(saveplot_path+saveplot_filename, bbox_inches='tight', dpi=150)

def plot_mean_alpha_by_layers_ABU(AF_names, afs_by_layers_means, title, saveplot_path, saveplot_filename, swish_betas_by_layers_means=np.array([0])):
    linewidth_default = '3'
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111)
    ax.plot([0,7],[1./5.,1./5.], '-', color='#000000', linewidth='1', alpha=0.5)
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
        ax2.plot(x, swish_betas_by_layers_means, '--', linewidth=linewidth_default, color=color_list[3], label=r'$Swish\ \beta$', alpha=1.0)
        ax2.set_ylabel(r'$Swish\ \beta$')
        ax2.set_ylim([0.1,1.4])
        ax2.set_xlim([0.8,6.2])
    for af in range(len(AF_names)):
        ax.plot(x, afs_by_layers_means[:,af], linewidth=linewidth_default, color=color_list[af], label=AF_names[af], alpha=1.0)
    ax.set_ylim([-0.3,0.9])
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

def plot_ABU(mean_alphas, mean_swish_beta, saveplot_path, saveplot_filename, normalize_weights=False):
    mean_alphas = np.copy(mean_alphas)
    mean_swish_beta = np.copy(mean_swish_beta)
    assert mean_alphas.shape[0] == 6 and mean_alphas.shape[1] == 5, 'got wrong mean_alphas matrix. expected [6 layers x 5 AFs].'
    fig = plt.figure(figsize=(18,3.3))
    show_range = [-4.,4.,.01]
    layer_names=['conv1','conv2','conv3','conv4','dense1','dense2']
    for layer_i in range(6):
        # calculate AF
        vec_alpha_layer_i = mean_alphas[layer_i,:]
        if normalize_weights:
            vec_alpha_layer_i /= np.sum(vec_alpha_layer_i)
        x = np.arange(show_range[0], show_range[1], show_range[2])
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
        plt.plot(x,resulting_AF_layer_i, linewidth='3', color='black')
        plt.xlim(show_range[0], show_range[1])
        plt.ylim(-1.5, 2.5)
        plt.title(layer_names[layer_i])
        if layer_i > 0:
            plt.tick_params(axis='y', which='both', labelleft='off') # labels along the left and right off
    plt.tight_layout()
    # save plot as image
    if not os.path.exists(saveplot_path):
        os.makedirs(saveplot_path)
    fig.savefig(saveplot_path+saveplot_filename, dpi=150)

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

# ##############################################################################
# ### SCRIPT CIFAR 10 ##########################################################
# ##############################################################################

path_finalweights = './2_output_cifar/SBF_6a/_af_weights/'
scaled_AFs_figname = 'mean_alpha_over_layers_scaled_afs_cifar10.png'
ABU_figname = 'mean_alpha_over_layers_ABU_cifar10.png'
ABU_N_figname = 'mean_alpha_over_layers_ABU_N_cifar10.png'
ABU_P_figname = 'mean_alpha_over_layers_ABU_P_cifar10.png'
ABU_AF_figname = 'mean_ABU_cifar10.png'
ABU_AF_N_figname = 'mean_ABU_cifar10_normalized.png'
ABU_N_AF_figname = 'mean_ABU_N_cifar10.png'
ABU_P_AF_figname = 'mean_ABU_P_cifar10.png'
print('\n ### CIFAR 10 ###############################################')

# get files from folder
linu_fw_files = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('adaptive_linu_' in f))]
tanh_fw_files = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('adaptive_tanh_' in f))]
relu_fw_files = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('adaptive_relu_' in f))]
elu_fw_files = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('adaptive_elu_' in f))]
selu_fw_files = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('adaptive_selu_' in f))]
swish_fw_files = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('adaptive_swish_' in f))]
blend5u_fw_files = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('blend5_unrest' in f))]
blend5n_fw_files = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('blend5_normed' in f))]
blend5p_fw_files = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('blend5_posnormed' in f))]

# extract weights from files
linu_wd, _ = smcn_extract_weights(path_finalweights, linu_fw_files)
tanh_wd, _ = smcn_extract_weights(path_finalweights, tanh_fw_files)
relu_wd, _ = smcn_extract_weights(path_finalweights, relu_fw_files)
elu_wd, _ = smcn_extract_weights(path_finalweights, elu_fw_files)
selu_wd, _ = smcn_extract_weights(path_finalweights, selu_fw_files)
swish_wd, swish_swishbeta = smcn_extract_weights(path_finalweights, swish_fw_files)
blend5u_wd, blend5u_swishbeta = smcn_extract_weights(path_finalweights, blend5u_fw_files)
blend5n_wd, blend5n_swishbeta = smcn_extract_weights(path_finalweights, blend5n_fw_files)
blend5p_wd, blend5p_swishbeta = smcn_extract_weights(path_finalweights, blend5p_fw_files)

# print mean and std for all weights
smcn_print_mean_std(linu_wd, 'alpha-I alpha')
smcn_print_mean_std(tanh_wd, 'alpha-tanh alpha')
smcn_print_mean_std(relu_wd, 'alpha-ReLU alpha')
smcn_print_mean_std(elu_wd, 'alpha-ELU alpha')
smcn_print_mean_std(selu_wd, 'alpha-SELU alpha')
smcn_print_mean_std(swish_wd, 'alpha-Swish alpha')

smcn_print_mean_std(swish_swishbeta, 'alpha-Swish beta')
smcn_print_mean_std(blend5u_wd, 'ABU-TERIS alpha')
smcn_print_mean_std(blend5u_swishbeta, 'ABU-TERIS SwB')
smcn_print_mean_std(blend5n_wd, 'ABU_N-TERIS alpha')
smcn_print_mean_std(blend5n_swishbeta, 'ABU_N-TERIS SwB')
smcn_print_mean_std(blend5p_wd, 'ABU_P-TERIS alpha')
smcn_print_mean_std(blend5p_swishbeta, 'ABU_P-TERIS SwB')

# plot all alphas for individual AFs
plot_all_runs_alphas(linu_wd, r'$\alpha I$', './2_result_plots/', 'final_scaling_weights_I.png')
plot_all_runs_alphas(tanh_wd, r'$\alpha tanh$', './2_result_plots/', 'final_scaling_weights_tanh.png', ylim=[0.0,1.8])
plot_all_runs_alphas(relu_wd, r'$\alpha ReLU$', './2_result_plots/', 'final_scaling_weights_ReLU.png')
plot_all_runs_alphas(elu_wd, r'$\alpha ELU$', './2_result_plots/', 'final_scaling_weights_ELU.png')
plot_all_runs_alphas(selu_wd, r'$\alpha SELU$', './2_result_plots/', 'final_scaling_weights_SELU.png')
plot_all_runs_alphas(swish_wd, r'$\alpha Swish$', './2_result_plots/', 'final_scaling_weights_Swish.png', beta=swish_swishbeta)

# plot mean alpha over layers for adaptively scaled functions
af_list = [linu_wd, tanh_wd, relu_wd, elu_wd, selu_wd, swish_wd, swish_swishbeta]
name_list = [r'$\alpha I$', r'$\alpha tanh$', r'$\alpha ReLU$', r'$\alpha ELU$', r'$\alpha SELU$', r'$\alpha Swish$', r'$\alpha Swish\ \beta$']
plot_mean_alpha_by_layers(af_list, name_list, 'CIFAR10', './2_result_plots/', scaled_AFs_figname, includes_beta=True)

# plot mean alpha over layers for adaptively scaled functions & plot resulting AFs: ABU
ABU_AF_names, ABU_afs_by_layers_means_C10, ABU_swish_betas_by_layers_means_C10 = get_mean_alphas_by_layers_ABU(blend5u_wd, blend5u_swishbeta)
plot_mean_alpha_by_layers_ABU(ABU_AF_names, ABU_afs_by_layers_means_C10, 'CIFAR10', './2_result_plots/', ABU_figname, swish_betas_by_layers_means=ABU_swish_betas_by_layers_means_C10)
plot_ABU(ABU_afs_by_layers_means_C10, ABU_swish_betas_by_layers_means_C10, './2_result_plots/', ABU_AF_figname, normalize_weights=False)
plot_ABU(ABU_afs_by_layers_means_C10, ABU_swish_betas_by_layers_means_C10, './2_result_plots/', ABU_AF_N_figname, normalize_weights=True)
# ABU_N
ABU_N_AF_names, ABU_N_afs_by_layers_means_C10, ABU_N_swish_betas_by_layers_means_C10 = get_mean_alphas_by_layers_ABU(blend5n_wd, blend5u_swishbeta)
plot_mean_alpha_by_layers_ABU(ABU_N_AF_names, ABU_N_afs_by_layers_means_C10, 'CIFAR10', './2_result_plots/', ABU_N_figname, swish_betas_by_layers_means=ABU_N_swish_betas_by_layers_means_C10)
plot_ABU(ABU_N_afs_by_layers_means_C10, ABU_N_swish_betas_by_layers_means_C10, './2_result_plots/', ABU_N_AF_figname, normalize_weights=False)
# ABU_P
ABU_P_AF_names, ABU_P_afs_by_layers_means_C10, ABU_P_swish_betas_by_layers_means_C10 = get_mean_alphas_by_layers_ABU(blend5p_wd, blend5u_swishbeta)
plot_mean_alpha_by_layers_ABU(ABU_P_AF_names, ABU_P_afs_by_layers_means_C10, 'CIFAR10', './2_result_plots/', ABU_P_figname, swish_betas_by_layers_means=ABU_P_swish_betas_by_layers_means_C10)
plot_ABU(ABU_P_afs_by_layers_means_C10, ABU_P_swish_betas_by_layers_means_C10, './2_result_plots/', ABU_P_AF_figname, normalize_weights=False)

# ##############################################################################
# ### SCRIPT CIFAR 100 #########################################################
# ##############################################################################

path_finalweights = './2_output_cifar/SBF_6b/_af_weights/'
scaled_AFs_figname = 'mean_alpha_over_layers_scaled_afs_cifar100.png'
ABU_figname = 'mean_alpha_over_layers_ABU_cifar100.png'
ABU_N_figname = 'mean_alpha_over_layers_ABU_N_cifar100.png'
ABU_P_figname = 'mean_alpha_over_layers_ABU_P_cifar100.png'
ABU_AF_figname = 'mean_ABU_cifar100.png'
ABU_AF_N_figname = 'mean_ABU_cifar100_normalized.png'
ABU_N_AF_figname = 'mean_ABU_N_cifar100.png'
ABU_P_AF_figname = 'mean_ABU_P_cifar100.png'
print('\n ### CIFAR 100 ###############################################')

# get files from folder
linu_fw_files = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('adaptive_linu_' in f))]
tanh_fw_files = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('adaptive_tanh_' in f))]
relu_fw_files = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('adaptive_relu_' in f))]
elu_fw_files = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('adaptive_elu_' in f))]
selu_fw_files = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('adaptive_selu_' in f))]
swish_fw_files = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('adaptive_swish_' in f))]
blend5u_fw_files = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('blend5_unrest' in f))]
blend5n_fw_files = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('blend5_normed' in f))]
blend5p_fw_files = [f for f in os.listdir(path_finalweights) if (os.path.isfile(os.path.join(path_finalweights, f)) and ('.pkl' in f) and ('blend5_posnormed' in f))]

# extract weights from files
linu_wd, _ = smcn_extract_weights(path_finalweights, linu_fw_files)
tanh_wd, _ = smcn_extract_weights(path_finalweights, tanh_fw_files)
relu_wd, _ = smcn_extract_weights(path_finalweights, relu_fw_files)
elu_wd, _ = smcn_extract_weights(path_finalweights, elu_fw_files)
selu_wd, _ = smcn_extract_weights(path_finalweights, selu_fw_files)
swish_wd, swish_swishbeta = smcn_extract_weights(path_finalweights, swish_fw_files)
blend5u_wd, blend5u_swishbeta = smcn_extract_weights(path_finalweights, blend5u_fw_files)
blend5n_wd, blend5n_swishbeta = smcn_extract_weights(path_finalweights, blend5n_fw_files)
blend5p_wd, blend5p_swishbeta = smcn_extract_weights(path_finalweights, blend5p_fw_files)

# print mean and std for all weights
smcn_print_mean_std(linu_wd, 'alpha-I alpha')
smcn_print_mean_std(tanh_wd, 'alpha-tanh alpha')
smcn_print_mean_std(relu_wd, 'alpha-ReLU alpha')
smcn_print_mean_std(elu_wd, 'alpha-ELU alpha')
smcn_print_mean_std(selu_wd, 'alpha-SELU alpha')
smcn_print_mean_std(swish_wd, 'alpha-Swish alpha')
smcn_print_mean_std(swish_swishbeta, 'alpha-Swish beta')
smcn_print_mean_std(blend5u_wd, 'ABU-TERIS alpha')
smcn_print_mean_std(blend5u_swishbeta, 'ABU-TERIS SwB')
smcn_print_mean_std(blend5n_wd, 'ABU_N-TERIS alpha')
smcn_print_mean_std(blend5n_swishbeta, 'ABU_N-TERIS SwB')
smcn_print_mean_std(blend5p_wd, 'ABU_P-TERIS alpha')
smcn_print_mean_std(blend5p_swishbeta, 'ABU_P-TERIS SwB')

# plot mean alpha over layers for adaptively scaled functions
af_list = [linu_wd, tanh_wd, relu_wd, elu_wd, selu_wd, swish_wd, swish_swishbeta]
name_list = [r'$\alpha I$', r'$\alpha tanh$', r'$\alpha ReLU$', r'$\alpha ELU$', r'$\alpha SELU$', r'$\alpha Swish$', r'$\alpha Swish\ \beta$']
plot_mean_alpha_by_layers(af_list, name_list, 'CIFAR100', './2_result_plots/', scaled_AFs_figname, includes_beta=True)

# plot mean alpha over layers for adaptively scaled functions & plot resulting AFs: ABU
ABU_AF_names, ABU_afs_by_layers_means_C100, ABU_swish_betas_by_layers_means_C100 = get_mean_alphas_by_layers_ABU(blend5u_wd, blend5u_swishbeta)
plot_mean_alpha_by_layers_ABU(ABU_AF_names, ABU_afs_by_layers_means_C100, 'CIFAR100', './2_result_plots/', ABU_figname, swish_betas_by_layers_means=ABU_swish_betas_by_layers_means_C100)
plot_ABU(ABU_afs_by_layers_means_C100, ABU_swish_betas_by_layers_means_C100, './2_result_plots/', ABU_AF_figname, normalize_weights=False)
plot_ABU(ABU_afs_by_layers_means_C100, ABU_swish_betas_by_layers_means_C100, './2_result_plots/', ABU_AF_N_figname, normalize_weights=True)
# ABU_N
ABU_N_AF_names, ABU_N_afs_by_layers_means_C100, ABU_N_swish_betas_by_layers_means_C100 = get_mean_alphas_by_layers_ABU(blend5n_wd, blend5u_swishbeta)
plot_mean_alpha_by_layers_ABU(ABU_N_AF_names, ABU_N_afs_by_layers_means_C100, 'CIFAR100', './2_result_plots/', ABU_N_figname, swish_betas_by_layers_means=ABU_N_swish_betas_by_layers_means_C100)
plot_ABU(ABU_N_afs_by_layers_means_C100, ABU_N_swish_betas_by_layers_means_C100, './2_result_plots/', ABU_N_AF_figname, normalize_weights=False)
# ABU_P
ABU_P_AF_names, ABU_P_afs_by_layers_means_C100, ABU_P_swish_betas_by_layers_means_C100 = get_mean_alphas_by_layers_ABU(blend5p_wd, blend5u_swishbeta)
plot_mean_alpha_by_layers_ABU(ABU_P_AF_names, ABU_P_afs_by_layers_means_C100, 'CIFAR100', './2_result_plots/', ABU_P_figname, swish_betas_by_layers_means=ABU_P_swish_betas_by_layers_means_C100)
plot_ABU(ABU_P_afs_by_layers_means_C100, ABU_P_swish_betas_by_layers_means_C100, './2_result_plots/', ABU_P_AF_figname, normalize_weights=False)

# ##############################################################################
# ### SCRIPT COMBINED ##########################################################
# ##############################################################################

# plot resulting trained ABU activation function
plot_ABU2(ABU_afs_by_layers_means_C10, ABU_swish_betas_by_layers_means_C10, ABU_afs_by_layers_means_C100, ABU_swish_betas_by_layers_means_C100, './2_result_plots/', 'mean_ABU_both.png', normalize_weights=False)
plot_ABU2(ABU_afs_by_layers_means_C10, ABU_swish_betas_by_layers_means_C10, ABU_afs_by_layers_means_C100, ABU_swish_betas_by_layers_means_C100, './2_result_plots/', 'mean_ABU_both_normalized.png', normalize_weights=True)
# plot resulting trained ABU_N activation function
plot_ABU2(ABU_N_afs_by_layers_means_C10, ABU_N_swish_betas_by_layers_means_C10, ABU_N_afs_by_layers_means_C100, ABU_N_swish_betas_by_layers_means_C100, './2_result_plots/', 'mean_ABU_N_both.png', normalize_weights=False)
# plot resulting trained ABU_P activation function
plot_ABU2(ABU_P_afs_by_layers_means_C10, ABU_P_swish_betas_by_layers_means_C10, ABU_P_afs_by_layers_means_C100, ABU_P_swish_betas_by_layers_means_C100, './2_result_plots/', 'mean_ABU_P_both.png', normalize_weights=False)
