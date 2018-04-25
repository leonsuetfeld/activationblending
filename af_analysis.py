import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pickle
import scipy
from numpy import ma
from scipy.stats import norm

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
    lgd = ax.legend(handles, labels, loc='center left',  bbox_to_anchor=(1,0.5), fancybox=False, shadow=True)
    ax.tick_params(axis='x', which='both', bottom='off', top='off')
    # save plot as image
    if not os.path.exists(saveplot_path):
        os.makedirs(saveplot_path)
    fig.savefig(saveplot_path+saveplot_filename, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=150)

def plot_mean_alpha_by_layers_ABU(AF_names, afs_by_layers_means, title, saveplot_path, saveplot_filename, swish_beta_by_layers_means=np.array([0])):
    linewidth_default = '3'
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111)
    ax.plot([0,7],[1./7.,1./7.], '-', color='#000000', linewidth='1', alpha=0.5)
    ax.plot([1,1],[-9,9], ':', color='#000000', linewidth='1', alpha=0.3)
    ax.plot([2,2],[-9,9], ':', color='#000000', linewidth='1', alpha=0.3)
    ax.plot([3,3],[-9,9], ':', color='#000000', linewidth='1', alpha=0.3)
    ax.plot([4,4],[-9,9], ':', color='#000000', linewidth='1', alpha=0.3)
    ax.plot([5,5],[-9,9], ':', color='#000000', linewidth='1', alpha=0.3)
    ax.plot([6,6],[-9,9], ':', color='#000000', linewidth='1', alpha=0.3)
    color_list = ['#66c2a5', '#fc8d62', '#8c9fcb', '#e78ac3', '#a5d853', '#b4b4b4', '#ffd82e', '#e4c494']
    x = np.arange(6)+1
    if swish_beta_by_layers_means.shape[0]>1:
        ax2 = ax.twinx()
        ax2.plot(x, swish_beta_by_layers_means, '--', linewidth=linewidth_default, color=color_list[3], label=r'$Swish\ \beta$', alpha=1.0)
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
    lgd = ax.legend(handles+handles2, labels+labels2, loc='lower left',  bbox_to_anchor=(1.05,0.0), fancybox=False, shadow=True)
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
    swish_beta_by_layers_means = np.zeros((n_layers,))
    if swish_beta:
        swish_beta_by_layers_means = np.zeros((n_layers,))
        swish_beta_by_layers_means[0] = np.mean(swish_beta['conv1'], axis=0)[0]
        swish_beta_by_layers_means[1] = np.mean(swish_beta['conv2'], axis=0)[0]
        swish_beta_by_layers_means[2] = np.mean(swish_beta['conv3'], axis=0)[0]
        swish_beta_by_layers_means[3] = np.mean(swish_beta['conv4'], axis=0)[0]
        swish_beta_by_layers_means[4] = np.mean(swish_beta['dense5'], axis=0)[0]
        swish_beta_by_layers_means[5] = np.mean(swish_beta['dense6'], axis=0)[0]
    return AF_names, afs_by_layers_means, swish_beta_by_layers_means

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
        plt.plot([-6.,6.],[0,0], '--', linewidth=1, color='black')
        plt.plot([0,0],[-100,100], '--', linewidth=1, color='black')
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
        plt.plot([-6.,6.],[0,0], '--', linewidth=1, color='black')
        plt.plot([0,0],[-100,100], '--', linewidth=1, color='black')
        plt.plot(x,resulting_AF_layer_i_C100, linewidth='3', color='blue', alpha=0.5, label='CIFAR100')
        plt.plot(x,resulting_AF_layer_i_C10, linewidth='3', color='orange', alpha=1.0, label='CIFAR10')
        plt.xlim(show_range[0], show_range[1])
        plt.ylim(-1.5, 2.5)
        plt.title(layer_names[layer_i])
        if layer_i > 0:
            plt.tick_params(axis='y', which='both', labelleft='off') # labels along the left and right off
        if layer_i == 5:
            handles, labels = ax.get_legend_handles_labels()
            lgd = ax.legend(handles, labels, loc='upper left',  bbox_to_anchor=(1,1), fancybox=False, shadow=True)
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
ABU_AF_figname = 'mean_ABU_cifar10.png'
ABU_AF_N_figname = 'mean_ABU_cifar10_normalized.png'
print('\n ### CIFAR 10 ###############################################')

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
smcn_print_mean_std(blend5_wd, 'ABU-TERIS alpha')
smcn_print_mean_std(blend5_swishbeta, 'ABU-TERIS SwB')

# plot mean alpha over layers for adaptively scaled functions
af_list = [linu_wd, tanh_wd, relu_wd, elu_wd, selu_wd, swish_wd, swish_swishbeta]
name_list = [r'$\alpha I$', r'$\alpha tanh$', r'$\alpha ReLU$', r'$\alpha ELU$', r'$\alpha SELU$', r'$\alpha Swish$', r'$\alpha Swish\ \beta$']
plot_mean_alpha_by_layers(af_list, name_list, 'CIFAR10', './2_result_plots/', scaled_AFs_figname, includes_beta=True)

# plot mean alpha over layers for adaptively scaled functions
AF_names, afs_by_layers_means_C10, swish_beta_by_layers_means_C10 = get_mean_alphas_by_layers_ABU(blend5_wd, blend5_swishbeta)
plot_mean_alpha_by_layers_ABU(AF_names, afs_by_layers_means_C10, 'CIFAR10', './2_result_plots/', ABU_figname, swish_beta_by_layers_means=swish_beta_by_layers_means_C10)

# plot resulting trained ABU activation function
# plot_ABU(afs_by_layers_means_C10, swish_beta_by_layers_means_C10, './2_result_plots/', ABU_AF_figname, normalize_weights=False)
# plot_ABU(afs_by_layers_means_C10, swish_beta_by_layers_means_C10, './2_result_plots/', ABU_AF_N_figname, normalize_weights=True)

# ##############################################################################
# ### SCRIPT CIFAR 100 #########################################################
# ##############################################################################

path_finalweights = './2_output_cifar/SBF_6b/_af_weights/'
scaled_AFs_figname = 'mean_alpha_over_layers_scaled_afs_cifar100.png'
ABU_figname = 'mean_alpha_over_layers_ABU_cifar100.png'
ABU_AF_figname = 'mean_ABU_cifar100.png'
ABU_AF_N_figname = 'mean_ABU_cifar100_normalized.png'
print('\n ### CIFAR 100 ###############################################')

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
smcn_print_mean_std(blend5_wd, 'ABU-TERIS alpha')
smcn_print_mean_std(blend5_swishbeta, 'ABU-TERIS SwB')

# plot mean alpha over layers for adaptively scaled functions
af_list = [linu_wd, tanh_wd, relu_wd, elu_wd, selu_wd, swish_wd, swish_swishbeta]
name_list = [r'$\alpha I$', r'$\alpha tanh$', r'$\alpha ReLU$', r'$\alpha ELU$', r'$\alpha SELU$', r'$\alpha Swish$', r'$\alpha Swish\ \beta$']
plot_mean_alpha_by_layers(af_list, name_list, 'CIFAR100', './2_result_plots/', scaled_AFs_figname, includes_beta=True)

# plot mean alpha over layers for adaptively scaled functions
AF_names, afs_by_layers_means_C100, swish_beta_by_layers_means_C100 = get_mean_alphas_by_layers_ABU(blend5_wd, blend5_swishbeta)
plot_mean_alpha_by_layers_ABU(AF_names, afs_by_layers_means_C100, 'CIFAR100', './2_result_plots/', ABU_figname, swish_beta_by_layers_means=swish_beta_by_layers_means_C100)

# plot resulting trained ABU activation function
# plot_ABU(afs_by_layers_means_C100, swish_beta_by_layers_means_C100, './2_result_plots/', ABU_AF_figname, normalize_weights=False)
# plot_ABU(afs_by_layers_means_C100, swish_beta_by_layers_means_C100, './2_result_plots/', ABU_AF_N_figname, normalize_weights=True)

# ##############################################################################
# ### SCRIPT COMBINED ##########################################################
# ##############################################################################

# plot resulting trained ABU activation function
plot_ABU2(afs_by_layers_means_C10, swish_beta_by_layers_means_C10, afs_by_layers_means_C100, swish_beta_by_layers_means_C100, './2_result_plots/', 'mean_ABU_both.png', normalize_weights=False)
plot_ABU2(afs_by_layers_means_C10, swish_beta_by_layers_means_C10, afs_by_layers_means_C100, swish_beta_by_layers_means_C100, './2_result_plots/', 'mean_ABU_both_normalized.png', normalize_weights=True)
