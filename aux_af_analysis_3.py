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

def smcn_extract_wstats(path_all_weights, keyword):

    # get list of files for spec
    files_of_spec = [f for f in os.listdir(path_all_weights) if (os.path.isfile(os.path.join(path_all_weights, f)) and ('.pkl' in f) and (keyword in f) and ('mb_1' in f or 'mb_60000' in f))]
    files_of_spec.sort()

    # get number of runs and timeteps
    complete_runs = []
    for filename in files_of_spec:
        run = int(filename.split('_mb')[0].split('run_')[-1])
        timestep = int(filename.split('.')[0].split('mb_')[-1])
        if timestep == 60000 and run not in complete_runs:
            complete_runs.append(run)

    # get relevant stats from files
    w_stats_t1, w_stats_t60000 = [], []
    for filename in files_of_spec:
        w_dict = pickle.load( open( path_all_weights+filename, "rb" ) )
        timestep = int(filename.split('.')[0].split('mb_')[-1])
        run = int(filename.split('_mb')[0].split('run_')[-1])
        if run in complete_runs and timestep in [1, 60000]:
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
            elif timestep == 60000:
                w_stats_t60000.append(w_stats_in_file)
            he_stds = [c1_he_std, c2_he_std, c3_he_std, c4_he_std, d5_he_std, d6_he_std, do_he_std]
    # convert to np array
    w_stats_t1 = np.array(w_stats_t1)
    w_stats_t60000 = np.array(w_stats_t60000)
    he_stds = np.array(he_stds)
    # return mean over runs
    return np.mean(w_stats_t1, axis=0), np.mean(w_stats_t60000, axis=0), he_stds

def smcn_extract_alphastats(path_all_weights, keyword, layer_list):
    # get list of files for spec
    files_of_spec = [f for f in os.listdir(path_all_weights) if (os.path.isfile(os.path.join(path_all_weights, f)) and ('.pkl' in f) and (keyword in f) and ('mb_1' in f or 'mb_60000' in f))]
    files_of_spec.sort()

    # get number of runs and timeteps
    complete_runs = []
    for filename in files_of_spec:
        run = int(filename.split('_mb')[0].split('run_')[-1])
        timestep = int(filename.split('.')[0].split('mb_')[-1])
        if timestep == 60000 and run not in complete_runs:
            complete_runs.append(run)

    # get relevant stats from files
    bws_mb_1, bws_mb_60000 = [], []
    for filename in files_of_spec:
        w_dict = pickle.load( open( path_all_weights+filename, "rb" ) )
        run = int(filename.split('_mb')[0].split('run_')[-1])
        timestep = int(filename.split('.')[0].split('mb_')[-1])
        if run in complete_runs and timestep in [1, 60000]:
            if 'smcn/conv1/blend_weights:0' in w_dict:
                bw_c1 = w_dict['smcn/conv1/blend_weights:0']
            else:
                bw_c1 = np.ones((1,))
            if 'smcn/conv1/blend_weights:0' in w_dict:
                bw_c2 = w_dict['smcn/conv2/blend_weights:0']
            else:
                bw_c2 = np.ones((1,))
            if 'smcn/conv1/blend_weights:0' in w_dict:
                bw_c3 = w_dict['smcn/conv3/blend_weights:0']
            else:
                bw_c3 = np.ones((1,))
            if 'smcn/conv1/blend_weights:0' in w_dict:
                bw_c4 = w_dict['smcn/conv4/blend_weights:0']
            else:
                bw_c4 = np.ones((1,))
            if 'smcn/conv1/blend_weights:0' in w_dict:
                bw_d5 = w_dict['smcn/dense5/blend_weights:0']
            else:
                bw_d5 = np.ones((1,))
            if 'smcn/conv1/blend_weights:0' in w_dict:
                bw_d6 = w_dict['smcn/dense6/blend_weights:0']
            else:
                bw_d6 = np.ones((1,))
            if timestep == 1:
                bws_mb_1.append(np.array([bw_c1, bw_c2, bw_c3, bw_c4, bw_d5, bw_d6]))
            elif timestep == 60000:
                bws_mb_60000.append(np.array([bw_c1, bw_c2, bw_c3, bw_c4, bw_d5, bw_d6]))
    bws_mb_1 = np.array(bws_mb_1)
    bws_mb_60000 = np.array(bws_mb_60000)

    mean_bws_per_layer_1 = np.mean(bws_mb_1, axis=0)
    std_bws_per_layer_1 = np.std(bws_mb_1, axis=0)
    mean_bws_per_layer_60000 = np.mean(bws_mb_60000, axis=0)
    std_bws_per_layer_60000 = np.std(bws_mb_60000, axis=0)

    return mean_bws_per_layer_1, std_bws_per_layer_1, mean_bws_per_layer_60000, std_bws_per_layer_60000

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

def print_weight_stats(path_all_weights, keyword, layer_list):
    wstats_1, wstats_60000, he_stds = smcn_extract_wstats(path_all_weights, keyword)
    stds_by_layer_1 = wstats_1[:,-1]
    stds_by_layer_60000 = wstats_60000[:,-1]
    mean_bws_per_layer_1, std_bws_per_layer_1, mean_bws_per_layer_60000, std_bws_per_layer_60000 = smcn_extract_alphastats(path_all_weights, keyword, layer_list)
    print(keyword)
    print('-----------------------------------------------------------------------------------------------------------------------------')
    print('stds (he) layer2+:         ', he_stds[1:])
    print('std (start) layer2+:       ', stds_by_layer_1[1:])
    print('alphas (start):            ', np.sum(mean_bws_per_layer_1, axis=1))
    print('alphas * stds (start):     ', stds_by_layer_1[1:]*np.mean(mean_bws_per_layer_1, axis=1))
    print('-----------------------------------------------------------------------------------------------------------------------------')
    print('std (end) layer2+:         ', stds_by_layer_60000[1:])
    print('alphas (end):              ', np.sum(mean_bws_per_layer_60000, axis=1))
    print('alphas * stds (end):       ', stds_by_layer_60000[1:]*np.mean(mean_bws_per_layer_60000, axis=1), '<= final corrected weight std')
    print('-----------------------------------------------------------------------------------------------------------------------------')
    print('')

# ##############################################################################
# ### SCRIPT ###################################################################
# ##############################################################################

sections = [1]

if 1 in sections:

    path_af_weights = './3_output_cifar/ASC_main/0_af_weights/'
    path_all_weights = './3_output_cifar/ASC_main/0_all_weights/'
    layer_list = ['c1', 'c2', 'c3', 'c4', 'd1', 'd2']
    abu_af_list = ['ReLU', 'ELU', 'tanh', 'Swish', 'I']

    print('')

    print_weight_stats(path_all_weights, '_c10_tanh', layer_list)
    print_weight_stats(path_all_weights, '_c10_alpha_tanh', layer_list)

    print_weight_stats(path_all_weights, '_c10_relu', layer_list)
    print_weight_stats(path_all_weights, '_c10_alpha_relu', layer_list)

    print_weight_stats(path_all_weights, '_c10_elu', layer_list)
    print_weight_stats(path_all_weights, '_c10_alpha_elu', layer_list)

    print_weight_stats(path_all_weights, '_c10_selu', layer_list)
    print_weight_stats(path_all_weights, '_c10_alpha_selu', layer_list)

    print_weight_stats(path_all_weights, '_c10_swish', layer_list)
    print_weight_stats(path_all_weights, '_c10_alpha_swish', layer_list)
