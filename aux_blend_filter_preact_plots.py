import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import os

plt.rcParams["font.family"] = ["FreeSerif"]

# ##############################################################################
# ### FUNCTIONS ################################################################
# ##############################################################################

def list_files_in_path(path, contains='', startswith='', add_to_path=True):
    files = [f for f in os.listdir(path) if os.path.isdir(path)]
    if len(contains) > 0:
        files = [f for f in files if contains in f]
    if len(startswith) > 0:
        files = [f for f in files if f.startswith(startswith)]
    if add_to_path:
        for i in range(len(files)):
            files[i] = path+files[i]
    return files

def list_subfolders(path, add_to_path=True):
    subfolders = [f for f in os.listdir(path) if os.path.isdir(path+f)]
    for i in range(len(subfolders)):
        subfolders[i] += '/'
        if add_to_path:
            subfolders[i] = path+subfolders[i]
    return subfolders

def add_cwd_to_list(list_of_paths):
    for i in range(len(list_of_paths)):
        list_of_paths[i] = os.getcwd()+'/'+list_of_paths[i]
    return list_of_paths

def add_cwd_to_path(path):
    return os.getcwd()+'/'+path

def get_single_spec_run_layer_column(df, spec, run, layer, column):
    tmp = df[df.spec == spec]
    tmp = tmp[tmp.run == run]
    tmp = tmp[tmp.layer == layer]
    return tmp[column].values

def plot_blend_filter_preact(data, savepath, filename, figsize=(11.695,2.4), linewidth=2):
    color_list = ['#8c9fcb','#fc8d62', '#e78ac3', '#a5d853', '#66c2a5', '#b4b4b4', '#ffd82e', '#e4c494']
    df_alpha = data[0]
    df_fixed = data[1]
    fig = plt.figure(figsize=figsize)

    ylim1 = [0.0,20.0]
    ylim2 = [0.0,0.25]
    ylim3 = [0.0,1.20]

    # plot 1
    ax3 = fig.add_subplot(1, 2, 2)
    ax2 = ax3.twinx()
    ax1 = ax3.twinx()

    l1 = ax1.plot(df_alpha['step'], df_alpha['d6_preact_std'], '-', color=color_list[1], linewidth=linewidth, label=r'$\sigma^{\mathrm{dense2}}_{\mathrm{pre-activations}}$')
    ax1.grid(linestyle=':', color=color_list[1], axis='y')
    ax1.tick_params(axis='y', colors=color_list[1], labelright=False)
    # ax1.get_yaxis().set_visible(False)
    ax1.set_ylim(ylim1)

    l2 = ax2.plot(df_alpha['step'], df_alpha['d6_weights_std'], '-', color=color_list[0], linewidth=linewidth, label=r'$\sigma^{\mathrm{dense2}}_{\mathrm{weights}}$')
    ax2.tick_params(axis='y', colors=color_list[0])
    # ax2.get_yaxis().set_visible(False)
    ax2.set_ylim(ylim2)

    l3 = ax3.plot(df_alpha['step'], df_alpha['d5_alpha'], '-', color=color_list[2], linewidth=linewidth, label=r'$\alpha_{\mathrm{dense1}}$')
    ax3.tick_params(axis='y', colors=color_list[2])
    ax3.set_ylim(ylim3)

    plt.locator_params(axis='x', nbins=3)
    plt.locator_params(axis='y', nbins=5)
    ax1.set_xticklabels(['0','20k','40k','60k'])
    ax1.set_title(r'$\alpha$tanh')

    # plot 2
    bx1 = fig.add_subplot(1, 2, 1)
    bx2 = bx1.twinx()
    bx1.plot(df_alpha['step'], df_fixed['d6_preact_std'], '-', color=color_list[1], linewidth=linewidth)
    bx1.grid(linestyle=':', color=color_list[1], axis='y')
    bx1.tick_params(axis='y', colors=color_list[1])
    bx1.set_ylim(ylim1)
    bx2.plot(df_alpha['step'], df_fixed['d6_weights_std'], '-', color=color_list[0], linewidth=linewidth)
    bx2.tick_params(axis='y', colors=color_list[0])
    bx2.set_ylim(ylim2)
    plt.locator_params(axis='x', nbins=3)
    plt.locator_params(axis='y', nbins=5)
    bx1.set_xticklabels(['0','20k','40k','60k'])
    bx1.set_title(r'tanh')


    handles = l1+l2+l3
    labels = [l.get_label() for l in handles]
    lgd = ax1.legend(handles, labels, loc='center left',  bbox_to_anchor=(1.1,0.5), fancybox=False, shadow=False)

    plt.tight_layout()
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    fig.savefig(savepath+filename, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=150)
    plt.close()

# ##############################################################################
# ### SCRIPT ###################################################################
# ##############################################################################

# alpha tanh
relative_path = '3_output_tensorboard/newplot_alpha_tanh/'
files_d5_scaling = list_files_in_path(relative_path, contains='dense5_blend_weights')
files_d6_weights = list_files_in_path(relative_path, contains='dense6_weights_stddev')
files_d6_preacts = list_files_in_path(relative_path, contains='dense6b_preact_stddev')
# build one large dataframe
alpha_df = pd.DataFrame
for i, file in enumerate(files_d5_scaling):
    # extract relevant info from file name
    spec = file.split('c10_')[-1].split('-tag-')[0][:-2]
    run = int(file.split('-tag')[0].split('_')[-1])
    # read std and mean csv
    csv = pd.read_csv(file)
    csv = csv.drop(columns=['Wall time'])
    csv = csv.rename(columns={"Value": 'd5_alpha'})
    csv = csv.rename(columns={"Step": 'step'})
    csv['spec'] = pd.Series(spec, index=csv.index)
    csv['run'] = pd.Series(run, index=csv.index)
    # extract info from other files 1
    preact_stats_file = file.split('dense')[0]+'dense6_preact_stddev.csv'
    csv_preact_stats = pd.read_csv(preact_stats_file)
    csv['d6_preact_std'] = pd.Series(csv_preact_stats['Value'], index=csv.index)
    # extract info from other files 2
    weights_file = file.split('dense')[0]+'dense6_weights_stddev.csv'
    csv_weights = pd.read_csv(weights_file)
    csv['d6_weights_std'] = pd.Series(csv_weights['Value'], index=csv.index)
    # concatenate
    if i == 0:
        alpha_df = csv
    else:
        alpha_df = pd.concat([alpha_df, csv])
# get means over runs
alpha_df = alpha_df[['spec', 'run', 'd5_alpha', 'd6_weights_std', 'd6_preact_std', 'step']]
alpha_df_mean = alpha_df.groupby('step').mean() # mean over runs
alpha_df_mean = alpha_df_mean.drop(columns=['run'])
alpha_df_mean = alpha_df_mean.reset_index()

# tanh
relative_path = '3_output_tensorboard/newplot_tanh/'
files_d6_weights = list_files_in_path(relative_path, contains='dense6_weights_stddev')
files_d6_preacts = list_files_in_path(relative_path, contains='dense6b_preact_stddev')
# build one large dataframe
fixed_df = pd.DataFrame
for i, file in enumerate(files_d6_weights):
    # extract relevant info from file name
    spec = file.split('c10_')[-1].split('-tag-')[0][:-2]
    run = int(file.split('-tag')[0].split('_')[-1])
    # read std and mean csv
    csv = pd.read_csv(file)
    csv = csv.drop(columns=['Wall time'])
    csv = csv.rename(columns={"Value": 'd6_weights_std'})
    csv = csv.rename(columns={"Step": 'step'})
    csv['spec'] = pd.Series(spec, index=csv.index)
    csv['run'] = pd.Series(run, index=csv.index)
    # extract info from other file
    weights_file = file.split('dense')[0]+'dense6_preact_stddev.csv'
    csv_weights = pd.read_csv(weights_file)
    csv['d6_preact_std'] = pd.Series(csv_weights['Value'], index=csv.index)
    # concatenate
    if i == 0:
        fixed_df = csv
    else:
        fixed_df = pd.concat([fixed_df, csv])
# get means over runs
fixed_df = fixed_df[['spec', 'run', 'd6_weights_std', 'd6_preact_std', 'step']]
fixed_df_mean = fixed_df.groupby('step').mean() # mean over runs
fixed_df_mean = fixed_df_mean.drop(columns=['run'])
fixed_df_mean = fixed_df_mean.reset_index()

# plot
plot_blend_filter_preact([alpha_df_mean, fixed_df_mean], '3_result_plots/', 'blend_filter_preact_tanh_d5to6.png', figsize=(9.5,2.4))
