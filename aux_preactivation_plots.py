import numpy as np
import pandas as pd
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

def plot_preact_comparison(specs, savepath, filename, plot_mean_over_runs=True, plot_individual_runs=False, linewidth=2, figsize=(14.27,5), hide_yticks=False):
    color_list = ['#8c9fcb','#fc8d62', '#e78ac3', '#a5d853', '#66c2a5', '#b4b4b4', '#ffd82e', '#e4c494']
    layer_list_paper = ['conv1','conv2','conv3','conv4','dense1','dense2']
    fig = plt.figure(figsize=figsize)
    for s, stat in enumerate(['mean', 'std']):
        for l, layer in enumerate(['conv1','conv2','conv3','conv4','dense5','dense6']):
            val_collection = []
            ax = fig.add_subplot(2, 6, s*6+l+1)
            ax.grid(linestyle=':', alpha=0.5)
            for sp, spec in enumerate(specs):
                # extract data from dataframe
                data_step, data_step, data_stat, data_stat = [], [], [], []
                for run in range(5):
                    data_step.append(get_single_spec_run_layer_column(df, spec=spec, run=run+1, layer=layer, column='step'))
                    data_stat.append(get_single_spec_run_layer_column(df, spec=spec, run=run+1, layer=layer, column=stat))
                data_step = np.array(data_step)
                data_stat = np.array(data_stat)
                mean_data_step = np.mean(data_step, axis=0, keepdims=True)
                mean_data_stat = np.mean(data_stat, axis=0, keepdims=True)
                val_collection.extend(mean_data_stat.tolist()) # cumulate specs for min and max values
                # plot
                if s==0:
                    ax.plot([0,60000],[0,0], '-', color='#000000', linewidth='1', alpha=1.0)
                if plot_individual_runs:
                    for run in range(5):
                        ax.plot(data_step[run,:], data_stat[run,:], '-', color=color_list[sp], linewidth=linewidth*0.5, alpha=0.2)
                if plot_mean_over_runs:
                    ax.plot(mean_data_step[0,:], mean_data_stat[0,:], '-', color=color_list[sp], linewidth=linewidth, alpha=1.0, label=spec)
                if hide_yticks:
                    # plt.tick_params(axis='y', which='both', labelleft='off') # labels along the left and right off
                    ax.set_yticklabels(['2','2','2','2','2','2','2','2','2'])
                if s==0:
                    plt.tick_params(axis='x', which='both', labelbottom='off') # labels along the left and right off
            # ylim
            max_val = np.array(val_collection).max()
            min_val = np.array(val_collection).min()
            val_range = max_val-min_val
            min_buffer = -0.15*val_range
            max_buffer = 0.15*val_range
            if stat == 'std':
                min_buffer = 0.
            ylim = [np.minimum(min_val+min_buffer, 0), np.maximum(max_val+max_buffer, 0)]
            ax.set_ylim(ylim)
            # prettier plot
            if s==0:
                ax.set_title(layer_list_paper[l])
            if l==0:
                ax.set_ylabel('pre-activation '+stat)
            plt.locator_params(axis='x', nbins=3)
            plt.locator_params(axis='y', nbins=5)
            ax.set_xticklabels(['0','20k','40k','60k','','',''])
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='center left',  bbox_to_anchor=(1.1,1), fancybox=False, shadow=False)
    plt.tight_layout()
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    fig.savefig(savepath+filename, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=150)
    plt.close()

# ##############################################################################
# ### SCRIPT ###################################################################
# ##############################################################################

relative_path = '3_output_tensorboard/'
spec_paths = list_subfolders(add_cwd_to_path(relative_path))
spec_paths = [f for f in spec_paths if not 'newplot' in f] # throw out irrelevant paths

# get list of all files in all subfolders of 'relative_path'
all_std_files = []
for spec_path in spec_paths:
    files_in_spec = list_files_in_path(spec_path, contains='stddev')
    all_std_files.extend(files_in_spec)

# build one large dataframe
df = pd.DataFrame
for i, file in enumerate(all_std_files):
    # extract relevant info from file name
    spec = file.split('c10_')[-1].split('-tag-')[0][:-2]
    run = int(file.split('-tag')[0].split('_')[-1])
    layer = file.split('_smcn_')[-1].split('_preact')[0]
    stat = 'std'
    # read std and mean csv
    csv = pd.read_csv(file)
    csv = csv.drop(columns=['Wall time'])
    csv = csv.rename(columns={"Value": 'std'})
    csv = csv.rename(columns={"Step": 'step'})
    mean_file = file.split('stddev')[0]+'mean'+file.split('stddev')[1]
    mean_file = mean_file.split('std/')[0]+'mean/'+mean_file.split('std/')[1]
    csv_mean = pd.read_csv(mean_file)
    # combine mean and std and add info
    csv['mean'] = pd.Series(csv_mean['Value'], index=csv.index)
    csv['spec'] = pd.Series(spec, index=csv.index)
    csv['run'] = pd.Series(run, index=csv.index)
    csv['layer'] = pd.Series(layer, index=csv.index)
    # concatenate
    if i == 0:
        df = csv
    else:
        df = pd.concat([df, csv])
df = df[['spec', 'run', 'layer', 'mean', 'std', 'step']]

# make plots
plot_preact_comparison(['tanh', 'alpha_tanh'], '3_result_plots/', 'preact_tanh.png', linewidth=1.5)
plot_preact_comparison(['relu', 'alpha_relu'], '3_result_plots/', 'preact_relu.png', linewidth=1.5)
plot_preact_comparison(['elu', 'alpha_elu'], '3_result_plots/', 'preact_elu.png', linewidth=1.5)
plot_preact_comparison(['ABU_TERIS', 'ABU_S_TERIS', 'ABU_N_TERIS', 'ABU_A_TERIS', 'ABU_P_TERIS'], '3_result_plots/', 'preact_ABU.png', linewidth=1.5, figsize=(11.69,4))
plot_preact_comparison(['ABU_TERIS', 'ABU_S_TERIS', 'ABU_N_TERIS', 'ABU_A_TERIS', 'ABU_P_TERIS'], '3_result_plots/', 'preact_ABU_space.png', linewidth=1.5, figsize=(11.69,4), hide_yticks=True)
