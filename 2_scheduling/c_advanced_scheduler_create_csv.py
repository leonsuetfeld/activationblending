import c_advanced_scheduler as asc

if __name__ == '__main__':

    scheduling_subfolder, experiment_path, experiment_name, spec_list, n_runs, gridjob_command = asc.get_settings()
    asc.csv_update(scheduling_subfolder, experiment_name+'.csv', experiment_path, experiment_name, spec_list, n_runs, mark_running_as_incomplete=True) # set this to False when executing during gridjob
