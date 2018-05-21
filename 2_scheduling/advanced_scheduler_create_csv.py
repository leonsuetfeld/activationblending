import advanced_scheduler as asc

if __name__ == '__main__':

    scheduling_subfolder, experiment_path, experiment_name, spec_list, n_runs, gridjob_command = asc.get_settings()
    asc.create_scheduler_csv(spec_list, n_runs, experiment_name)
    asc.csv_update(scheduling_subfolder, experiment_name+'.csv', experiment_path, experiment_name, mark_running_as_incomplete=False) # set this to False when executing during gridjob
