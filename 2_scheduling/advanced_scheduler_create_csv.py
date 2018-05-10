import advanced_scheduler as asc

if __name__ == '__main__':

    scheduling_subfolder, experiment_path, experiment_name, spec_list, n_runs = asc.settings()
    asc.create_scheduler_csv(spec_list, n_runs, experiment_name)
