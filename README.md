# activations

deepnet_supervisor.py
Launches deepnet_main.py, contains all externally available settings, passes them on as arguments.

deepnet_main.py
Creates all relevant object instances, launches training, testing or analysis.

deepnet_network.py
Contains the network including all available options.

deepnet_task_cifar.py
Contains everything task-related. Most importantly, train(), test(), and analyze().
