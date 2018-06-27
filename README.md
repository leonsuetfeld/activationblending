# Adaptive Blending Units: Trainable Activation Functions for Deep Neural Networks
https://128.84.21.199/abs/1806.10064v1

## Abstract
The most widely used activation functions in current deep feed-forward neural networks are rectified linear units (ReLU), and many alternatives have been successfully applied, as well. However, none of the alternatives have managed to consistently outperform the rest and there is no unified theory connecting properties of the task and network with properties of activation functions for most efficient training. A possible solution is to have the network learn its preferred activation functions. In this work, we introduce Adaptive Blending Units (ABUs), a trainable linear combination of a set of activation functions. Since ABUs learn the shape, as well as the overall scaling of the activation function, we also analyze the effects of adaptive scaling in common activation functions. We experimentally demonstrate advantages of both adaptive scaling and ABUs over common activation functions across a set of systematically varied network specifications. We further show that adaptive scaling works by mitigating covariate shifts during training, and that the observed advantages in performance of ABUs likewise rely largely on the activation function's ability to adapt over the course of training.

## Code Files
### deepnet_supervisor.py
Launches deepnet_main.py, contains all externally available settings, passes them on as arguments.

### deepnet_main.py
Creates all relevant object instances, launches training, testing or analysis.

### deepnet_network.py
Contains the network including all available options.

### deepnet_task_cifar.py
Contains everything task-related. Most importantly, train(), test(), and analyze().
