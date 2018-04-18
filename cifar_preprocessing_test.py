import numpy as np
import cifar_preprocessing as cpp
import tensorflow as tf

print('')
print('#######################################################################')

# load and reshape images
dataset, _ = cpp.load_cifar10(max_size=1000)
dataset = cpp.reshape_cifar(dataset)
dataset = dataset.astype(np.float32) # set to float 32 before the calculations to test if this decreases performance by introducing noise

# ##############################################################################
# ### TF #######################################################################
# ##############################################################################

X1 = tf.convert_to_tensor(dataset)
X1_per_image_mean, X1_per_image_var = tf.nn.moments(X1, [1,2,3], name='image_var', keep_dims=True)
X2 = X1-X1_per_image_mean
minval = tf.divide(tf.fill(X1_per_image_var.get_shape(),1.), tf.sqrt(3072.))
X1_var_adjusted = tf.maximum(X1_per_image_var, minval)
X3 = tf.divide(X2, X1_var_adjusted)

sess = tf.Session()
with sess.as_default():
	TF_X1 = X1.eval()
	TF_X2 = X2.eval()
	TF_X3 = X3.eval()
	TF_X1_mean = X1_per_image_mean.eval()
	TF_X1_var = X1_per_image_var.eval()
	TF_X1_var_adj = X1_var_adjusted.eval()
	TF_minval = minval.eval()

# ##############################################################################
# ### NP #######################################################################
# ##############################################################################

X1 = []
X2 = []
X3 = []
X1_mean = []
X1_var = []
X1_var_adj = []
minval = []

for i in range(dataset.shape[0]):
	# X1
	img1 = dataset[i,:,:,:]
	img1 = img1.astype(np.float32)
	X1.append(img1)
	# X1_mean
	img1_mean = np.mean(img1, dtype=np.float32)
	X1_mean.append(img1_mean)
	# X1_var
	img1_var = np.var(img1, dtype=np.float32) # changed std to var to be able to compare to tf. change back!
	X1_var.append(img1_var)
	# minval
	mv = 1. / np.sqrt(3072.)
	minval.append(mv)
	# X1_var_adj
	img1_var = np.maximum(img1_var, mv, dtype=np.float32)
	X1_var_adj.append(img1_var)
	# X2
	img2 = img1-img1_mean
	X2.append(img2)
	# X3
	img3 = np.divide(img2, img1_var, dtype=np.float32)
	X3.append(img3)

NP_X1 = np.array(X1)
NP_X2 = np.array(X2)
NP_X3 = np.array(X3)
NP_X1_mean = np.array(X1_mean)
NP_X1_var = np.array(X1_var)
NP_X1_var_adj = np.array(X1_var_adj)
NP_minval = np.array(minval)

# ##############################################################################
# ### COMPARISON ###############################################################
# ##############################################################################

DIFF_X1 = NP_X1 - TF_X1
DIFF_X2 = NP_X2 - TF_X2
DIFF_X3 = NP_X3 - TF_X3
DIFF_X1_mean = np.squeeze(NP_X1_mean) - np.squeeze(TF_X1_mean)
DIFF_X1_var = np.squeeze(NP_X1_var) - np.squeeze(TF_X1_var)
DIFF_X1_var_adj = np.squeeze(NP_X1_var_adj) - np.squeeze(TF_X1_var_adj)
DIFF_minval = np.squeeze(NP_minval) - np.squeeze(TF_minval)

# print(NP_minval.shape, TF_minval.shape)
# print(DIFF_minval)

print(DIFF_X1_var)
# cpp.show_images(X3)

print('')
print('=======================================================================')
print('')
