import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math

def two_sample_t_test(sample_1, sample_2, plotting=False):
    # unzip
    n_1 = sample_1[0]
    mean_1 = sample_1[1]
    var_1 = sample_1[2]
    n_2 = sample_2[0]
    mean_2 = sample_2[1]
    var_2 = sample_2[2]
    # get t-value
    mvar = ( (n_1-1)*var_1 + (n_2-1)*var_2 ) / (n_1+n_2-2)
    t_val = (mean_1 - mean_2) / (mvar * np.sqrt( 1/n_1 + 1/n_2 ))
    # get p-value
    df = n_1+n_2-2
    p_val = 1.0-t.cdf(t_val, df)
    # print & plot
    print(sample_1[3] + ' vs. ' + sample_2[3] + ': t = ' + str(t_val) + ', df = ' + str(df) + ', p = ' + str(p_val))
    if plotting:
        sigma1 = math.sqrt(var_1)
        sigma2 = math.sqrt(var_2)
        x1 = np.linspace(mean_1 - 6*sigma1, mean_1 + 6*sigma1, 100)
        x2 = np.linspace(mean_2 - 6*sigma2, mean_2 + 6*sigma2, 100)
        plt.plot(x1,mlab.normpdf(x1, mean_1, sigma1))
        plt.plot(x2,mlab.normpdf(x2, mean_2, sigma2))
        plt.title(p_val)
        plt.show()
    # return
    return t_val, p_val

# ############################################################################ #
# ### SCRIPT ################################################################# #
# ############################################################################ #

elu = [100, 81.3578, 00.005295, 'elu']
aelu = [100, 81.8938, 00.005422, 'a-elu']
b5u = [100, 82.2148, 00.004827, 'b5u']

print('')
t_aelu_elu, p_aelu_elu = two_sample_t_test(aelu, elu, plotting=False)
t_b5u_aelu, p_b5u_aelu = two_sample_t_test(b5u, aelu, plotting=False)
t_b5u_elu, p_b5u_elu = two_sample_t_test(b5u, elu, plotting=False)
print('')
