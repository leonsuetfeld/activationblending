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
    print(sample_1[3] + ' vs. ' + sample_2[3] + ':')
    print('t = ' + str(t_val) + ', df = ' + str(df) + ', p = ' + str(p_val))
    print(sample_1[3] + ' var=' + str(sample_1[2]) + ', std=' + str(np.sqrt(sample_1[2])))
    print(sample_2[3] + ' var=' + str(sample_2[2]) + ', std=' + str(np.sqrt(sample_2[2])))
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

spec1 = [20, 0.813, 7*(1./10**5), 'A']
spec2 = [20, 0.814, 7*(1./10**5), 'B']

print('')
t_val, p_val = two_sample_t_test(spec1, spec2, plotting=True)
print('')
