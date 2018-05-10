import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math

def two_sample_t_test_unpaired_data(sample_1, sample_2, plotting=False, equal_variances=False):
    # unzip
    n_1 = sample_1[0]
    mean_1 = sample_1[1]
    var_1 = sample_1[2]
    n_2 = sample_2[0]
    mean_2 = sample_2[1]
    var_2 = sample_2[2]
    # get t-value
    if equal_variances:
        mvar = ( (n_1-1)*var_1 + (n_2-1)*var_2 ) / (n_1+n_2-2)
        t_val = (mean_1 - mean_2) / (np.sqrt(mvar) * np.sqrt( 1/n_1 + 1/n_2 ))
    else:
        t_val = (mean_1 - mean_2) / np.sqrt( var_1/n_1 + var_2/n_2 )
    # get p-value
    if equal_variances:
        df = n_1+n_2-2
    else:
        df = (var_1/n_1 + var_2/n_2) / ( ((var_1/n_1)**2)/(n_1-1) + ((var_2/n_2)**2)/(n_2-1) )
    p_val = t.cdf(t_val, df)
    # print & plot
    print(sample_1[3] + ' vs. ' + sample_2[3] + ':')
    if equal_variances:
        print('t = ' + str(t_val) + ', df = ' + str(df) + ', p = ' + str(p_val) + ', std_pooled = ' +str(np.sqrt(mvar)))
    else:
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

spec1 = [30, 0.813, 5*(1./10**5), 'A']
spec2 = [30, 0.816, 5*(1./10**5), 'B']

print('\n####################################################################\n')
t_val, p_val = two_sample_t_test_unpaired_data(spec1, spec2, plotting=False, equal_variances=True)
print('\n####################################################################\n')
t_val, p_val = two_sample_t_test_unpaired_data(spec1, spec2, plotting=False, equal_variances=False)
print('\n####################################################################\n')
