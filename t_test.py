import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt

def two_sample_t_test(triplet_1, triplet_2):
    # unzip
    n_1 = triplet_1[0]
    mean_1 = triplet_1[1]
    var_1 = triplet_1[2]
    n_2 = triplet_2[0]
    mean_2 = triplet_2[1]
    var_2 = triplet_2[2]
    # get t-value
    mvar = ( (n_1-1)*var_1 + (n_2-1)*var_2 ) / (n_1+n_2-2) # is it correct to use var here, or std?
    t_val = (mean_1 - mean_2) / (mvar * np.sqrt(1/n_1+1/n_2)) # is all of this correct?

    print(mean_1-mean_2)
    print(mvar)
    print(np.sqrt(1/n_1+1/n_2))
    print((mvar * np.sqrt(1/n_1+1/n_2)))
    print("\n")

    # get p-value
    df = n_1+n_2-2
    p_val = t.cdf(t_val, df)
    # return
    return t_val, p_val

elu = [100, 81.3578, 00.005295]
aelu = [100, 81.8938, 00.005422]
b5u = [100, 82.2148, 00.004827]

# TO DO:
# plot normal distributions to make sure the values are correct!
# check calculation of var values in other script!

t_aelu_elu, p_aelu_elu = two_sample_t_test(aelu, elu)
t_b5u_aelu, p_b5u_aelu = two_sample_t_test(b5u, aelu)
t_b5u_elu, p_b5u_elu = two_sample_t_test(b5u, elu)

print(t_aelu_elu, p_aelu_elu)
print(t_b5u_aelu, p_b5u_aelu)
print(t_b5u_elu, p_b5u_elu)
print('\n')

for t_val in [0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
    p = t.cdf(t_val, 198)
    print(1.0-p)

# plot t-distribution

df = 198
x = np.linspace(t.ppf(0.0001, df), t.ppf(0.9999, df), 100)
y = t.pdf(x, df)
plt.plot(x, y, color='black', lw=2, alpha=1.0, label='t pdf')
plt.grid()
plt.ylim([0.0,0.5])
plt.xlim([-3,3])
plt.show()
