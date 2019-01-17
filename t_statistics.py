# Compute t-statistics with p-value = 0.05 that DETERMINISTIC and STOCHASTIC are different:
import pickle
import numpy as np
from scipy import stats


### Load DETERMINISTIC error:
with open('networks/networks_dropout/results/det_validation_error_dropout.pkl') as f:  # Python 3: open(..., 'wb')
    det_error_dropout, det_std_dropout = pickle.load(f)

with open('networks/networks_NOdropout/results/deterministic_error_validationNOdropout.pkl') as f:  # Python 3: open(..., 'wb')
    det_error_NOdropout, det_std_NOdropout = pickle.load(f)



### Load STOCHASTIC error:
with open('networks/networks_dropout/results/stoch_validation_error_dropout.pkl') as f:  # Python 3: open(..., 'wb')
    stoch_error_dropout, stoch_std_dropout = pickle.load(f)

with open('networks/networks_NOdropout/results/stoch_validation_error_NOdropout.pkl') as f:  # Python 3: open(..., 'wb')
    stoch_error_NOdropout, stoch_std_NOdropout = pickle.load(f)




############
det_error = det_error_dropout
det_error_std = det_std_dropout
stoch_error = stoch_error_dropout
stoch_error_std = stoch_std_dropout


det_error = det_error_NOdropout
det_error_std = det_std_NOdropout
stoch_error = stoch_error_NOdropout
stoch_error_std = stoch_std_NOdropout

#Sample size (conservative): size of deterministic distrib. 5 networks
p_desired = 0.01
n1 = 5
n2 = 250

p_table = np.zeros((np.shape(stoch_error_dropout)))
p_table_significant = np.zeros((np.shape(stoch_error_dropout)))

for e in xrange(0, np.shape(stoch_error_dropout)[0]):
    for m in xrange (0, np.shape(stoch_error_dropout)[1]):


        print(e)
        print(m)

        det_mean = det_error[e][m]
        det_std = det_error_std[e][m]
        stoch_mean = stoch_error[e][m]
        stoch_std = stoch_error_std[m][e]

        # t-statistics
        s1 = det_std**2
        s2 = stoch_std**2
        std_t = np.sqrt( (s1/n1) + (s2/n2) )

        t = (det_mean - stoch_mean) / std_t

        ## Compare with the critical t-value
        #Degrees of freedom
        df = n1 -1

        #p-value after comparison with the t
        p = 1 - stats.t.cdf(t,df=df)


        # print("t = " + str(t))
        # print("p = " + str(2*p)) #Note that we multiply the p value by 2 because its a twp tail t-test

        p_table[e][m] = 2*p

        if (2*p < p_desired):
            p_table_significant[e][m] = 1

print(p_table_significant)




with open('networks/networks_dropout/results/t_stats.pkl', 'w') as f:  # Python 3: open(..., 'wb')
    pickle.dump([p_table, p_table_significant], f)


with open('networks/networks_NOdropout/results/t_stats.pkl', 'w') as f:  # Python 3: open(..., 'wb')
    pickle.dump([p_table, p_table_significant], f)
