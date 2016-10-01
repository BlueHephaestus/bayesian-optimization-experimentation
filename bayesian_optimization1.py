"""
Penultimate program - making to port over to full on machine learning applications after testing here.

Still working on a name for the full one, it won't be CHO MK 2 since it now uses a conventional hyper 
  parameter tuning method, but I don't want to call it just Bayesian Optimization (BO) since that would
  omit some details, like hyper parameters tuning. I'll figure it out, but for now, this program works like this:

Given an input function,

1. Get input domains via xrange(min, max, ((max-min)/n)) where n is arbitrary level of detail, i.e. 10,000 
    and min/max are defined beforehand for each input
2. Generate two random input vectors x1, x2 where each element is min <= x <= max and on our domain
3. Evaluate x1 & x2 to get f1 & f2
4. Generate n test means(u* s) and test variances(c* s) across domain via the following for each test input x*:
    a. Generate covariance matrix with known inputs
    b. Generate covariance vector and transpose with known inputs and test input, as well as diagonal
    c. u* = (K* transpose) * (K inverse) * (known_evaluations)
    d. c* = (K* diag) - (K* transpose) * (K inverse) * (K*)
5. 
    If maximizing, set output at each test x (f(x*)) = u* + c*
    If minimizing, set f(x*) = u* - c*

6. Generate upper confidence bound acquisition function vector and get best new inputs via argmaxing it:
    x_i = domain[argmax(u* - confidence_interval * c*)

    Note: Using 95% confidence interval (.95) since that's what everyone seems to be using

7. Go back to step 3 until i == end_iteration
8. Final inputs = domain[argmax(f*)]
"""
import sys

import numpy as np
import matplotlib.pyplot as plt

import hyperparameter
from hyperparameter import HyperParameter

#Our chosen test black box optimization function
def bbf(x):
   #return np.sin(x**(2.5/2.0))+(x/3.6)**2
   return np.exp(-(x-2)**2) + np.exp(-((x-6)**2)/10.0) + (1.0/(x**2 + 1))
   #return (x-7)**3 -3.5*(x-7)**2 + 4

def gaussian_distribution(x, mean, variance):
    #Gets input x, mean, and variance
    #Returns vector or scalar from input
    return (np.exp(-((x-mean)**2)/(2*variance))) / (np.sqrt(2*variance*np.pi))

def cdf(x, mean, variance):
    #Get values to compute cdf over
    dist_values = gaussian_distribution(np.arange(x-100, x, .1), mean, variance)
    
    #Equivalent to the last element of cumulative sum
    return sum(dist_values)

def cov(x_i, x_j):
    return np.exp(-.95*np.linalg.norm(x_i - x_j)**2) 

def get_cov_matrix(f):
    #Given a vector f, generate the covariance matrix 
    #f because known inputs
    f_n = len(f)
    f_m = np.zeros(shape=(f_n, f_n))
    for row_i, f_i in enumerate(f):
        for col_i, f_j in enumerate(f):
            f_m[row_i, col_i] = cov(f_i, f_j)

    return f_m

def get_cov_vector(f, test_f):
    #Given a vector f and scalar f* (test_f)
    #Generate a covariance vector for each value in f
    f_n = len(f)
    f_v = np.zeros(shape=(f_n, 1))
    for row_i, f_i in enumerate(f):
        f_v[row_i] = cov(test_f, f_i)

    return f_v
        
class upper_confidence_bound(object):
    """CURRENTLY BROKEN AFAIK"""
    
    @staticmethod
    def evaluate(means, variances, values, confidence_interval):
        return np.argmax(means + confidence_interval * variances)

class probability_improvement(object):
    
    @staticmethod
    def evaluate(means, variances, values, confidence_interval):
        improvement_probs = [np.nan_to_num(cdf(val, mean, variance)) for val, mean, variance in zip(values, means, variances)]
        return np.argmax(improvement_probs)

#Number of evaluated input points / level of detail
n = 1000

#If we want the highest point or lowest point
maximizing = True

#Number of bbf evaluations allowed to perform before ending optimization
bbf_evaluation_n = 10

#Choice of acquisition function
#acquisition_function = upper_confidence_bound()
acquisition_function = probability_improvement()

#Acquisition function hyper parameter choice
confidence_interval = .95

#Initialize ranges for each parameter into a resulting matrix
hps = [HyperParameter(0, 10)]

"""END TUNABLE PARAMETERS"""

#Initialize multivariate(if len(hps) > 1) input domain
#We transpose it so we can loop through our dimensional inputs as you would since they are vectors
domain = np.array([np.arange(hp.min, hp.max, ((hp.max-hp.min)/float(n))) for hp in hps])
domain_T = domain.transpose()

#Temporary domain so we can make sure to get two DIFFERENT values 
shuffled_domain = np.copy(domain.transpose())
np.random.shuffle(shuffled_domain)

#Get our different values
x1, x2 = shuffled_domain[:2]

#Now that we have our two random input vectors, evaluate them and store them in our bbf inputs and outputs vector
#Modify the bbf function when you make this more complicated with input to a bot
#This needs to not be a np.array since we have to append
bbf_inputs = [x1, x2]

#This needs to be np array so we can do vector multiplication
bbf_evaluations = np.array([bbf(x1), bbf(x2)])

#Our main loop to go through every time we evaluate a new point, until we have exhausted our allowed 
#   black box function evaluations.
for bbf_evaluation_i in range(2, bbf_evaluation_n):

    #Since we reset this every time we generate through the domain
    test_means = np.zeros(shape=(n))
    test_variances = np.zeros(shape=(n))
    test_values = np.zeros(shape=(n))

    for test_input_i, test_input in enumerate(domain_T):
        #Generate our covariance matrices and vectors
        training_cov_m = get_cov_matrix(bbf_inputs)#K
        training_cov_m_inv = np.linalg.inv(training_cov_m)#K^-1
        test_cov = get_cov_vector(bbf_inputs, test_input)#K*
        test_cov_T = test_cov.transpose()#K*T
        test_cov_diag = cov(test_input, test_input)#K**

        #Compute test mean using our Multivariate Gaussian Theorems
        #print test_cov_T.shape, training_cov_m_inv.shape, bbf_evaluations.shape
        test_mean = np.dot(np.dot(test_cov_T, training_cov_m_inv), bbf_evaluations)

        #Compute test variance using our Multivariate Gaussian Theorems
        test_variance = test_cov_diag - np.dot(np.dot(test_cov_T, training_cov_m_inv), test_cov)

        #Store them for use with our acquisition function
        test_means[test_input_i] = test_mean
        test_variances[test_input_i] = test_variance

    #Now that we have all our means u* and variances c* for every point in the domain,
    #Move on to determining next point to evaluate using our acquisition function
    #If we want the point that will give us next greatest input, do u + c, otherwise u - c
    if maximizing:
        test_values = test_means + test_variances
    else:
        test_values = test_means - test_variances

    #Plot our updates
    """
    plt.plot(domain[0], test_means)
    plt.plot(bbf_inputs, bbf_evaluations, 'bo')
    plt.plot(domain[0], test_means+test_variances, 'r')
    plt.plot(domain[0], test_means-test_variances, 'r')
    plt.plot(domain[0], bbf(domain[0]), 'y')
    plt.show()
    """

    #Get the index of the next input to evaluate in our black box function
    #Since acquisition functions return argmax values
    next_input_i = acquisition_function.evaluate(test_means, test_variances, test_values, confidence_interval)

    #Add our new input
    next_input = domain_T[next_input_i]
    bbf_inputs.append(np.array(next_input))

    #Evaluate new input
    #We need this as nparray for vector multiplication
    #But we need to append as well, so we have to do this.
    #Luckily, it's our smallest np array
    bbf_evaluations = list(bbf_evaluations)
    bbf_evaluations.append(bbf(next_input))
    bbf_evaluations = np.array(bbf_evaluations)

best_input = bbf_inputs[np.argmax(bbf_evaluations)]
print "Best input found after {} iterations: {}".format(bbf_evaluation_n, best_input[0])
