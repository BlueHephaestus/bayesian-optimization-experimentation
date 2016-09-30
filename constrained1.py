import sys

import numpy as np
import matplotlib.pyplot as plt#For plotting end results

#Covariance Function
#Squared exponential covariance function
cov = lambda x_i, x_j: np.exp(-.95*np.linalg.norm(x_i - x_j)**2) 

#Given only this data, compute point mean and variance for test input
#alex = lambda x: np.sin(x)*np.cos(x)
alex = lambda x: np.sin(x**(2.5/2.0))+(x/3.6)**2

def gaussian_distribution(x, mean, variance):
    #Gets input x, mean, and variance
    #Returns vector or scalar from input
    return (np.exp(-((x-mean)**2)/(2*variance))) / (np.sqrt(2*variance*np.pi))


def cdf(x, mean, variance):
    #Get values to compute cdf over
    dist_values = gaussian_distribution(np.arange(x-100, x, .1), mean, variance)
    #print mean, variance
    #plt.plot(np.arange(x-10, x, .01), dist_values)
    #plt.show()
    #sys.exit()
    
    #Equivalent to the last element of cumulative sum
    return sum(dist_values)


training = [[[1], alex(1)], [[3], alex(3)], [[5], alex(5)], [[7], alex(7)], [[8.45], alex(8.45)], [[9.9], alex(9.9)]]
#training = 
#For debugging currently
n = 1000
test_means = np.zeros(shape=(n))
test_variances = np.zeros(shape=(n))

#print len(np.arange(0, n/100.0, 0.01))
for i in range(n):
    j = i / (n/10.0)

    test_input = np.array([j])

    #Get accordingly
    training_inputs = [np.array(pair[0]) for pair in training]
    training_outputs = [np.array(pair[1]) for pair in training]

    #Covariance matrix
    cov_m = np.zeros(shape=(len(training), len(training)))

    #Populate Covariance matrix
    for row, x_i in enumerate(training_inputs):
        for col, x_j in enumerate(training_inputs):
            cov_m[row, col] = cov(x_i, x_j)

    #Alternative way to do noisy regression
    #cov_m = cov_m + 0.5*np.eye(len(training))

    #Test data covariance values
    test_cov = np.zeros(shape=(len(training), 1))

    #Populate test data covariance vector
    for row, x_i in enumerate(training_inputs):
        test_cov[row] = cov(test_input, x_i)

    #Get our transpose
    test_cov_T = test_cov.transpose()

    #Get diag value (This may be different than 1.0 if different covariance function)
    test_cov_diag = cov(test_input, test_input)

    """
    #Compute test mean using our Multivariate Gaussian Theorems
    test_mean = test_cov_T * np.linalg.inv(cov_m) * training_outputs

    #Compute test variance using our Multivariate Gaussian Theorems
    test_variance = test_cov_diag - test_cov_T * np.linalg.inv(cov_m) * test_cov
    """
    #Compute test mean using our Multivariate Gaussian Theorems
    test_mean = np.dot(np.dot(test_cov_T, np.linalg.inv(cov_m)), training_outputs)

    #Compute test variance using our Multivariate Gaussian Theorems
    test_variance = test_cov_diag - np.dot(np.dot(test_cov_T, np.linalg.inv(cov_m)), test_cov)

    #print test_input, test_mean, test_variance

    test_means[i] = test_mean
    test_variances[i] = test_variance + 0.05

test_x = np.arange(0, n/100.0, 0.01)
training_x = np.array(training_inputs).flatten()

#Get the probability of improvement for each point in our means
improvement_probs = np.zeros(shape=(len(test_means)))
for i in range(len(test_means)):
    #len(test_means) just used so we can reference
    mean = test_means[i]
    variance = test_variances[i]
    val = mean + variance
    improvement_probs[i] = mean - 0.95*variance
    #improvement_probs[i] = cdf(val, mean, variance)

print np.argmax(improvement_probs)
print np.max(improvement_probs)

#plt.subplot(1, 2, 1)
plt.plot(test_x, test_means, 'b')
plt.plot(training_x, training_outputs, 'ro')
plt.plot(test_x, test_means + test_variances, 'm')
plt.plot(test_x, test_means - test_variances, 'm')

#plt.subplot(1, 2, 2)
#plt.plot(test_x, test_variances, 'g')
plt.show()


