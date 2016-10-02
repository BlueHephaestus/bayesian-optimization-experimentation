import numpy as np

import gaussian_base
from gaussian_base import *

class probability_improvement(object):
    
    @staticmethod
    def evaluate(means, variances, values, confidence_interval):
        improvement_probs = np.array([np.nan_to_num(cdf(val, mean, np.sqrt(variance))) for val, mean, variance in zip(values, means, variances)])
        return np.argmax(improvement_probs)

class expected_improvement(object):
    @staticmethod
    def evaluate(means, variances, values, confidence_interval):
        #Get our output values from plugging into our distribution x values
        dist_values = np.array([gaussian_distribution(val, mean, np.sqrt(variance)) for val, mean, variance in zip(values, means, variances)])

        #Get same values but for normal distribution
        normal_dist_values = np.array([gaussian_distribution(val, 0, 1) for val in values])

        #Get our cumulative distribution values, the equivalent of our probability_improvement function but without argmax yet
        cdfs = np.array([np.nan_to_num(cdf(val, mean, np.sqrt(variance))) for val, mean, variance in zip(values, means, variances)])

        #Finally, generate full array now that we have all the component parts
        expected_improvs = np.sqrt(variance) * (dist_values * cdfs + normal_dist_values)

        return np.argmax(expected_improvs)

class upper_confidence_bound(object):
    
    @staticmethod
    def evaluate(means, variances, values, confidence_interval):
        return np.argmax(means + confidence_interval * np.sqrt(variances))

