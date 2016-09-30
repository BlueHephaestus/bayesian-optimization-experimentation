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
4. Generate n test means(u* s) and test variances(c* s) across domain
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
    return np.exp(-(x-2)**2) + np.exp(-((x-6)**2)/10.0) + (1.0/(x**2 + 1))

#Number of evaluated input points / level of detail
n = 1000

#Number of bbf evaluations allowed to perform before ending optimization
bbf_evaluation_n = 10

#Initialize ranges for each parameter into a resulting matrix
hps = [HyperParameter(0, 10), HyperParameter(0.1, 0.8)]

"""END TUNABLE PARAMETERS"""

#Initialize multivariate(if len(hps) > 1) input domain
domain = [np.arange(hp.min, hp.max, ((hp.max-hp.min)/float(n))) for hp in hps]

#Initialize known function outputs vector
bbf_evaluations = np.zeros(shape=(iterations))

#Temporary domain so we can make sure to get two DIFFERENT values 
shuffled_domain = domain
for hp_range in shuffled_domain:
    np.random.shuffle(hp_range) 

"""
At this point our shuffled domain is of the form [[a1, a2, ..., an], [b1, b2, ..., bn]]
    where a and b are different parameters, or different input dimensions.
    When we do param[:2] we change this to be only the first two shuffled values:
        [[a1, a2], [b1, b2]]
    So that we can get two random input values without having the risk of them being the same.
    Then, we can assign them both at the same time by transposing this matrix, so it becomes:
        [[a1, b1], [a2, b2]]
    Then we assign x1 to the first row, and x2 to the second.
"""

x1, x2 = np.array([param[:2] for param in shuffled_domain]).transpose()

#Now that we have our two random input vectors, evaluate them and store them
#Modify the bbf function when you make this more complicated with input to a bot
bbf_evaluations[0], bbf_evaluations[1] = bbf(x1), bbf(x2)

for bbf_evaluation_i in range(2, bbf_evaluation_n):



