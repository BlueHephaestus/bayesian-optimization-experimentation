#TODO
#Add in the speed improvements irc guy suggested

import sys

import numpy as np
import matplotlib.pyplot as plt#For plotting end results

#Mean and stddev for distribution we draw weights from
mean = 0
stddev = 1

#Value ranges
x = np.arange(0, 1, 0.005)
n = len(x)

def apply_weight(U, S):

    #Draw n weights from distribution, weighted by mean and stddev
    w = mean + stddev * np.random.randn(n)

    #Get a linear combination of our weights and eigenvalues S via S * w
    #Then convert our U matrix into a vector in our np.dot(U, S*w)
    return np.dot(U, S*w)

#Covariance functions
#k = lambda x, y: x*y
#k = lambda x, y: np.min(x, y)
k = lambda x, y: np.exp(-100*(x-y)**2)
#k = lambda x, y: np.exp(-1*np.sqrt((x-y)**2))
#k = lambda x, y: np.exp(-np.sin(5*np.pi*(x-y))**2)
#k = lambda x, y: np.exp(-100*np.minimum(np.abs(x-y), np.abs(x+y))**2)

#Covariance Matrix
C = np.zeros(shape=(n, n))

#Populate covariance matrix
for i in range(1, n):
    for j in range(1, n):
        C[i, j] = k(x[i], x[j])

#Get the singular value decomposition of our covariance matrix
#Note: V = U transpose, S = eigenvalues of C
[U, S, V] = np.linalg.svd(C)

"""
My understanding is that:
    S is the eigenvalues of our covariance matrix C,
    which is symmetric because of our covariance function.
    So when we get the eigenvalues, we already have to the power 2
    higher values than we need, because it's taking into account the same
    values in the matrix twice, since symmetric.

    So when we do sqrt(S), we are removing that extra stuff to get it just right.
"""

S = np.sqrt(S)

for i in range(100):
    Z = apply_weight(U, S)

    #plt.plot(x, Z, 'bo')
    #plt.plot(x, Z, 'b')
    plt.plot(x, Z)

#plt.axis([0, 1, -2, 2])

plt.show()
