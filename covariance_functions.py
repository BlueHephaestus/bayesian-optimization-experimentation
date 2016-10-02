import numpy as np

class squared_exponential(object):

    @staticmethod
    def evaluate(x_i, x_j):
        return np.exp(-.5*np.linalg.norm(x_i - x_j)**2) 
