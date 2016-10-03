import numpy as np
import scipy.special as ss

class covariance_function(object):
    #Superclass

    def __init__(self, lengthscale, v):
        self.lengthscale = lengthscale
        self.v = v

class dot_product(covariance_function):

    def __init__(self, lengthscale, v):
        covariance_function.__init__(self, lengthscale, v)

    def evaluate(self, x_i, x_j):
        return 1*np.dot(x_i.transpose(), x_j)

class brownian_motion(covariance_function):

    def __init__(self, lengthscale, v):
        covariance_function.__init__(self, lengthscale, v)

    def evaluate(self, x_i, x_j):
        return 1*np.min(x_i, x_j)

class squared_exponential(covariance_function):

    def __init__(self, lengthscale, v):
        covariance_function.__init__(self, lengthscale, v)

    def evaluate(self, x_i, x_j):
        return np.exp((-1/(2.0*self.lengthscale))*np.linalg.norm(x_i - x_j)**2) 

class ornstein_uhlenbeck(covariance_function):

    def __init__(self, lengthscale, v):
        covariance_function.__init__(self, lengthscale, v)

    def evaluate(self, x_i, x_j):
        return np.exp(-1*np.sqrt(np.dot((x_i-x_j).transpose(), (x_i-x_j))))

class periodic1(covariance_function):

    def __init__(self, lengthscale, v):
        covariance_function.__init__(self, lengthscale, v)
    
    def evaluate(self, x_i, x_j):
        return np.exp(-1*np.sin(5*np.pi*(x_i-x_j))**2)
        
class matern(covariance_function):

    def __init__(self, lengthscale, v):
        covariance_function.__init__(self, lengthscale, v)

    def evaluate(self, x_i, x_j):
        dist = np.linalg.norm(x_i-x_j)
        return np.nan_to_num(((2**(1-self.v))/(ss.gamma(self.v))) * ((np.sqrt(2*self.v) * (dist/self.lengthscale))**self.v) * ss.kv(self.v, (np.sqrt(2*self.v) * (dist/self.lengthscale))))
