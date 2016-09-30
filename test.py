import numpy as np
import matplotlib.pyplot as plt

mean = 0
variance = 1

norm_dist = lambda x, mean, variance: (np.exp(-((x-mean)**2)/(2*variance))) / (np.sqrt(2*variance*np.pi))

x = np.arange(-5, 5, .1)
y = [norm_dist(x_i, mean, variance) for x_i in x]

plt.plot(x, y)
plt.show()

