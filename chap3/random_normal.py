import numpy as np
import matplotlib.pyplot as plt

mean = 0
std_dev = 1
num_samples = 1000

samples = np.random.normal(mean, std_dev, num_samples)

plt.hist(samples, bins=30, density=True, alpha=0.5, color='b')

plt.show()
