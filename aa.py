import numpy as np
from matplotlib import pyplot as plt


actions = np.loadtxt('actions.csv')
plt.plot(actions[:,3])
plt.show()