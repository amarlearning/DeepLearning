import numpy as np
import matplotlib.pyplot as plt

python = 500
javascript = 500

python_array = 28 + 4 * np.random.randn(python)
javascript_array = 24 + 4 * np.random.randn(javascript)

plt.hist([python_array, javascript_array], stacked=True, color = ['r', 'b'])
plt.show()