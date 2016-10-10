import numpy as np
import matplotlib.pyplot as plt

pug = 500
dap = 500

pug_height = 6 +3* np.random.randn(pug)
dap_height = 10 + 3 * np.random.randn(dap)

plt.hist([pug_height, dap_height], stacked =True, color = ['r', 'b'])
plt.show()
