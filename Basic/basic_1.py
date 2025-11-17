x = [1.0, 2.0, 3.0]
y = {"a": 665, "b": 354}

for ele in list(zip(x, y.items())):
    print(ele)


import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import os

lb = 10
ub = 45
dist = ub - lb
x = np.linspace(lb, ub, 500)
y1 = norm.pdf(x, 17.0, 1.3)
y2 = norm.pdf(x, 32.0, 2.3)
y = 0.3 * y1 + 0.7 * y2
y = y / (np.sum(y) * (dist / 500))

print(np.sum(y) * (dist / 500))

sns.set()
fig, ax = plt.subplots(1, 1)
# ax.plot(x, y1, 'r--', lw=3, alpha=0.6, label='norm pdf')
# ax.plot(x, y2, 'b--', lw=3, alpha=0.6, label='norm pdf')
ax.plot(x, y, "k-", lw=3, alpha=0.7, label="multimodal feasible path distribution")
ax.xaxis.set_tick_params("minor")
ax.yaxis.set_tick_params("minor")
fig.savefig(os.path.join(os.path.dirname(__file__), "multi_modal.png"), dpi=1200)
plt.show()
