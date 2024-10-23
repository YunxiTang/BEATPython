import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['axes.unicode_minus'] = False

plt.style.use('ggplot')


feature = ['obj1', 'obj2', 'obj3',
           'obj4', 'obj5', 'obj6']
values = [70, 61, 85, 54, 33, 76]

N = len(feature)

angles = np.linspace(0, 2*np.pi, N, endpoint=False)

values = np.concatenate((values,[values[0]]))
angles = np.concatenate((angles,[angles[0]]))
feature.append( feature[0] )

fig=plt.figure()
ax = fig.add_subplot(111, polar=True)

ax.plot(angles, values, 'o-', color='b', linewidth=2)
ax.fill(angles, values, 'b', alpha=0.25)

values = [60, 71, 88, 73, 45, 86]
values = np.concatenate((values,[values[0]]))
ax.plot(angles, values, 'o-', color='k', linewidth=2)
ax.fill(angles, values, 'k', alpha=0.25)

ax.set_thetagrids(angles * 180/np.pi, feature)
ax.set_ylim(0, 100)
ax.grid(True)
plt.savefig('./lidar_map.jpg', dpi=1500)
plt.show()
