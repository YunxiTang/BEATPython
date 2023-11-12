from scipy import spatial
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

data = np.random.normal(
    loc=0.2,
    scale=1.2,
    size=(1500, 3)
    )

bunny_pcd = o3d.io.read_point_cloud('../../bunny.pcd', print_progress=True)
# o3d.visualization.draw_geometries([bunny_pcd,], 
#                                   # front=[0.4,0.2, 0.8]
# )
data = np.asarray(bunny_pcd.points)
tree = spatial.KDTree(data)


res = tree.query_ball_point(data[5], r=0.05)

for point_idx in res:
    print(data[point_idx])


plt.figure(0)
plt.scatter(data[:,0], data[:,1],data[:,2])
plt.show()
