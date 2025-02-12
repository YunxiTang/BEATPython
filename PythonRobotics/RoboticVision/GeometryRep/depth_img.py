from pydrake.perception import DepthImageToPointCloud
import open3d as o3d


if __name__ == '__main__':
    converter = DepthImageToPointCloud()