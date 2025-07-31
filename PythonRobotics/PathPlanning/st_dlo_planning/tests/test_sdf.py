if __name__ == "__main__":
    import os
    import sys
    import time
    import matplotlib.pyplot as plt

    sys.path.append("/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning")

    from st_dlo_planning.utils import PathSet, compute_enery
    import jax.numpy as jnp
    import jax
    import numpy as np
    from omegaconf import OmegaConf

    from st_dlo_planning.utils.world_map import Block, WorldMap, MapCfg

    map_case = "map_case0"

    cfg_path = f"/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/st_dlo_planning/envs/map_cfg/{map_case}.yaml"
    map_cfg_file = OmegaConf.load(cfg_path)

    map_cfg = MapCfg(
        resolution=map_cfg_file.workspace.resolution,
        map_xmin=map_cfg_file.workspace.map_xmin,
        map_xmax=map_cfg_file.workspace.map_xmax,
        map_ymin=map_cfg_file.workspace.map_ymin,
        map_ymax=map_cfg_file.workspace.map_ymax,
        map_zmin=map_cfg_file.workspace.map_zmin,
        map_zmax=map_cfg_file.workspace.map_zmax,
        robot_size=map_cfg_file.workspace.robot_size,
        dim=3,
    )

    world_map = WorldMap(map_cfg)
    # ============== add some obstacles =========================
    size_z = map_cfg_file.workspace.map_zmax
    obstacles = map_cfg_file.obstacle_info.obstacles
    i = 0
    for obstacle in obstacles:
        world_map.add_obstacle(
            Block(
                obstacle[0],
                obstacle[1],
                size_z,
                obstacle[2],
                obstacle[3],
                angle=obstacle[4] * np.pi,
                clr=[0.3 + 0.01 * i, 0.5, 0.4],
            )
        )
        i += 1
    world_map.finalize()
    _, ax = world_map.visualize_passage(full_passage=False)

    p = np.array([0.2, 0.3])

    for obs in world_map._obstacle:
        ax.scatter(p[0], p[1], c="r")
        sdf_val, vrtxs = obs.get_2d_sdf_val(p)
        print(sdf_val)
        for vrtx in vrtxs:
            ax.scatter(vrtx[0], vrtx[1], c="b")
    plt.axis("equal")
    plt.show()
