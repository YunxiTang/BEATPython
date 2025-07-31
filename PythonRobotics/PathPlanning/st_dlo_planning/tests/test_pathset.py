if __name__ == "__main__":
    import sys
    import os
    import pathlib
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from pprint import pprint
    from omegaconf import OmegaConf

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    print(ROOT_DIR)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

    from st_dlo_planning.utils import PathSet, compute_enery
    import jax.numpy as jnp
    import jax

    fig = plt.figure()
    ax = fig.add_subplot(111)

    jax.config.update("jax_enable_x64", True)  # enable fp64
    jax.config.update("jax_platform_name", "cpu")  # use the CPU instead of GPU

    waypoints1 = jnp.array([[0.25, 0.0], [0.35, 0.5], [0.35, 1.0], [0.25, 1.5]])
    print(waypoints1.devices())
    waypoints2 = jnp.array([[0.5, 0.0], [0.5, 0.5], [0.5, 1.0], [0.5, 1.5]])
    waypoints3 = jnp.array([[0.75, 0.0], [0.65, 0.5], [0.65, 1.0], [0.75, 1.5]])

    all_path = [waypoints1, waypoints2, waypoints3]

    pathset = PathSet(all_path, T=50, seg_len=0.25)

    sigma1 = jnp.array([0.5, 0.45, 0.3])
    sigma2 = jnp.array([0.1, 0.05, 0.1])
    dlo_shape_sample = pathset.query_dlo_shape(sigma1)
    dlo_shape_sample2 = pathset.query_dlo_shape(sigma2)
    plt.plot(
        dlo_shape_sample[:, 0],
        dlo_shape_sample[:, 1],
        "m-",
        linewidth=3,
        label="DLO Shape 1",
    )
    plt.plot(
        dlo_shape_sample2[:, 0],
        dlo_shape_sample2[:, 1],
        "r-",
        linewidth=3,
        label="DLO Shape 2",
    )

    dlo_shape_sample0 = pathset.query_dlo_shape(jnp.array([0.0, 0.0, 0.0]))
    plt.plot(
        dlo_shape_sample0[:, 0],
        dlo_shape_sample0[:, 1],
        "b-",
        linewidth=3,
        label="DLO Shape 0",
    )
    u = compute_enery(dlo_shape_sample0, k1=1.0, k2=0.5, segment_len=pathset.seg_len)
    print(u)

    dlo_shape_sample2 = pathset.query_dlo_shape(jnp.array([1.0, 1.0, 1.0]))
    plt.plot(
        dlo_shape_sample2[:, 0],
        dlo_shape_sample2[:, 1],
        "g-",
        linewidth=3,
        label="DLO Shape 4",
    )
    u = compute_enery(dlo_shape_sample2, k1=1.0, k2=0.5, segment_len=pathset.seg_len)
    print(u)

    for i in range(20000):
        dlo_shape_sample2 = pathset.query_dlo_shape(jnp.array([0.5, 0.5, 0.5]))

    u = compute_enery(dlo_shape_sample2, k1=1.0, k2=0.5, segment_len=pathset.seg_len)
    print(u)
    plt.plot(
        dlo_shape_sample2[:, 0],
        dlo_shape_sample2[:, 1],
        "k-",
        linewidth=3,
        label="DLO Shape 4",
    )

    pathset.vis_all_path(ax)
    plt.axis("equal")
    plt.show()
