if __name__ == '__main__':
    import os
    import sys
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    
    sys.path.append('/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning')
    jax.config.update("jax_enable_x64", True)     # enable fp64
    jax.config.update('jax_platform_name', 'cpu') # use the CPU instead of GPU

    from st_dlo_planning.temporal_config_opt.opt_solver import DloOptProblem, TcDloSolver
    from st_dlo_planning.utils import PathSet

    waypoints1 = jnp.array([[0.28, 0.1], 
                            [0.4, 0.45], 
                            [0.35, 1.0], 
                            [0.25, 1.42]]) + jnp.array([[0.03, 0.0]])
    waypoints2 = jnp.array([[0.53, 0.0], 
                            [0.52, 0.5], 
                            [0.48, 1.0], 
                            [0.47, 1.40]])
    waypoints3 = jnp.array([[0.75, 0.0], 
                            [0.6, 0.55], 
                            [0.6, 1.0], 
                            [0.68, 1.5]])
    waypoints4 = jnp.array([[0.75, 0.0], 
                            [0.6, 0.55], 
                            [0.6, 1.0], 
                            [0.68, 1.5]]) + jnp.array([[0.12, 0.1]])
    waypoints5 = jnp.array([[0.75, 0.0], 
                            [0.6, 0.55], 
                            [0.6, 1.0], 
                            [0.68, 1.5]]) + jnp.array([[0.22, 0.15]])
    waypoints6 = jnp.array([[0.75, 0.0], 
                            [0.6, 0.55], 
                            [0.6, 1.0], 
                            [0.68, 1.5]]) + jnp.array([[0.32, 0.2]])
    
    all_path = [waypoints1, waypoints2, waypoints3, waypoints4, waypoints5, waypoints6]

    pathset = PathSet(all_path, T=100, seg_len=0.2)

    solver = TcDloSolver(pathset=pathset, k1=1.0, k2=5.0, max_iter=800)
    
    opt_sigmas, info = solver.solve()

    solution = jnp.reshape(opt_sigmas, (pathset.T + 1, pathset.num_path))


    plt.plot(np.linspace(0.0, 1.0, pathset.T+1, endpoint=True), solution)
    plt.plot(np.linspace(0.0, 1.0, pathset.T+1, endpoint=True), np.linspace(0.0, 1.0, pathset.T+1), 'r-.')
    plt.axis('equal')
    plt.show()

    clrs = np.linspace(0.0, 1.0, pathset.T+1)
    rever_clrs = np.flip(clrs)
    import matplotlib.animation as animation
    fig, ax = plt.subplots()

    artists = []
    for i in range(0, pathset.T+1, 1):
        dlo_shape = pathset.query_dlo_shape(solution[i])
        container1 = ax.plot(dlo_shape[:, 0], dlo_shape[:, 1], color=[clrs[i], rever_clrs[i], clrs[i]], linewidth=3)
        # container2 = ax.scatter(dlo_shape[:, 0], dlo_shape[:, 1], color='k')
        artists.append(container1)
        # artists.append(container2)
    ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=100)
    pathset.vis_all_path()
    ani.save(filename="./optimized.gif", writer="pillow")

    # fig1, ax1 = plt.subplots()
    # solution = np.linspace(0.0, 1.0, pathset.T+1, endpoint=True)
    # solution = np.repeat(solution, repeats=pathset.num_path, axis=0).reshape(-1, pathset.num_path,)
    # artists = []
    # for i in range(0, pathset.T+1, 1):
    #     print(solution[i])
    #     dlo_shape = pathset.query_dlo_shape(solution[i])
    #     container1 = ax1.plot(dlo_shape[:, 0], dlo_shape[:, 1], color=[clrs[i], rever_clrs[i], clrs[i]], linewidth=3)
    #     # container2 = ax.scatter(dlo_shape[:, 0], dlo_shape[:, 1], color='k', label="Intermediate Nodes")
    #     artists.append(container1)
    #     # artists.append(container2)
    # ani = animation.ArtistAnimation(fig=fig1, artists=artists, interval=100)
    # pathset.vis_all_path()
    # ani.save(filename="./sim.gif", writer="pillow")
    