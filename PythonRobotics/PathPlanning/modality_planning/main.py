if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    print(ROOT_DIR)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

from components import Node, State, Modality, Terrian, WorldMap
from modality_rrt_star import ModalityRRTStar
import matplotlib.pyplot as plt
import jax
import numpy as np
import jax.numpy as jnp
import pathlib
import dill


if __name__ == '__main__':
    print('modality planning')
    
    start_pos = 0.
    end_pos = 40.
    start_node = Node(state=State(jnp.array([start_pos]), Modality.Rolling))
    goal_node = Node(state=State(jnp.array([end_pos]), Modality.Rolling))
    
    terrian_map = WorldMap(x_min=start_pos, x_max=end_pos, resolution=0.25)
    
    planner = ModalityRRTStar(start_node,
                              goal_node,
                              step_size=1.0,
                              connect_range=2.0,
                              map=terrian_map,
                              seed=40,
                              max_iter=2000,
                              goal_sample_rate=5,)
    sol = planner.plan(early_stop=False)
    xs = []
    for state in sol:
        x = jax.device_get(state.x)
        x = np.array(x)
        xs.append(x[0])
    
    modes = [int(state.mode.value) for state in sol]

    X = list(reversed(xs))
    modes = list(reversed(modes))
    print(X)
    print(modes)

    result_path = pathlib.Path('./res/case0_res.pkl')
    with open(result_path, 'wb') as f:
        dill.dump({'xs': X, 'modes': modes}, f)
    
    plt.figure()
    plt.step(X, modes)
    plt.savefig('./res/case0_res.png', dpi=1200)
    plt.show()

    