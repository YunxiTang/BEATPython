if __name__ == '__main__':
    
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import jax.numpy as jnp   
from world_map import CityMap, Block
from cost_map import CityCostMapLayer

if __name__ == '__main__':
    start = jnp.array([0., 0., 0.])
    goal = jnp.array([200., 200., 5.])

    city_map = CityMap(start=start,
                       goal=goal,
                       resolution=0.05)

    # add some obstacles
    obs1 = Block(30., 30., 120., 
                 100., 90., 
                 clr=[0.4, 0.5, 0.4])
    obs2 = Block(30., 20., 180., 
                 120., 50., 
                 clr=[0.5, 0.5, 0.6])
    obs3 = Block(40., 40., 90., 
                 30., 70., 
                 clr=[0.3, 0.3, 0.4])
    city_map.add_obstacle(obs1)
    city_map.add_obstacle(obs2)
    city_map.add_obstacle(obs3)
    city_map.add_obstacle(Block(20., 30., 70., 
                                70., 160., clr=[0.3, 0.3, 0.4]))
    city_map.add_obstacle(Block(20., 30., 180., 
                                150., 140., clr=[0.3, 0.3, 0.4]))
    city_map.finalize()

    city_cost_layer = CityCostMapLayer(city_map)