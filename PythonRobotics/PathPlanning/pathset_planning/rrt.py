"""RRT Implementation"""
import jax.numpy as jnp
import jax.random as random
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import time


class SimpleNode:
    def __init__(self, state, path_len):
        self._state = state
        self._path_len = path_len

    def set_state(self, state):
        self._state = state

    @property
    def state(self):
        return self._state
    
    @property
    def path_len(self):
        return self._path_len
    
    def __repr__(self) -> str:
        return 'Simple Node' + f' at {self._state} obstacles' \
                            + f' with {self._path_len}'


class Node:
    """
        Node in RRT/RRT* algorithm
    """
    def __init__(self, state):
        self._state = state
        self._parent = None
        
        # For usage in RRR*
        self._cost = -100. * 200.
        self.min_dist = 200.
        
    def set_parent(self, node):
        self._parent = copy.deepcopy(node)
        
    def set_cost(self, cost):
        self._cost = cost

    def reset_parent(self, node):
        self._parent = copy.deepcopy(node)

    def reset_min_dist(self, min_dist):
        self.min_dist = min_dist

    def __eq__(self, other):
        eq = False
        if jnp.linalg.norm(self.state - other.state) < 1e-3:
            eq = True
        return eq
        
    @property
    def state(self):
        return self._state
    
    @property
    def parent(self):
        return self._parent
    
    @property
    def cost(self):
        return self._cost
        
        
class RRT:
    """
        RRT Implementation
    """
    def __init__(self, 
                 start_config: jnp.ndarray,
                 goal_config: jnp.ndarray,
                 map,
                 step_size: float = 0.01,
                 goal_sample_rate: int = 50,
                 max_iter: int = 500,
                 seed: int = 0
                 ):
        self._start = Node(start_config)
        self._goal = Node(goal_config)
        
        self._map = map
        self._resolution = map._resolution

        self._node_list = []
        
        self._step_size = step_size if step_size > self._resolution else 2.0 * self._resolution
        self._max_iter = max_iter
        self._goal_sample_rate = goal_sample_rate
        
        self._rng_key = random.PRNGKey(seed=seed)
        

    def _get_random_node(self):
        '''
            sample a collision-free random node
        '''
        self._rng_key, rng_key = random.split(self._rng_key, 2)
        
        if random.randint(rng_key, (1,), 0, 100)[0] > self._goal_sample_rate:
            rand_state = self._map.sample_free_pos()
        else:
            rand_state = self._map.sample_free_pos(toward_goal=True)
        
        return Node(rand_state)
    

    @staticmethod
    def _compute_node_distance(node1: Node, node2: Node):
        return jnp.linalg.norm(node1._state - node2._state)
    
    
    def _get_nearest_node(self, rand_node):
        dlist = [RRT._compute_node_distance(rand_node, node) for node in self._node_list]
        dlist = jnp.array(dlist)
        min_idx = jnp.argmin(dlist)
        min_dist = dlist[min_idx]
        return min_idx, min_dist
        
        
    def _steer(self, from_node: Node, to_node: Node, step_size: float) -> Node:
        dist = jnp.linalg.norm(to_node.state - from_node.state)
        if dist <= step_size:
            new_node_state = to_node.state
        else:
            new_node_state = from_node.state + step_size * (to_node.state - from_node.state) / dist
        new_node = Node(new_node_state)
        new_node.set_parent(from_node)
        return new_node
    
    
    def _check_edge_collision(self, node1: Node, node2: Node) -> bool:
        return self._map.check_line_collision(node1.state, node2.state)
    
    
    def _check_node_collision(self, node: Node):
        return self._map.check_pos_collision(node.state)
        
        
    def plan(self, verbose=True, animation=True):
        # initialize the tree 
        self._node_list = [self._start]
        
        for i in range(self._max_iter):
            # sample a random valid node
            rand_node = self._get_random_node()
            
            # search for the nearest tree node
            nearest_ind, _ = self._get_nearest_node(rand_node)
            nearest_node = self._node_list[nearest_ind]
            
            # steer
            new_node = self._steer(nearest_node, rand_node, self._step_size)
            
            # check edge collision
            if not self._check_node_collision(new_node):
                if not self._check_edge_collision(nearest_node, new_node):
                    self._node_list.append(new_node)
                
            if animation and i % 5 == 0:
                self._draw_graph(rand_node)
                
            if self._calc_dist_to_goal(self._node_list[-1]) <= self._step_size:
                final_node = Node(self._goal.state)
                final_node.set_parent(self._node_list[-1])
                self._node_list.append(final_node)
                sol = self._generate_final_course()
                print(f'Find a feasible path with {len(sol)} nodes!')
                return sol
            else:
                if verbose and i % 10 == 0:
                    print(f"Iter: {i} || No. of Tree Nodes: {len(self._node_list)}")
        print('Failed to find a feasible path...')
        return None
    

    def _draw_graph(self, node: Node):
        pass
    

    def _calc_dist_to_goal(self, node: Node):
        return RRT._compute_node_distance(node, self._goal)
    

    def _generate_final_course(self):
        path = [self._goal.state]
        node = self._node_list[len(self._node_list) - 1]
        while node.parent is not None:
            path.append(node.state)
            node = node.parent
        path.append(node.state)
        return path
            
            
if __name__ == '__main__':
    
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

    from world_map import TwoDimMap
    from utils import plot_circle


    world_map = TwoDimMap([0., 2., 0., 2.], resolution=0.02)
    start = jnp.array([0.0, 0.])
    goal = jnp.array([2.0, 2.0])
    
    world_map.update_start(start)
    world_map.update_goal(goal)
    
    rng_key = random.PRNGKey(seed=4)

    for i in range(100):
        rng_key, rng_key_x, rng_key_y, rng_key_r = random.split(rng_key, 4)
        x = random.uniform(rng_key_x, shape=(1,), minval=0.1, maxval=1.75)
        y = random.uniform(rng_key_y, shape=(1,), minval=0.1, maxval=1.75)
        r = random.uniform(rng_key_r, shape=(1,), minval=0.05, maxval=0.15)
        obs = (x[0], y[0], r[0])
        world_map.add_obstacle(obs)
    
    if not world_map.check_pos_collision(start) and not world_map.check_pos_collision(goal):
        rrt = RRT(
            start_config=start,
            goal_config=goal,
            map=world_map,
            step_size=0.1,
            goal_sample_rate=0,
            seed=250,
            max_iter=1500
        )
        path_solution = rrt.plan()
    else:
        path_solution = None
    
    if path_solution is not None:
        print(f'Path Length: {len(path_solution)}')
        path = jnp.array(path_solution)
        
        sns.set_theme('notebook')
        for obs in world_map._obstacle:
            plot_circle(obs[0], obs[1], obs[2])
            
        for node in rrt._node_list:
            plt.scatter(node.state[0], node.state[1], c='k', s=0.5)
            if node.parent != None:
                plt.plot([node.state[0], node.parent.state[0]],
                          [node.state[1], node.parent.state[1]], 
                          'k-.', linewidth=0.5)
        
        for i in range(len(path_solution)-1):
            plt.scatter(path[i,0], path[i,1])
            plt.plot(path[i:i+2,0], path[i:i+2,1], linewidth=2.5)
        
        plt.scatter(path[len(path_solution)-1,0], path[len(path_solution)-1,1]) 
        
        plt.scatter(start[0], start[1], marker='*', linewidths=2)
        plt.scatter(goal[0], goal[1], marker='*', linewidths=2) 
        plt.axis('equal')
        plt.show()        
    
    
    
        
    