"""RRT* Implementation"""
import jax.numpy as jnp
import jax.random as random
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple
import copy
from typing import List


if __name__ == '__main__':
    
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)
    from world_map import TwoDimMap

from rrt import Node, RRT
from utils import plot_circle
from cost_map import EuclideanCostMapLayer, CostMap


class RRTStar(RRT):
    """
        RRT star implementation
    """
    def __init__(self, connect_range, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._connect_range = connect_range
        self._potential_final_node = self._start
        self._costMap = CostMap()
        self._costMap.add_cost_layer(EuclideanCostMapLayer())

        
        self._d = self._map._dim
        self._gamma_rrt_star = 2. * (1. + 1. / self._d)**(1. / self._d) * (1.0)**(1. / self._d)
        self._yita = self._step_size

    def _card(self):
        return len(self._node_list)
        
    def _compute_connective_range(self):
        return jnp.minimum(self._gamma_rrt_star * (jnp.log(self._card())/ self._card())**(1./self._d), self._connect_range)
        
    def _find_near_node_idx(self, new_node: Node):
        """find all the nodes in some range

        Args:
            new_node (Node): node
        """
        node_dists = [RRT._compute_node_distance(new_node, node) for node in self._node_list]
        near_idxs = [node_dists.index(i) for i in node_dists if i <= self._compute_connective_range()]
        return near_idxs
    

    def _choose_parent(self, near_node_idxs: List[int], new_node: Node):
        """
            computes the cheapest node to new_node contained in the list
            and reset the node as new_node.parent.
        """
        costs = [self._node_list[idx].cost + self._costMap.compute_edge_cost(self._node_list[idx], new_node) for idx in near_node_idxs]
        
        if costs:
            sorted_idx = sorted(range(len(costs)), key=lambda k: costs[k])
            for idx in sorted_idx:
                if not self._check_edge_collision(self._node_list[near_node_idxs[idx]], new_node):
                    new_node.reset_parent(self._node_list[near_node_idxs[idx]])
                    break

        new_node_cost = new_node.parent.cost + self._costMap.compute_edge_cost(new_node.parent, new_node)
        new_node.set_cost(new_node_cost)
        return new_node
    
    
    def _rewire(self, new_node: Node, near_node_idxs: List[int]):
        """
            For each node in near_inds, this will check if it is cheaper to
            arrive to them from new_node.
            In such a case, this will re-assign the parent of the nodes in
            near_inds to new_node.
        """
        for idx in near_node_idxs:
            node = self._node_list[idx]
            if new_node.parent != node:
                updated_cost = new_node.cost + self._costMap.compute_edge_cost(new_node, node)
                if updated_cost < node.cost and (not self._check_edge_collision(node, new_node)):
                    node.reset_parent(new_node)
                    node.set_cost(updated_cost)


    def _generate_final_course(self):
        path = [self._goal.state]
        node = self._potential_final_node
        while node.parent is not None:
            path.append(node.state)
            node = node.parent
        path.append(node.state)
        return path

        
    def plan(self, verbose=True, animation=True, early_stop=True):
        self._node_list.append(self._start)
        
        for i in range(self._max_iter):
            # sample a random node
            rand_node = self._get_random_node()

            # find the nearest node in the tree
            nearest_ind, _ = self._get_nearest_node(rand_node)
            nearest_node = self._node_list[nearest_ind]
            
            # get new node candidate
            new_node = self._steer(nearest_node, rand_node, self._step_size)
            
            if not self._check_node_collision(new_node):
                # reset node parent
                near_idxs = self._find_near_node_idx(new_node)
                new_node = self._choose_parent(near_idxs, new_node)
                self._node_list.append(new_node)

                # rewire the tree
                self._rewire(new_node, near_idxs)

                if self._calc_dist_to_goal(new_node) <= self._calc_dist_to_goal(self._potential_final_node):
                    self._potential_final_node = copy.deepcopy(new_node)

                # assemble solution due to early stop
                if self._calc_dist_to_goal(self._potential_final_node) <= self._step_size \
                    and early_stop:
                    self._goal.set_parent(self._potential_final_node)
                    self._goal.set_cost(self._potential_final_node.cost \
                            + self._compute_node_distance(self._potential_final_node, self._goal))
                    sol = self._generate_final_course()
                    print(f'Find a path with {len(sol)} nodes due to early stop at iter {i}. Goal cost: {self._goal.cost}')
                    return sol

            # verbose (print information)
            if verbose and i % 10 == 0:
                print(f"Iter: {i} || No. of Tree Nodes: {len(self._node_list)}")
                print(self._gamma_rrt_star * (jnp.log(self._card())/ self._card())**(1./self._d), self._compute_connective_range())
        
        # assemble solution due to max iter
        if not self._check_edge_collision(self._potential_final_node, self._goal):
            self._goal.set_parent(self._potential_final_node)
            self._goal.set_cost(self._potential_final_node.cost \
                                + self._costMap.compute_edge_cost(self._potential_final_node, self._goal))
            sol = self._generate_final_course()
            print(f'Find a path with {len(sol)} nodes due to max_iter. Goal cost: {self._goal.cost}')
        else:
            print(f'Failed to find a feasible path with max_iter {self._max_iter}')
            sol = None
        return sol
        

if __name__ == '__main__':
    
    import sys
    import os
    import pathlib
    import scipy.interpolate as interpolate
    import numpy as np

    ROOT_DIR = str(pathlib.Path(__file__).parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

    from world_map import TwoDimMap

    world_map = TwoDimMap([0., 2., 0., 2.], resolution=0.01)
    start = jnp.array([0.0, 0.])
    goal = jnp.array([2.0, 2.0])    
    world_map.update_start(start)
    world_map.update_goal(goal)
    
    rng_key = random.PRNGKey(seed=88)

    for i in range(100):
        rng_key, rng_key_x, rng_key_y, rng_key_r = random.split(rng_key, 4)
        x = random.uniform(rng_key_x, shape=(1,), minval=0.1, maxval=1.75)
        y = random.uniform(rng_key_y, shape=(1,), minval=0.1, maxval=1.75)
        r = random.uniform(rng_key_r, shape=(1,), minval=0.05, maxval=0.1)
        obs = (x[0], y[0], r[0])
        world_map.add_obstacle(obs)
    
    if not world_map.check_pos_collision(start) and not world_map.check_pos_collision(goal):
        planner = RRTStar(
            connect_range=0.1,
            start_config=start,
            goal_config=goal,
            map=world_map,
            step_size=0.05,
            goal_sample_rate=0,
            seed=50,
            max_iter=4500
        )
        path_solution = planner.plan(early_stop=True)
    else:
        path_solution = None
    
    if path_solution is not None:
        print(f'Path Length: {len(path_solution)}')
        path = jnp.array(path_solution)
        
        sns.set()
        for obs in world_map._obstacle:
            plot_circle(obs[0], obs[1], obs[2])
            
        for node in planner._node_list:
            plt.scatter(node.state[0], node.state[1], c='k', s=0.5)
            if node.parent is not None:
                plt.plot([node.state[0], node.parent.state[0]],
                          [node.state[1], node.parent.state[1]], 
                          'k-.', linewidth=0.5)
        
        for i in range(len(path_solution)-1):
            plt.scatter(path[i,0], path[i,1])
            plt.plot(path[i:i+2,0], path[i:i+2,1], linewidth=2.5)

        # path_solution.reverse()
        # xs = [path[0] for path in path_solution]
        # ys = [path[1] for path in path_solution]
        # tck, u = interpolate.splprep([xs, ys], k=5, s=3)
        # new_points = interpolate.splev(u, tck)
        # plt.plot(new_points[0], new_points[1], 'r-', linewidth=2.5)

        plt.scatter(path[len(path_solution)-1,0], path[len(path_solution)-1,1]) 
        
        plt.scatter(start[0], start[1], marker='*', linewidths=2)
        plt.scatter(goal[0], goal[1], marker='*', linewidths=2) 
        plt.axis('equal')
        plt.show()