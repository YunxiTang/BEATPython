"""RRT* Implementation"""
import jax.numpy as jnp
import jax.random as random
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple
import copy, math
from rrt import WorldMap, Node, RRT, plot_circle
from typing import List


class RRTStar(RRT):
    def __init__(self, connect_range, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._connect_range = connect_range
        
    def _find_near_node_idx(self, new_node: Node):
        """find all the nodes in some range

        Args:
            new_node (Node): node
        """
        node_dists = [RRT._compute_node_distance(new_node, node) for node in self._node_list]
        near_idx = [node_dists.index(i) for i in node_dists if i <= self._connect_range**2]
        return near_idx
    
    def _choose_parent(self, near_node_idxs: List[int], new_node: Node):
        """
            Computes the cheapest node to new_node contained in the list
            and set such node as the parent of new_node.
        """
        costs = [RRT._compute_node_distance(self._node_list[idx], new_node) + self._node_list[idx].cost for idx in near_node_idxs]
        if costs:
            costs = jnp.array(costs)
            min_idx = jnp.argmin(costs)
            if not self._check_edge_collision(self._node_list[near_node_idxs[min_idx]], new_node):
                updated_parent_node = copy.deepcopy(new_node)
                updated_parent_node.set_parent(self._node_list[near_node_idxs[min_idx]])
                return updated_parent_node
            else:
                return new_node
        else:
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
                updated_cost = new_node.cost + RRT._compute_node_distance(node, new_node)
                if updated_cost < node.cost and (not self._check_edge_collision(node, new_node)):
                    node.set_parent(new_node)
                    node.set_cost(updated_cost)
        
    def plan(self, animation=True, verbose=False):
        self._node_list.append(self._start)
        
        for i in range(self._max_iter):
            # sample a random node
            rand_node = self._get_random_node()

            # find the nearest node in the tree
            nearest_ind, _ = self._get_nearest_node(rand_node)
            nearest_node = self._node_list[nearest_ind]
            
            # get new node candidate
            new_node = self._steer(nearest_node, rand_node, self._step_size)
            
            if not self._check_edge_collision(nearest_node, new_node):
                near_idxs = self._find_near_node_idx(new_node)
                new_node = self._choose_parent(near_idxs, new_node)
                new_node_cost = new_node.parent.cost + self._compute_node_distance(new_node, new_node.parent)
                new_node.set_cost(new_node_cost)
                self._node_list.append(new_node)
                self._rewire(new_node, near_idxs)
                
                if self._calc_dist_to_goal(self._node_list[-1]) <= self._step_size * 1.0:
                    final_node = copy.deepcopy(self._goal)
                    final_node.set_parent(new_node)
                    self._node_list.append(new_node)
                    print('Find a feasible path.')
                    return self._generate_final_course()
            print(f"Iter: {i} || No. Nodes: {len(self._node_list)}")
        return self._generate_final_course()
        
if __name__ == '__main__':
    
    world_map = WorldMap([0., 2., 0., 2.])
    start = jnp.array([0., 0.])
    goal = jnp.array([0.75, 1.0])
    
    world_map.update_start(start)
    world_map.update_goal(goal)
    
    obs1 = (0.5, 0.4, 0.25)
    obs2 = (0.4, 0.8, 0.2)
    obs3 = (0.8, 0.8, 0.15)
    obs4 = (1.0, 0.4, 0.15)
    world_map.add_obstacle(obs1)
    world_map.add_obstacle(obs2)
    world_map.add_obstacle(obs3)
    world_map.add_obstacle(obs4)
    
    planner = RRTStar(0.2, start, goal, step_size=0.1, max_iter=1000, map=world_map)
    
    path_solution = planner.plan()
    
    if path_solution is not None:
        path = jnp.array(path_solution)
        
        sns.set()
        for obs in world_map._obstacle:
            plot_circle(obs[0], obs[1], obs[2])
            
        for node in planner._node_list:
            plt.scatter(node.state[0], node.state[1], c='k')
            if node.parent != None:
                plt.plot([node.state[0], node.parent.state[0]], [node.state[1], node.parent.state[1]], 'k-.')
        
        for i in range(len(path_solution)-1):
            plt.scatter(path[i,0], path[i,1])
            plt.plot(path[i:i+2,0], path[i:i+2,1])
        
        plt.scatter(path[len(path_solution)-1,0], path[len(path_solution)-1,1]) 
        
        plt.scatter(start[0], start[1], marker='*', linewidths=5)
        plt.scatter(goal[0], goal[1], marker='*', linewidths=5) 
        plt.axis('equal')
        plt.show()    
    
    