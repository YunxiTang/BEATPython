"""RRT* Implementation"""
import jax.numpy as jnp
import jax.random as random
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple
import copy
import pickle
from typing import List


if __name__ == '__main__':
    
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)
    from world_map import TwoDimMap

from rrt import Node, RRT
from utils import plot_circle
from pathset_planning.cost_map import EuclideanCostMapLayer, CostMap


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
    

    def add_cost_layer(self, cost_layer):
        self._costMap.add_cost_layer(cost_layer)

        
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

                # if self._calc_dist_to_goal(new_node) <= self._calc_dist_to_goal(self._potential_final_node):
                #     self._potential_final_node = copy.deepcopy(new_node)

                # # assemble solution due to early stop
                # if self._calc_dist_to_goal(self._potential_final_node) <= self._step_size \
                #     and early_stop:
                #     self._goal.set_parent(self._potential_final_node)
                #     self._goal.set_cost(self._potential_final_node.cost \
                #             + self._compute_node_distance(self._potential_final_node, self._goal))
                #     sol = self._generate_final_course()
                #     print(f'Find a path with {len(sol)} nodes due to early stop at iter {i}. Goal cost: {self._goal.cost}')
                #     return sol

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
        

class CityRRTStar(RRT):
    """
        RRT star implementation
    """
    def __init__(self, connect_range, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._connect_range = connect_range
        
        self._costMap = CostMap()
        self._costMap.add_cost_layer(EuclideanCostMapLayer())
        
        self._d = self._map._dim

        self._gamma_rrt_star = 2. * (1. + 1. / self._d)**(1. / self._d) * (50.0)**(1. / self._d)
        self._yita = self._step_size
        
        
    def add_cost_layer(self, cost_layer):
        self._costMap.add_cost_layer(cost_layer)

    def _card(self):
        return len(self._node_list)
        
    def _compute_connective_range(self):
        return jnp.maximum(self._gamma_rrt_star * (jnp.log(self._card())/ self._card())**(1./self._d), self._connect_range)
        

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
        costs = [self._node_list[idx].cost + self._costMap.compute_edge_cost(self._node_list[idx], new_node)[0] \
                 for idx in near_node_idxs]
        
        if costs:
            sorted_idx = sorted(range(len(costs)), key=lambda k: costs[k])
            for idx in sorted_idx:
                if not self._check_edge_collision(self._node_list[near_node_idxs[idx]], new_node):
                    new_node.reset_parent(self._node_list[near_node_idxs[idx]])
                    break
        edge_cost, aux_infos = self._costMap.compute_edge_cost(new_node.parent, new_node)
        new_node_cost = new_node.parent.cost + edge_cost
        new_node.set_cost(new_node_cost)
        for aux_info in aux_infos:
            if aux_info is not None:
                new_node.min_dist = aux_info
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
                edge_cost, aux_infos = self._costMap.compute_edge_cost(new_node, node)
                updated_cost = new_node.cost + edge_cost
                if updated_cost < node.cost and (not self._check_edge_collision(node, new_node)):
                    node.reset_parent(new_node)
                    node.set_cost(updated_cost)
                    for aux_info in aux_infos:
                        if aux_info is not None:
                            node.min_dist = aux_info
    
    def _generate_final_course(self, node):
        path = []
        path_nodes = [node]
        while node.parent is not None:
            path.append(node._state)
            node = node.parent
            path_nodes.append(node)
        path.append(node._state)
        path_nodes.reverse()
        return path, path_nodes
    
        
    def make_animation(self, path_solution):
        
        path = jnp.array(path_solution)
        _, ax = plt.subplots(1, 1)
        for j in range(len(path_solution)-1):
            ax.scatter(path[j,0], path[j,1], color='r', zorder=1)
            ax.plot(path[j:j+2,0], path[j:j+2,1], color='r', linewidth=2.5, zorder=1)
        for node in self._node_list:
            ax.scatter(node.state[0], node.state[1], c='k', s=0.5)
            if node.parent is not None:
                ax.plot([node.state[0], node.parent.state[0]],
                        [node.state[1], node.parent.state[1]], 
                        'k-.', linewidth=0.5)
            
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim([0, 200])
        ax.set_xlabel('x')
        ax.set_ylim([0, 200])
        ax.set_ylabel('y')
        plt.show()

        
    def plan(self, verbose=True, interval=20, animation=None, early_stop=True):
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

            # verbose (print information)
            if verbose and i % interval == 0:
                tmp_new_node = copy.deepcopy(self._goal)
                tmp_near_node_idxs = self._find_near_node_idx(tmp_new_node)
                if tmp_near_node_idxs:
                    tmp_new_node = self._choose_parent(tmp_near_node_idxs, tmp_new_node)
                    print(f"Iter: {i} || No. of Tree Nodes: {len(self._node_list)} || Cost: {tmp_new_node.cost} || PW: {tmp_new_node.min_dist}")
                    if animation:
                        last_node = copy.deepcopy(self._goal)
                        near_last_node_idxs = self._find_near_node_idx(last_node)
                        last_node = self._choose_parent(near_last_node_idxs, last_node)
                        sol = self._generate_final_course(last_node)
                        path_solution = sol[0]
                        self.make_animation(path_solution)
                else:
                    print('No path can be constructed...')
                    
            self._step_size = 0.9995 * self._step_size if self._step_size > 0.5 else 0.5
            self._connect_range = 0.9995 * self._connect_range if self._connect_range > 10.0 else 10.0
            
        # assemble solution due to max iter
        last_node = copy.deepcopy(self._goal)
        near_last_node_idxs = self._find_near_node_idx(last_node)
        last_node = self._choose_parent(near_last_node_idxs, last_node)
        sol, path_nodes = self._generate_final_course(last_node)
        return sol, path_nodes
    

    def save_result(self, sol, file_name):
        '''
            save the solution
        '''
        # np.save(file_name, np.array(sol))
        with open(file_name, 'wb') as f:
            pickle.dump(sol, f)
        return None
    

class CityRRTStarV2(RRT):
    """
        RRT star implementation
    """
    def __init__(self, connect_range, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._connect_range = connect_range
        
        self._costMap = CostMap()
        self._costMap.add_cost_layer(EuclideanCostMapLayer())
        
        self._d = self._map._dim

        self._gamma_rrt_star = 2. * (1. + 1. / self._d)**(1. / self._d) * (50.0)**(1. / self._d)
        self._yita = self._step_size
        
        
    def add_cost_layer(self, cost_layer):
        self._costMap.add_cost_layer(cost_layer)

    def _card(self):
        return len(self._node_list)
        
    def _compute_connective_range(self):
        return jnp.maximum(self._gamma_rrt_star * (jnp.log(self._card())/ self._card())**(1./self._d), self._connect_range)
        

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
        costs = [self._node_list[idx].cost + self._costMap.compute_edge_cost(self._node_list[idx], new_node)[0] \
                 for idx in near_node_idxs]
        
        if costs:
            sorted_idx = sorted(range(len(costs)), key=lambda k: costs[k])
            for idx in sorted_idx:
                if not self._check_edge_collision(self._node_list[near_node_idxs[idx]], new_node):
                    new_node.reset_parent(self._node_list[near_node_idxs[idx]])
                    break
        edge_cost, aux_infos = self._costMap.compute_edge_cost(new_node.parent, new_node)
        new_node_cost = new_node.parent.cost + edge_cost
        new_node.set_cost(new_node_cost)
        for aux_info in aux_infos:
            if aux_info is not None:
                new_node.min_dist = aux_info
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
                edge_cost, aux_infos = self._costMap.compute_edge_cost(new_node, node)
                updated_cost = new_node.cost + edge_cost
                if updated_cost < node.cost and (not self._check_edge_collision(node, new_node)):
                    node.reset_parent(new_node)
                    node.set_cost(updated_cost)
                    for aux_info in aux_infos:
                        if aux_info is not None:
                            node.min_dist = aux_info
    
    def _generate_final_course(self, node):
        path = []
        path_nodes = [node]
        while node.parent is not None:
            path.append(node._state)
            node = node.parent
            path_nodes.append(node)
        path.append(node._state)
        path_nodes.reverse()
        return path, path_nodes
    
        
    def make_animation(self, path_solution):
        
        path = jnp.array(path_solution)
        _, ax = plt.subplots(1, 1)
        for j in range(len(path_solution)-1):
            ax.scatter(path[j,0], path[j,1], color='r', zorder=1)
            ax.plot(path[j:j+2,0], path[j:j+2,1], color='r', linewidth=2.5, zorder=1)
        for node in self._node_list:
            ax.scatter(node.state[0], node.state[1], c='k', s=0.5)
            if node.parent is not None:
                ax.plot([node.state[0], node.parent.state[0]],
                        [node.state[1], node.parent.state[1]], 
                        'k-.', linewidth=0.5)
            
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim([0, 200])
        ax.set_xlabel('x')
        ax.set_ylim([0, 200])
        ax.set_ylabel('y')
        plt.show()

        
    def plan(self, verbose=True, interval=20, animation=None, early_stop=True):
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

            # verbose (print information)
            if verbose and i % interval == 0:
                tmp_new_node = copy.deepcopy(self._goal)
                tmp_near_node_idxs = self._find_near_node_idx(tmp_new_node)
                if tmp_near_node_idxs:
                    tmp_new_node = self._choose_parent(tmp_near_node_idxs, tmp_new_node)
                    print(f"Iter: {i} || No. of Tree Nodes: {len(self._node_list)} || Cost: {tmp_new_node.cost} || PW: {tmp_new_node.min_dist}")
                    if animation:
                        last_node = copy.deepcopy(self._goal)
                        near_last_node_idxs = self._find_near_node_idx(last_node)
                        last_node = self._choose_parent(near_last_node_idxs, last_node)
                        sol = self._generate_final_course(last_node)
                        path_solution = sol[0]
                        self.make_animation(path_solution)
                else:
                    print('No path can be constructed...')
                    
            self._step_size = 0.9995 * self._step_size if self._step_size > 0.5 else 0.5
            self._connect_range = 0.9995 * self._connect_range if self._connect_range > 10.0 else 10.0
            
        # assemble solution due to max iter
        last_node = copy.deepcopy(self._goal)
        near_last_node_idxs = self._find_near_node_idx(last_node)
        last_node = self._choose_parent(near_last_node_idxs, last_node)
        sol, path_nodes = self._generate_final_course(last_node)
        return sol, path_nodes
    

    def save_result(self, sol, file_name):
        '''
            save the solution
        '''
        # np.save(file_name, np.array(sol))
        with open(file_name, 'wb') as f:
            pickle.dump(sol, f)
        return None