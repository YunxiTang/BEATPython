"""RRT* Implementation"""
import jax.numpy as jnp
import jax
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import dill
from typing import List, NamedTuple, Tuple
from itertools import combinations
import random

from components import (CostMap, Node, State, WorldMap, 
                        Modality, EuclideanCostMapLayer, ModalityCostMapLayer)


class ModalityRRTStar:
    """
        Modality RRT* for motion modality planning
    """
    def __init__(self, 
                 start_node: Node, 
                 goal_node: Node, 
                 connect_range: float, 
                 map: WorldMap,
                 step_size: float = 1.0,
                 goal_sample_rate: int = 50,
                 max_iter: int = 500,
                 seed: int = 0):
        
        self._start = start_node
        self._goal = goal_node
        self._step_size = step_size
        self._goal_sample_rate = goal_sample_rate
        self._max_iter = max_iter
        self._rng_key = jax.random.PRNGKey(seed=seed)
        
        # physical map
        self._map = map
        
        self._connect_range = connect_range
        self._potential_final_node = self._start
        
        # cost map
        self._costMap = CostMap()
        self._costMap.add_cost_layer(EuclideanCostMapLayer(map))
        self._costMap.add_cost_layer(ModalityCostMapLayer(map))

        self._node_list = []
        
        self._d = self._map._dim
        self._gamma_rrt_star = 2. * (1. + 1. / self._d)**(1. / self._d) * (1.0)**(1. / self._d)
        self._yita = self._step_size

    def _card(self):
        return len(self._node_list)
    
    def _get_random_node(self) -> Node:
        '''
            sample a feasible random node
        '''
        self._rng_key, rng_key, pos_key = jax.random.split(self._rng_key, 3)
        
        # sample a position
        if jax.random.randint(rng_key, (1,), 0, 100)[0] > self._goal_sample_rate:
            x, terrain = self._map.sample_pos(pos_key)
        else:
            x, terrain = self._map.sample_pos(pos_key, toward_goal=True)
        # sample a modality 
        modality = random.choice(list(Modality))
        state = State(x, modality)
        
        rand_node = Node(state)
        return rand_node
    
    @staticmethod
    def _compute_node_distance(node1: Node, node2: Node):
        return jnp.linalg.norm(node1._state.x - node2._state.x)
    
    def _get_nearest_node(self, rand_node: Node):
        '''
            find the nearest node for a given node
        '''
        dlist = [ModalityRRTStar._compute_node_distance(rand_node, node) for node in self._node_list]
        dlist = jnp.array(dlist)
        min_idx = jnp.argmin(dlist)
        min_dist = dlist[min_idx]
        return min_idx, min_dist
    
    def _steer(self, from_node: Node, to_node: Node, step_size: float) -> Node:
        '''
            steer to generate new node (the modality is aligned with the to_node's modality)
        '''
        dist = jnp.linalg.norm(to_node._state.x - from_node._state.x)
        if dist <= step_size:
            new_node_state = to_node._state
        else:
            new_node_state_x = from_node._state.x + step_size * (to_node._state.x - from_node._state.x) / dist
            new_node_state = State(new_node_state_x, to_node._state.mode)
        new_node = Node(new_node_state)
        new_node.set_parent(from_node)
        new_node.set_node_cost(self._costMap.compute_node_cost(new_node)[0])
        return new_node
        
    def _compute_connective_range(self):
        return self._connect_range
        # return jnp.minimum(self._gamma_rrt_star * (jnp.log(self._card())/ self._card())**(1./self._d), self._connect_range)
        
    def _find_near_node_idx(self, new_node: Node):
        """find all the nodes in the connective range

        Args:
            new_node (Node): node
        """
        node_dists = [ModalityRRTStar._compute_node_distance(new_node, node) for node in self._node_list]
        near_idxs = [node_dists.index(i) for i in node_dists if i <= self._compute_connective_range()]
        return near_idxs
    

    def _choose_parent(self, near_node_idxs: List[int], new_node: Node):
        """
            computes the cheapest node to new_node contained in the list
            and reset the node as new_node.parent.
        """
        costs = [new_node.node_cost + self._node_list[idx].path_cost + self._costMap.compute_edge_cost(self._node_list[idx], new_node)[0] for idx in near_node_idxs]
        
        if costs:
            sorted_idx = sorted(range(len(costs)), key=lambda k: costs[k])
            for idx in sorted_idx:
                new_node.reset_parent(self._node_list[near_node_idxs[idx]])
                break

        new_node_path_cost = new_node.node_cost + new_node.parent.path_cost + self._costMap.compute_edge_cost(new_node.parent, new_node)[0]
        new_node.set_path_cost(new_node_path_cost)
        return new_node
    
    
    def _rewire(self, new_node: Node, near_node_idxs: List[int]):
        """
            For each node in `near_node_idxs`, this will check if it is cheaper to
            arrive to the node from the new_node.
            In such a case, this will re-assign the parent of some nodes in
            `near_node_idxs` to new_node.
        """
        for idx in near_node_idxs:
            node = self._node_list[idx]
            if new_node.parent != node:
                updated_cost = new_node.path_cost + node.node_cost + self._costMap.compute_edge_cost(new_node, node)[0]
                if updated_cost < node.path_cost:
                    node.reset_parent(new_node)
                    node.set_path_cost(updated_cost)


    def _generate_final_course(self, node):
        path = []
        # node = self._potential_final_node
        while node.parent is not None:
            path.append(node._state)
            node = node.parent
        path.append(node._state)
        return path
    

    def add_cost_layer(self, cost_layer):
        self._costMap.add_cost_layer(cost_layer)

    
    def _calc_dist_to_goal(self, node: Node):
        return ModalityRRTStar._compute_node_distance(node, self._goal)    
    
    
    def plan(self, verbose=True, early_stop=True):
        stater_cost = self._costMap.compute_node_cost(self._start)[0]
        self._start.set_node_cost(stater_cost)
        self._start.set_path_cost(stater_cost)
        
        self._goal.set_node_cost(self._costMap.compute_node_cost(self._goal)[0])
        
        self._node_list.append(self._start)
        
        for i in range(self._max_iter):
            # 1. sample a random node
            rand_node = self._get_random_node()
            
            # find the nearest node in the tree
            nearest_ind, _ = self._get_nearest_node(rand_node)
            nearest_node = self._node_list[nearest_ind]
            
            # 2. get new NODE CANDIDATE
            new_node = self._steer(nearest_node, rand_node, self._step_size)
            
            # 3. reset the new node's parent
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
            #     self._goal.set_path_cost(
            #         self._potential_final_node.path_cost \
            #         + self._goal.node_cost + self._costMap.compute_edge_cost(self._potential_final_node, self._goal)[0])
            #     sol = self._generate_final_course()
            #     print(f'Find a path with {len(sol)} nodes due to early stop at iter {i}. Goal cost: {self._goal.path_cost}')
            #     return sol

            # verbose (print information)
            if i % 10 == 0:
                # path validation
                tmp_new_node = copy.deepcopy(self._goal)
                tmp_near_node_idxs = self._find_near_node_idx(tmp_new_node)
                if tmp_near_node_idxs:
                    tmp_new_node = self._choose_parent(tmp_near_node_idxs, tmp_new_node)
                    print(f"Iter: {i} || {tmp_new_node.path_cost * self._step_size}")
                else:
                    print('No path can be constructed...')
                
        # assemble solution due to max iter
        # self._goal.set_parent(self._potential_final_node)
        # self._goal.set_path_cost(
        #     self._potential_final_node.path_cost \
        #     + self._goal.node_cost + self._costMap.compute_edge_cost(self._potential_final_node, self._goal)[0])
        last_node = copy.deepcopy(self._goal)
        near_last_node_idxs = self._find_near_node_idx(last_node)
        last_node = self._choose_parent(near_last_node_idxs, last_node)
        sol = self._generate_final_course(last_node)
        print(f'Find a path with {len(sol)} nodes due to max_iter. Goal cost: {last_node.path_cost}')
        
        return sol