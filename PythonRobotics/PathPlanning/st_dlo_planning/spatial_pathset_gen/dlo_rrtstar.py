"""RRT* Implementation"""
import jax.numpy as jnp
import jax.random as random
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple
import copy
import pickle
from typing import List
import kdtree


if __name__ == '__main__':
    
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)


class Node(object):
    """
        Node in RRT/RRT* algorithm
    """
    def __init__(self, state):
        self._state = state
        self._parent = None
        
        # For usage in RRT*
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
    
    def __len__(self):
        return len(self.state)

    def __getitem__(self, i):
        return self.state[i]

    def __repr__(self):
        return '[State({}), Cost({})]'.format(self.state, self.cost)
    

def compute_node_distance(node1: Node, node2: Node):
    return jnp.linalg.norm(node1._state - node2._state)


class PassageAwareRRTStar:
    """
        RRT star implementation
    """
    def __init__(self, connect_range, start, goal, world_map, 
                 step_size, max_iter, goal_sample_rate, seed):

        self._start = Node(start)
        self._goal = Node(goal)
        
        self._world_map = world_map
        self._resolution = world_map._resolution

        self._node_tree = kdtree.create(dimensions=3)
        
        self._step_size = step_size if step_size > self._resolution else 2.0 * self._resolution
        self._max_iter = max_iter
        self._goal_sample_rate = goal_sample_rate
        
        self._rng_key = random.PRNGKey(seed=seed)

        self._connect_range = connect_range
        
        self._costMap = CostMap()
        self._costMap.add_cost_layer(EuclideanCostMapLayer())
        
        self._d = self._world_map._dim

        self._gamma_rrt_star = 2. * (1. + 1. / self._d)**(1. / self._d) * (50.0)**(1. / self._d)
        self._yita = self._step_size
        
        
    def add_cost_layer(self, cost_layer):
        self._costMap.add_cost_layer(cost_layer)

    # def _card(self):
    #     return len(self._node_list)

    def _get_random_node(self):
        '''
            sample a collision-free random node
        '''
        self._rng_key, rng_key = random.split(self._rng_key, 2)
        
        if random.randint(rng_key, (1,), 0, 100)[0] > self._goal_sample_rate:
            rand_state = self._world_map.sample_free_pos()
        else:
            rand_state = self._world_map.sample_free_pos(toward_goal=True)
        
        return Node(rand_state)
        
    def _compute_connective_range(self):
        # return jnp.maximum(self._gamma_rrt_star * (jnp.log(self._card())/ self._card())**(1./self._d), self._connect_range)
        return self._connect_range
    
    def _get_nearest_node(self, rand_node: Node):
        nearest_node, _ = self._node_tree.search_nn( rand_node )
        return nearest_node

    def _find_near_nodes(self, new_node: Node):
        """find all the nodes in some range

        Args:
            new_node (Node): node
        Returns:
            the nearest nodes in the connective range
        """
        near_nodes = self._node_tree.search_nn_dist(new_node, self._connect_range)
        return near_nodes
    
    def _steer(self, from_node: Node, to_node: Node, step_size: float) -> Node:
        dist = jnp.linalg.norm(to_node.state - from_node.state)
        if dist <= step_size:
            new_node_state = to_node.state
        else:
            new_node_state = from_node.state + step_size * (to_node.state - from_node.state) / dist
        new_node = Node(new_node_state)
        new_node.set_parent(from_node)
        return new_node

    def _choose_parent(self, near_nodes: List[Node], new_node: Node):
        """
            computes the cheapest node to new_node contained in the list
            and reset the node as new_node.parent.
        """
        costs = [near_node.cost + self._costMap.compute_edge_cost(near_node, new_node)[0] \
                 for near_node in near_nodes]
        
        if costs:
            sorted_idx = sorted(range(len(costs)), key=lambda k: costs[k])
            for idx in sorted_idx:
                if not self._check_edge_collision(near_nodes[idx], new_node):
                    new_node.reset_parent(near_nodes[idx])
                    break
        edge_cost, aux_infos = self._costMap.compute_edge_cost(new_node.parent, new_node)
        new_node_cost = new_node.parent.cost + edge_cost
        new_node.set_cost(new_node_cost)
        for aux_info in aux_infos:
            if aux_info is not None:
                new_node.min_dist = aux_info
        return new_node
    
    
    def _rewire(self, new_node: Node, near_nodes: List[Node]):
        """
            For each node in near_inds, this will check if it is cheaper to
            arrive to them from new_node.
            In such a case, this will re-assign the parent of the nodes in
            near_inds to new_node.
        """
        for node in near_nodes:
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
        self._node_tree.add(self._start)
        
        for i in range(self._max_iter):
            # sample a random node
            rand_node = self._get_random_node()

            # find the nearest node in the tree
            nearest_node = self._get_nearest_node(rand_node)
            
            # get new node candidate
            new_node = self._steer(nearest_node, rand_node, self._step_size)
            
            if not self._check_node_collision(new_node):
                # reset node parent
                near_nodes = self._find_near_nodes(new_node)
                new_node = self._choose_parent(near_nodes, new_node)
                self._node_tree.add(new_node.state)

                # rewire the tree
                self._rewire(new_node, near_nodes)

            # verbose (print information)
            if verbose and i % interval == 0:
                tmp_new_node = copy.deepcopy(self._goal)
                tmp_near_nodes = self._find_near_nodes(tmp_new_node)
                if tmp_near_nodes:
                    tmp_new_node = self._choose_parent(tmp_near_nodes, tmp_new_node)
                    print(f"Iter: {i} || Cost: {tmp_new_node.cost} || PW: {tmp_new_node.min_dist}")
                    
                    if animation:
                        sol = self._generate_final_course(tmp_new_node)
                        path_solution = sol[0]
                        self.make_animation(path_solution)
                else:
                    print('No path can be constructed...')
                    
            self._step_size = 0.9995 * self._step_size if self._step_size > 0.5 else 0.5
            self._connect_range = 0.9995 * self._connect_range if self._connect_range > 10.0 else 10.0
            
        # assemble solution due to max iter
        last_node = copy.deepcopy(self._goal)
        near_last_nodes = self._find_near_nodes(last_node)
        last_node = self._choose_parent(near_last_nodes, last_node)
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