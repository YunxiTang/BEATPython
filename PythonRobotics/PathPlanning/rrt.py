"""RRT Implementation"""
import jax.numpy as jnp
import jax.random as random
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple
import copy


def plot_circle(x, y, size, color="-b"):
    deg = list(range(0, 360, 5))
    deg.append(0)
    xl = [x + size * jnp.cos(jnp.deg2rad(d)) for d in deg]
    yl = [y + size * jnp.sin(jnp.deg2rad(d)) for d in deg]
    plt.plot(xl, yl, color)

class Node:
    def __init__(self, state) -> None:
        self._state = state
        self._parent = None
        
        # For addtional usage
        self._cost = 0.0
        
    def set_parent(self, node):
        self._parent = node
        
    def set_cost(self, cost):
        self._cost = cost
        
    @property
    def state(self):
        return self._state
    
    @property
    def parent(self):
        return self._parent
    
    @property
    def cost(self):
        return self._cost
    
    
class WorldMap:
    def __init__(self, arena, resolution: float = 0.01) -> None:
        
        self.xmin = float(arena[0])
        self.xmax = float(arena[1])
        self.ymin = float(arena[2])
        self.ymax = float(arena[3])
        
        self._lb = jnp.array([self.xmin, self.ymin])
        self._ub = jnp.array([self.xmax, self.ymax])
        
        self._resolution = resolution
        
        self._dim = 2
        
        self._obstacle = []
        
        self._start = None
        self._goal = None
        
        self._rng_key = random.PRNGKey(seed=146)
        
    def update_start(self, start: jnp.ndarray):
        self._start = start
        
    def update_goal(self, goal: jnp.ndarray):
        self._goal = goal
        
    def add_obstacle(self, obstacle: Tuple):
        self._obstacle.append(obstacle)
        
    def sample_pos(self, toward_goal: bool = False) -> jnp.ndarray:
        if toward_goal and self._goal is not None:
            return self._goal
        else:
            self._rng_key, rng_key = random.split(self._rng_key, 2)
            return random.uniform(rng_key, shape=(self._dim,), minval=self._lb, maxval=self._ub)
        
    def check_line_collision(self, start_state: jnp.ndarray, end_state: jnp.ndarray) -> bool:
        state_distance = jnp.linalg.norm(start_state - end_state)
        N = int(state_distance / self._resolution)
        ratios = jnp.linspace(0., 1.0, num=N)
        for (ox, oy, size) in self._obstacle:
            center = jnp.array([ox, oy])
            for ratio in ratios:
                state_sample = (1 - ratio) * start_state + ratio * end_state
                dist = jnp.linalg.norm(state_sample - center) - size
                if dist <= 0.:
                    # collision
                    return True  
        return False
    
    def check_pos_collision(self, state):
        for (ox, oy, size) in self._obstacle:
            center = jnp.array([ox, oy])
            dist = jnp.linalg.norm(state - center) - size
            if dist <= 0.:
                # collision
                return True  
        return False
        
        
class RRT:
    def __init__(self, 
                 start_config: jnp.ndarray,
                 goal_config: jnp.ndarray,
                 map: WorldMap,
                 step_size: float = 0.01,
                 goal_sample_rate: int = 50,
                 max_iter: int = 500,
                 seed: int = 0
                 ) -> None:
        self._start = Node(start_config)
        self._goal = Node(goal_config)
        
        self._resolution = 0.1
        
        self._map = map
        self._node_list = []
        
        self._step_size = step_size
        self._max_iter = max_iter
        self._goal_sample_rate = goal_sample_rate
        
        self._rng_key = random.PRNGKey(seed=seed)
        
    def _get_random_node(self):
        self._rng_key, rng_key = random.split(self._rng_key, 2)
        
        if random.randint(rng_key, (1,), 0, 100) > self._goal_sample_rate:
            rand_state = self._map.sample_pos()
        else:
            rand_state = self._map.sample_pos(toward_goal=True)
        
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
        new_node_state = from_node.state + step_size * (to_node.state - from_node.state) / jnp.linalg.norm(to_node.state - from_node.state)
        new_node = Node(new_node_state)
        new_node.set_parent(from_node)
        return new_node
    
    
    def _check_edge_collision(self, node1: Node, node2: Node) -> bool:
        return self._map.check_line_collision(node1.state, node2.state)
    
    
    def _check_node_collision(self, node: Node):
        return self._map.check_pos_collision(node.state)
        
        
    def plan(self, animation=True, verbose=False):
        self._node_list = [self._start]
        
        for i in range(self._max_iter):
            rand_node = self._get_random_node()
            
            nearest_ind, _ = self._get_nearest_node(rand_node)
            nearest_node = self._node_list[nearest_ind]
            
            new_node = self._steer(nearest_node, rand_node, self._step_size)
            
            if not self._check_edge_collision(nearest_node, new_node):
                self._node_list.append(new_node)
                
            if animation and i % 5 == 0:
                self._draw_graph(rand_node)
                
            if self._calc_dist_to_goal(self._node_list[-1]) <= self._step_size:
                final_node = copy.deepcopy(self._goal)
                final_node.set_parent(new_node)
                self._node_list.append(new_node)
                print('Find a feasible path.')
                return self._generate_final_course()
            else:
                print(f"Iter: {i} || No. Nodes: {len(self._node_list)}")
        print('Failed to find a feasible path.')
        return self._generate_final_course()
    

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
    
    rrt = RRT(
        start_config=start,
        goal_config=goal,
        map=world_map,
        step_size=0.1
    )
    
    path_solution = rrt.plan()
    print(f'Path Length: {len(path_solution)}')
    
    if path_solution is not None:
        path = jnp.array(path_solution)
        
        sns.set()
        for obs in world_map._obstacle:
            plot_circle(obs[0], obs[1], obs[2])
            
        for node in rrt._node_list:
            plt.scatter(node.state[0], node.state[1], c='k')
            if node.parent != None:
                plt.plot([node.state[0], node.parent.state[0]], [node.state[1], node.parent.state[1]], 'k-.')
        
        for i in range(len(path_solution)-1):
            plt.scatter(path[i,0], path[i,1])
            plt.plot(path[i:i+2,0], path[i:i+2,1])
        
        plt.scatter(path[len(path_solution)-1,0], path[len(path_solution)-1,1]) 
        
        plt.scatter(start[0], start[1], marker='*', linewidths=2)
        plt.scatter(goal[0], goal[1], marker='*', linewidths=2) 
        plt.axis('equal')
        plt.show()        
    
    
    
        
    