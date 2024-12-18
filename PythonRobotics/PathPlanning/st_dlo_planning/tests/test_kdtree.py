import kdtree
import jax.numpy as jnp
import copy


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



tree = kdtree.create(dimensions=3)

tree.add( jnp.array([5, 4, 3]) )
tree.add( (15, 4, 3) )
tree.add( (5, 14, 3) )
tree.add( (5, 4, 13) )

print( tree.search_nn( (1, 2, 3)) )

print( tree.search_knn( (1, 2, 3), 2) )

res = tree.search_nn_dist( (1, 2, 3), 200.)

for r in res:
    print(r)

print(' * ' * 30)

new_tree = kdtree.create(dimensions=3)

state = jnp.array([10, 10, 50.])
print(len(state))
node = Node(state)
new_tree.add(node)

state1 = jnp.array([1, 2, 3])
node1 = Node(state1)
new_tree.add(node1)

state2 = jnp.array([12, 2, 3])
node2 = Node(state2)

kdtree.visualize(new_tree)
res, dist = new_tree.search_nn( node2 )
print(res.data)

print(' * ' * 30)
res.data.set_cost(0)
kdtree.visualize(new_tree)


