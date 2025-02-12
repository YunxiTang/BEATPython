"""dataclass in python"""
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import functools 
from jax.tree_util import register_pytree_node


def set_jax_config():
    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_platform_name', 'cpu')

@dataclass(frozen=True)
class NNModule:
    name: str = None
    

@dataclass(init=True, repr=True, eq=True, order=True, frozen=True) 
# determine which dunder methods should be auto-generated
# frozen is usefule to store constants, settings et al
class Linearlayer(NNModule):
    scale: float = 1.0
    bias: float = 0.0

    # def __post_init__(self): 
    #     '''
    #         flexible add some operations after initialization (reqyure to set `frozen=False`!)
    #     '''
    #     if self.name == None:
    #         self.name = 'linear_layer'

    def __call__(self, x):
        return self.scale * x + self.bias
    

def special_flatten(v: Linearlayer):
  """Specifies a flattening recipe.

    Params:
        v: the value of registered type to flatten.
    Returns:
        a pair of an iterable with the children to be flattened recursively,
        and some opaque auxiliary data to pass back to the unflattening recipe.
        The auxiliary data is stored in the treedef for use during unflattening.
        The auxiliary data could be used, e.g., for dictionary keys.
  """
  children = (v.scale, v.bias)
  aux_data = (v.name)
  return (children, aux_data)



def special_unflatten(aux_data, children):
  """Specifies an unflattening recipe.

    Params:
        aux_data: the opaque data that was specified during flattening of the
        current treedef.
        children: the unflattened children

    Returns:
        a re-constructed object of the registered type, using the specified
        children and auxiliary data.
  """
  return Linearlayer(aux_data[0], *children)


register_pytree_node(
    Linearlayer,
    special_flatten,    # tell JAX what are the children nodes
    special_unflatten   # tell JAX how to pack back into a RegisteredSpecial
)


if __name__ == '__main__':
    set_jax_config()
    linear1 = Linearlayer('linear1', 0.5, 0.2)
    print(linear1)

    # linear1.scale += 0.2 # error due to frozen is true

    unjitted_res = linear1(1.3)
    print(unjitted_res)
    
    # compatiable with jax conventions
    def mse_loss(model, x, y):
        tmp = model(x)
        return jnp.sum((x - tmp) ** 2)
    
    grad_func = jax.grad(mse_loss, argnums=[0,])
    grads = grad_func(linear1, 1.0, 0.6)
    print('grads: ', grads)

    # ==================================
    jitted_linear = jax.jit(linear1, static_argnums=[0])
    # print(type(jitted_linear))
    print(jitted_linear(0.5))
    x = jax.numpy.array([1, 2, 3, 4])
    print(x.device())
    res = jax.vmap(jitted_linear)(x)
    print(res)
