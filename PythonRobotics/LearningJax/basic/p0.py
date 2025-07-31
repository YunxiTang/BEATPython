import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if __name__ == "__main__":
    import numpy as np
    import jax.numpy as jnp
    import jax
    import time

    def f(x):  # function we're benchmarking (works in both NumPy & JAX)
        return x.T @ (x - x.mean(axis=0))

    x_np = np.ones((8000, 8000), dtype=np.float32)  # same as JAX default dtype

    tc = time.time()
    for i in range(20):
        f(x_np)  # measure NumPy runtime
    print((time.time() - tc) / 20)

    tc = time.time()
    x_jax = jax.device_put(x_np)  # measure JAX device transfer time
    print(time.time() - tc)

    tc = time.time()
    f_jit = jax.jit(f)
    print(time.time() - tc)  # measure JAX compilation time
    f_jit(x_jax)

    tc = time.time()
    for i in range(200):
        f_jit(x_jax)  # measure JAX runtime
    print((time.time() - tc) / 200)
