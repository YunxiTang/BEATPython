import numpy as np
import jax
from typing import Iterable


class FlaxDataloader(Iterable):
    """
        A simple custom data loader for flax
    """
    def __init__(self, X, y, batch_size, rng):
        super().__init__()
        self.X = X
        self.y = y
        self.num_samples = X.shape[0]
        self.rng = rng
        self.batch_size = batch_size
        
    def __iter__(self):
        indices = jax.random.permutation(self.rng, self.num_samples)
        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.X[batch_indices], self.y[batch_indices]


if __name__ == '__main__':
    # Example usage
    X = np.random.rand(1000, 32, 32, 3)  # Random dataset (1000 images, 32x32x3)
    y = np.random.randint(0, 10, 1000)   # Random labels (10 classes)

    rng = jax.random.PRNGKey(0)
    batch_size = 32

    dataloader = FlaxDataloader(X, y, 2, rng)
    for batch_X, batch_y in dataloader:
        batch_X = jax.device_put(batch_X)  # Put data on device (e.g., GPU)
        print(batch_X.shape, batch_y.shape)
        break