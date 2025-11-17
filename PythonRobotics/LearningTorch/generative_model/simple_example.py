import torch
import numpy as np


def generate_sample():
    """
    make a data distribution
    """
    dist = torch.distributions.Normal(
        loc=torch.tensor([0.0]), scale=torch.tensor([1.0])
    )
    ys = 0

    return dist.sample(
        sample_shape=torch.tensor(
            [
                5,
            ]
        )
    ), ys


if __name__ == "__main__":
    xgt, ygt = generate_sample()
    print(xgt.shape)
