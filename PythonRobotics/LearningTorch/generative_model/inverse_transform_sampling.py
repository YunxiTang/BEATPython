"""
Inverse sampling
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


if __name__ == "__main__":
    u = torch.distributions.Uniform(
        low=torch.tensor(
            [
                0.0,
            ]
        ),
        high=torch.tensor(
            [
                1.0,
            ]
        ),
    )
    dist = torch.distributions.Normal(0.0, 0.2)

    p = u.sample((20000,)).flatten()
    x = dist.icdf(p)

    print(p.shape, x.shape)

    sns.set_theme(context="paper")
    fig, ax = plt.subplots(1, 2)
    ax[0].set_title("prob")
    ax[0].hist(p, bins=10, density=False)

    ax[1].set_title("samples")
    ax[1].hist(x, bins=20, density=False)
    plt.show()
