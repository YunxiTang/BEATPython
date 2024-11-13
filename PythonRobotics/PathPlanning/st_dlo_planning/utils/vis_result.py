import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation


def animate_solution(solution):
    fig, ax = plt.subplots()
    container = ax.plot(x, data, color=colors)

if __name__ == '__main__':
    fig, ax = plt.subplots()
    rng = np.random.default_rng(19680801)
    data = np.array([20, 20, 20, 20])
    x = np.array([1, 2, 3, 4])

    artists = []
    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple']
    for i in range(20):
        data += rng.integers(low=0, high=10, size=data.shape)
        container = ax.barh(x, data, color=colors)
        artists.append(container)


    ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=400)
    plt.show()