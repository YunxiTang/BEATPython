import seaborn as sns
import dill
import pathlib
import matplotlib.pyplot as plt

if __name__ == "__main__":
    num = 1
    res_path = pathlib.Path(f"./case{num}_res.pkl").absolute()
    with open(res_path, "rb") as f:
        res = dill.load(f)

    xs = res["xs"]
    modes = res["modes"]
    X = list(reversed(xs))

    sns.set_theme("talk")
    plt.step(X, modes, "k-")
    plt.tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
    )  # labels along the bottom edge are off
    plt.yticks(
        [1, 2, 3], ["Roll", "Hop", "Flip"], rotation=45
    )  # Set text labels and properties.
    plt.savefig(f"./case{num}_res.png", dpi=1200)
    plt.show()
