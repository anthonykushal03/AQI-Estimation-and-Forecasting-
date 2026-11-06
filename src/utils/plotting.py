"""Figure saving."""
import matplotlib.pyplot as plt

from ..setup import config


def save_fig(name, fig=None, folder=None):
    """Save a figure to <folder>/<name>.png (default results/figures/), close it, and return it
    (so a notebook can display it)."""
    fig = fig or plt.gcf()
    folder = folder or config.FIG_DIR
    folder.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(folder / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
