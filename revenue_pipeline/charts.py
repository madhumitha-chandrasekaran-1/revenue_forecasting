"""Consistent matplotlib styling for the notebook figures and the README images."""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter

from . import config

C = config.COLORS
SERIES = [C["series_1"], C["series_2"], C["series_3"]]
REFERENCE = "#9a9893"


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": C["surface"], "axes.facecolor": C["surface"], "savefig.facecolor": C["surface"],
        "axes.edgecolor": C["grid"], "axes.labelcolor": C["text_muted"], "axes.titlecolor": C["text"],
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": C["grid"], "grid.linewidth": 0.8,
        "xtick.color": C["text_muted"], "ytick.color": C["text_muted"], "xtick.labelsize": 9, "ytick.labelsize": 9,
        "lines.linewidth": 2, "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": C["text"],
        "font.size": 10, "text.color": C["text"], "figure.dpi": 110,
    })


def money(ax, axis="y", unit="M"):
    scale = {"M": 1e6, "K": 1e3}[unit]
    fmt = FuncFormatter(lambda v, _: f"${v / scale:,.1f}{unit}" if unit == "M" else f"${v / scale:,.0f}{unit}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def percent(ax, axis="y", decimals=0):
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(PercentFormatter(1.0, decimals=decimals))


def month_axis(ax, every=3):
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=range(1, 13, every)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))


def end_label(ax, x, y, text, color=None, dy=0):
    """Direct label at the end of a line, in text ink (the line beside it carries the colour)."""
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center",
                fontsize=9, color=color or C["text"])


def save(fig, name):
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.IMAGES_DIR / name, dpi=160, bbox_inches="tight")
