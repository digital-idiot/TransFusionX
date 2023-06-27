import torch
import pandas as pd
from typing import Any
from typing import Dict
from typing import Sequence
from seaborn import heatmap
from tabulate import tabulate
from matplotlib import cm as mpl_cm
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from model.registry import FLAG_ENUMS
from matplotlib.colors import Colormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


# noinspection PyUnresolvedReferences,SpellCheckingInspection
def plot_confusion_matrix(
        cm_df: pd.DataFrame,
        annot: bool = True,
        fig_size: float = 10,
        dpi: float = 300,
        font_size: int = 22,
        fmt: str = '.2f',
        cmap: Colormap = mpl_cm.plasma,
        cbar: bool = True,
        key: str = None
) -> Figure:
    """
    Makes pretty plot of confusion matrix
    Args:
        cm_df: confusion matrix DataFrame
        annot: annotation flag
        fig_size: FIgure Size
        dpi: DPI of the figure
        font_size: font size
        fmt: float format signature
        cmap: color map
        cbar: color bar flag
        key: addition string for title
    Returns:
        Instance of matplotlib.figure.Figure containing the plot
    """
    # noinspection PyUnresolvedReferences
    plt.rcParams.update({'font.size': font_size})
    fig = plt.figure(figsize=(fig_size, (fig_size * 1.055)), dpi=dpi)
    if key:
        fig.suptitle(f"Confusion Matrix: {key}")
    else:
        fig.suptitle(f"Confusion Matrix")

    ax = fig.gca()
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    ax.xaxis.tick_top()
    heatmap(
        data=cm_df,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        cbar=cbar,
        ax=ax,
        cbar_ax=cax
    )
    ax.tick_params(axis="y", rotation=0)
    return fig


def format_report(report_dict: dict):
    if report_dict['prefix'] and report_dict['postfix']:
        key = f"Report: {report_dict['prefix']} | {report_dict['postfix']}"
    elif report_dict['prefix']:
        key = f"Report: {report_dict['prefix']}"
    elif report_dict['postfix']:
        key = f"Report: {report_dict['postfix']}"
    else:
        key = "Report"
    symbol_length = 80 - (len(key) + 2)
    pre = symbol_length // 2
    post = symbol_length - pre
    report_start = f"\n{'=' * pre} {key} {'=' * post}"

    end_key = "Report End"
    symbol_length = 80 - (len(end_key) + 2)
    pre = symbol_length // 2
    post = symbol_length - pre
    report_end = f"{'=' * pre} {end_key} {'=' * post}\n"

    scalar_table = tabulate(
        tabular_data=report_dict['quality_report'],
        headers='keys',
        showindex=True,
        tablefmt="fancy_grid",
        floatfmt='.2f',
        numalign='center',
        stralign='left'
    )

    class_table = tabulate(
        tabular_data=report_dict['class_report'],
        headers='keys',
        showindex=True,
        tablefmt="fancy_grid",
        floatfmt='.2f',
        numalign='center',
        stralign='left',
    )

    # noinspection SpellCheckingInspection
    confmat_table = tabulate(
        tabular_data=report_dict['confusion_matrix'],
        headers='keys',
        showindex=True,
        tablefmt="fancy_grid",
        floatfmt='.2f',
        numalign='center',
        stralign='left',
    )

    report = "\n\n".join([
        report_start,
        f"Quality Scores:\n{scalar_table}",
        f"Class-wise Metrics:\n{class_table}",
        f"Confusion Matrix:\n{confmat_table}",
        report_end
    ])
    return report


def delete_indices(tensor: torch.Tensor, indices: Sequence[int]):
    if indices is None:
        return tensor
    else:
        mask = torch.full_like(
            input=tensor,
            fill_value=True,
            dtype=torch.bool,
            device=tensor.device
        )
        mask[indices] = False
        return tensor[mask]


def replace_enum(d: Dict[str, Any]):
    for k, v in d.items():
        if isinstance(v, dict) and ("enum" in v.keys()) and (len(v) == 1):
            d[k] = getattr(FLAG_ENUMS, v["enum"])
    return d
