import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from typing import List, Dict, Literal, Tuple, Union
from numpy.typing import ArrayLike

def scatter(
    *args: List[ArrayLike],
    scatter_label: List[str],
    output_path: str,
    s: Union[ArrayLike, float] = 10.0,
    marker: List[Literal['o', 'v', '^', 's', 'x', 'D', ]] = ['o', 'o', 'o', 'o'],
    text: str = None,
    dot_color_list: List = ['#03788C', '#F27457', '#03488C', 'indianred', 'steelblue'],
    line: bool = True,
    line_color: str = 'r',
    line_width: float = 0.5,
    xlabel: str = 'Target',
    ylabel: str = 'Predict',
    figsize: Tuple = (10,10),
    axis_fontsize: float = 14,
    ylabel_fontsize: float = 16,
    xlabel_fontsize: float = 16,
    legend_fontsize: float = 10,
    bbox_to_anchor: Tuple[float, float] = (1.0, 1.0),
    label_fontweight: Literal['normal', 'bold'] = 'normal',
    ):
    '''scatter plot
    '''
    plt.rcParams['font.size'] = axis_fontsize
    plt.rcParams['mathtext.fontset'] = 'custom'

    fig = plt.figure(figsize = figsize, dpi = 300)
    ax_1 = fig.add_subplot(111)
    ax_1.set_ylabel(ylabel, fontsize = ylabel_fontsize, fontweight = label_fontweight)
    ax_1.set_xlabel(xlabel, fontsize = xlabel_fontsize, fontweight = label_fontweight)

    lower_limit = np.inf
    upper_limit = -np.inf
    for i, data in enumerate(args):
        x = data[0].cpu().detach().numpy() if isinstance(data[0], torch.Tensor) else data[0]
        y = data[1].cpu().detach().numpy() if isinstance(data[1], torch.Tensor) else data[1]
        lower_limit = min([min(x), min(y)]) if min([min(x), min(y)]) < lower_limit else lower_limit
        upper_limit = max([max(x), max(y)]) if max([max(x), max(y)]) > upper_limit else upper_limit
        ax_1.scatter(x, y, s, dot_color_list[i], label = scatter_label[i], marker = marker[i])

    if line:
        displacement = (upper_limit - lower_limit) * 0.1
        ax_1.plot(
            [lower_limit - displacement, upper_limit + displacement],
            [lower_limit - displacement, upper_limit + displacement],
            color = line_color,
            linewidth = line_width,
            label = 'y = x'
            )
    fig.legend(
        bbox_to_anchor = bbox_to_anchor,
        bbox_transform = ax_1.transAxes,
        fontsize = legend_fontsize
        )

    if text:
        plt.text(lower_limit, upper_limit, f'{text}')
    plt.savefig(
        output_path,
        bbox_inches = 'tight',
        dpi = 300
        )
    plt.close()
    return fig

def loss_epoch(
    data_list: List[List[List[float]]],
    label_list: List[str],
    color_list: List[str],
    xlabel: str,
    ylabel: str,
    output_path: str,
    fontsize: float = 20,
    ):
    '''line graph
    '''
    assert len(data_list) == len(label_list), 'Data_list and Label_list are not of the same length.'
    assert len(data_list) <= len(color_list), 'There are not enough colors.'
    plt.figure(
        figsize = (10, 10),
        dpi = 300
        )
    for i, data in enumerate(data_list):
        x = data[0]
        y = data[1]
        plt.plot(x, y, color = color_list[i], label = label_list[i])
    plt.legend(loc = 'best', fontsize = fontsize)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(
        output_path,
        bbox_inches = 'tight',
        dpi = 300
        )
    plt.close()

def corr_heatmap(
    data: Dict,
    output_path: str,
    annot: bool,
    corr_method: Literal['pearson', 'kendall', 'spearman'] = 'pearson',
    ):
    length_list = [len(value) for value in data.values()]
    assert min(length_list) == max(length_list), 'The data of features are not of the same length.'
    corr = pd.DataFrame(data).corr(corr_method)
    if len(data.keys()) < 10:
        figsize = 10
    else:
        figsize = 50
    plt.figure(
        figsize = (figsize, figsize),
        dpi = 300
        )
    sns.heatmap(
        corr,
        annot = annot,
        annot_kws = {"size": 12},
        vmin = -1,
        vmax = 1,
        cmap = "RdBu_r"
        )
    plt.xticks(rotation = 45, fontsize = 15)
    plt.yticks(rotation = 45, fontsize = 15)
    plt.savefig(
        output_path,
        bbox_inches = 'tight',
        dpi = 300
        )
    plt.close()

def hist(
    data: ArrayLike,
    bins: int,
    range: Tuple,
    output_path: str,
    title: str = None,
    xlabel: str = None,   # e.g. r'Value $(g \cdot cm^3)$'
    ylabel: str = 'Frequency',
    ):
    plt.figure(dpi=300)
    plt.hist(
        data,
        bins = bins,
        color = '#03658C',
        alpha=1,
        range = range,
        edgecolor = 'white'
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.savefig(
        output_path,
        bbox_inches = 'tight',
        dpi = 300
        )

def bar_2y(
    data: Dict[str, List[ArrayLike]],   # e.g. {'Unit1 $\mathrm{g \cdot cm^3}$': [[1,2,3,4,5],[2,3,4,5,6]]}
    labels: List,
    bar_label: Dict[str, List[str]],   # e.g. {'Unit1 $\mathrm{g \cdot cm^3}$': ['AARD','MAE']}
    xlabel: str,
    output_path: str,
    width: float = 0.2,
    color_list: List = ['#03788C', '#F27457', '#03488C', 'indianred', 'steelblue'],
    figsize: Tuple[float, float] = (10.0, 10.0),
    axis_fontsize: float = 14,
    ylabel_fontsize: float = 16,
    xlabel_fontsize: float = 16,
    legend_fontsize: float = 10,
    bbox_to_anchor: Tuple[float, float] = (1.0, 1.0),
    x_rotation: float = 0.0,
    label_fontweight: Literal['normal', 'bold'] = 'normal',
    ):
    plt.rcParams['font.size'] = axis_fontsize
    plt.rcParams['mathtext.fontset'] = 'custom'
    fig = plt.figure(figsize = figsize, dpi = 300)

    num_bars = sum([len(i) for i in data.values()])
    x = np.arange(len(labels))

    i = 0
    for key, value in data.items():
        if i == 0:
            ax_1 = fig.add_subplot(111)
            ax_1.set_ylabel(key, fontsize = ylabel_fontsize, fontweight = label_fontweight)
            for idx, data in enumerate(value):
                ax_1.bar(
                    x = x - (num_bars - 1 - 2 * i) * width / 2,
                    height = data,
                    color = color_list[i],
                    width = width,
                    label = bar_label[key][idx]
                    )
                i += 1
        else:
            ax_2 = ax_1.twinx()
            ax_2.set_ylabel(key, fontsize = ylabel_fontsize, fontweight = label_fontweight)
            for idx, data in enumerate(value):
                ax_2.bar(
                    x = x - (num_bars - 1 - 2 * i) * width / 2,
                    height = data,
                    color = color_list[i],
                    width = width,
                    label = bar_label[key][idx]
                    )
                i += 1
    ax_1.set_xlabel(xlabel, fontsize = xlabel_fontsize, fontweight = label_fontweight)
    fig.legend(
        bbox_to_anchor = bbox_to_anchor,
        bbox_transform = ax_1.transAxes,
        fontsize = legend_fontsize
        )
    plt.xticks(x, labels = labels)
    for tick in ax_1.get_xticklabels():
        tick.set_rotation(x_rotation)
    plt.savefig(
        output_path,
        bbox_inches = 'tight',
        dpi = 300
        )
    plt.close()

def bar(
    *args: ArrayLike,
    labels: List[str],
    bar_label: List[str],
    ylabel: str,
    xlabel: str,
    output_path: str,
    width: float = 0.3,
    color_list: List = ['#03788C', '#F27457', '#03488C', 'indianred', 'steelblue'],
    figsize: Tuple = (10, 10),
    axis_fontsize: float = 14,
    ylabel_fontsize: float = 16,
    xlabel_fontsize: float = 16,
    legend_fontsize: float = 10,
    bbox_to_anchor: Tuple[float, float] = (1.0, 1.0),
    x_rotation: float = 0.0,
    label_fontweight: Literal['normal', 'bold'] = 'normal',
    ):
    plt.rcParams['font.size'] = axis_fontsize
    plt.rcParams['mathtext.fontset'] = 'custom'
    fig = plt.figure(figsize = figsize, dpi = 300)

    num_bars = len(args)
    x = np.arange(len(labels))

    ax_1 = fig.add_subplot(111)
    ax_1.set_ylabel(ylabel, fontsize = ylabel_fontsize, fontweight = label_fontweight)
    for i, data in enumerate(args):
        ax_1.bar(
            x = x - (num_bars - 1 - 2 * i) * width / 2,
            height = data,
            color = color_list[i],
            width = width,
            label = bar_label[i]
            )
    ax_1.set_xlabel(xlabel, fontsize = xlabel_fontsize, fontweight = label_fontweight)
    fig.legend(
        bbox_to_anchor = bbox_to_anchor,
        bbox_transform = ax_1.transAxes,
        fontsize = legend_fontsize
        )
    plt.xticks(x, labels = labels, rotation = x_rotation)
    plt.savefig(
        output_path,
        bbox_inches = 'tight',
        dpi = 300
        )
    plt.close()
