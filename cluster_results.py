import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from copy import deepcopy as dcopy
from utils.utils import join
import math
from utils.visualization import create_legend, fill_quadrants, \
    set_grid, plot_axis_lines, remove_ticklabels


import matplotlib as mpl
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms


mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.size'] = 20
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['mathtext.fontset'] = 'stix'
# mpl.rcParams['text.usetex'] = True

# mpl.rcParams.update(mpl.rcParamsDefault)
# import scienceplots
# plt.style.use('science')
# plt.style.use(['science','ieee'])

metric_to_id_dict = {'acc': 0, 'robacc': 1,
                     'anfr': 2, 'rnfr': 3,
                     'bnfr': 4, 'snfr': 5,
                     'custom': 6,
                     'cleanerr': 7, 'roberr': 8}

METRIC_TITLES = ['Clean Accuracy', 'Robust Accuracy',
                 'ANFR reduction', 'RNFR reduction',
                 'BNFR', 'SNFR', 'CUSTOM', 'Clean Error reduction', 'Adversarial Error reduction']

# METRIC_TITLES = ['C', 'R',
#                  'ANFR', 'RNFR',
#                  'BNFR', 'SNFR']
# TODO: qui usare \% serve solo con scienceplot ma senza quello non formatta
# METRIC_TITLES = [metric_title + ' (\%)' for metric_title in METRIC_TITLES]
METRIC_TITLES = [metric_title + ' (%)' for metric_title in METRIC_TITLES]

LOSS_NAMES = ['baseline', 'PCT', 'PCT-AT', 'RF-AT (Ours)']
COLORS = ['tab:grey', 'tab:red', 'tab:blue', 'tab:green']
MARKERS = ['o', 'v', '^', 'D']
# ALL_MARKERS = list(Line2D.markers.keys())
ALL_MARKERS = "ov^<>1234sP*XD"


def scatter_ft_results(ax, res_list,
                       x_metric,
                       y_metric,
                       loss_ids,
                       lines=False,
                       diff=False):
    """
    :param ax:
    :param res_list: list containing ndarrays. Each matrix has 6 columns,
    one for each metric in METRIC_TITLES
    :param x_metric: xaxis metric taken from the keys of the dict "metric_to_id_dict"
    :param y_metric: yaxis metric taken from the keys of the dict "metric_to_id_dict"
    :param loss_ids: id referring to the positions in "LOSS_NAMES"
    :param lines: set True to draw connecting lines between baseline and derived finetuned versions
    :return:

    Serve solo per fare un singolo plot scatter in un singolo axis
    """
    xmin, xmax = math.inf, 0
    ymin, ymax = math.inf, 0

    res_new = res_list[0]
    for loss_id in loss_ids:
        if lines:
            if 0 in loss_ids:  # if baseline is present
                if loss_id > 0:
                    for k in range(res_new.shape[0]):
                        ax.plot(
                            [res_new[k, metric_to_id_dict[x_metric]], res_list[loss_id][k, metric_to_id_dict[x_metric]]],
                            [res_new[k, metric_to_id_dict[y_metric]], res_list[loss_id][k, metric_to_id_dict[y_metric]]],
                            linestyle='dashed', linewidth=0.3,
                            alpha=0.7,
                            color=COLORS[loss_id])

        res_x = dcopy(res_list[loss_id][:, metric_to_id_dict[x_metric]])
        res_y = dcopy(res_list[loss_id][:, metric_to_id_dict[y_metric]])

        res_new_x = dcopy(res_new[:, metric_to_id_dict[x_metric]])
        res_new_y = dcopy(res_new[:, metric_to_id_dict[y_metric]])

        if diff:
            if loss_id == 0:
                continue
            res_x -= res_new_x
            res_y -= res_new_y
            if 'nfr' or 'err' in x_metric:
                res_x = -res_x
            if 'nfr' or 'err' in y_metric:
                res_y = -res_y

            plot_axis_lines(ax, alpha=0.3)

        # # TODO: accrocchio per marker diverso per ogni
        # for i, (x_i, y_i) in enumerate(zip(res_x, res_y)):
        #     ax.scatter(x_i, y_i,
        #                alpha=0.5, marker=ALL_MARKERS[i], s=40,
        #                color=COLORS[loss_id],
        #                label=LOSS_NAMES[loss_id])
        #

        # print(LOSS_NAMES[loss_id])
        # print(f"x: {x_metric}, y: {y_metric}")
        # print(res_x.mean(), res_y.mean())
        # print("")
        ax.scatter(res_x, res_y,
                   alpha=0.4, marker=MARKERS[loss_id], s=80,
                   color=COLORS[loss_id],
                   label=LOSS_NAMES[loss_id])
        ax.scatter(res_x.mean(), res_y.mean(), marker=MARKERS[loss_id],
                   color=COLORS[loss_id], s=120,
                   edgecolor='black',
                   label=LOSS_NAMES[loss_id])

        # Rescale the plot for improved visualization
        margin_perc = 0.05
        xmin_i, xmax_i = res_x.min(), res_x.max()
        ymin_i, ymax_i = res_y.min(), res_y.max()
        xmin = min(xmin, xmin_i)
        xmax = max(xmax, xmax_i)
        ymin = min(ymin, ymin_i)
        ymax = max(ymax, ymax_i)

    x_margin = math.ceil((xmax - xmin)*margin_perc)
    y_margin = math.ceil((ymax - ymin)*margin_perc)
    xmin -= x_margin
    xmax += x_margin
    ymin -= y_margin
    ymax += y_margin
    # ax.set_xscale('symlog', base=2)
    # ax.set_yscale('symlog', base=2)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def scatter_models(path, csv_fname,
                   loss_ids=(0, 1, 2, 3),
                   mx1='acc', my1='anfr',
                   mx2='robacc', my2='rnfr',
                   lines=True,
                   fig_fname='clusters'):
    # todo: DA FINIRE E DEBUGGARE
    df = pd.read_csv(join(path, csv_fname)).rename(columns={'Unnamed: 0': 'Models'})

    model_pairs = df['Models'].unique()
    old_models = np.array([int(mp.split('old-')[-1].split('_new-')[0]) for mp in model_pairs])
    new_models = np.array([int(mp.split('old-')[-1].split('_new-')[-1]) for mp in model_pairs])

    unique_old_models = np.unique(old_models)

    nrows, ncols = 1, len(unique_old_models)

    old_id = None
    new_id = None

    res_new = df.loc[df['Loss'] == 'new'].iloc[:, 2:].to_numpy()
    res_pct = df.loc[df['Loss'] == 'PCT'].iloc[:, 2:].to_numpy()
    res_pctat = df.loc[df['Loss'] == 'PCT-AT'].iloc[:, 2:].to_numpy()
    res_rfat = df.loc[df['Loss'] == 'MixMSE-AT'].iloc[:, 2:].to_numpy()

    res_list = [res_new, res_pct, res_pctat, res_rfat]

    fig, axs = plt.subplots(nrows, ncols, squeeze=False, figsize=(5 * ncols, 5 * nrows))

    if old_id is not None:
        assert isinstance(old_id, int)
        df = df.loc[df['Models'].str.contains(f"old-{old_id}")]  # Select old reference

    if new_id is not None:
        assert isinstance(new_id, int)
        df = df.loc[df['Models'].str.contains(f"new-{new_id}")]  # Select old reference

    for col_j, (x_metric, y_metric) in enumerate([(mx1, my1), (mx2, my2)]):
        scatter_ft_results(axs[0, col_j], res_list,
                           x_metric, y_metric, loss_ids, lines)

        axs[0, col_j].set_xlabel(METRIC_TITLES[metric_to_id_dict[x_metric]])
        axs[0, col_j].set_ylabel(METRIC_TITLES[metric_to_id_dict[y_metric]])

    fig.tight_layout()
    fig.show()

    fig.savefig(f"images/cluster_results/{fig_fname}.pdf")

    legend_fig = create_legend(ax=axs[0, 0])
    legend_fig.show()
    legend_fig.savefig(f"images/cluster_results/{fig_fname}_legend.pdf")


def scatter_methods(path, csv_fname, lines,
                mx1='acc', my1='anfr',
                mx2='robacc', my2='rnfr',
                loss_ids=(0, 1, 2, 3),
                fig_fname='clusters',
                diff=True):

    nrows, ncols = 2, len(loss_ids) - 1

    df = pd.read_csv(join(path, csv_fname)).rename(columns={'Unnamed: 0': 'Models'})

    res_new = df.loc[df['Loss'] == 'new'].iloc[:, 2:].to_numpy()
    res_pct = df.loc[df['Loss'] == 'PCT'].iloc[:, 2:].to_numpy()
    res_pctat = df.loc[df['Loss'] == 'PCT-AT'].iloc[:, 2:].to_numpy()
    res_rfat = df.loc[df['Loss'] == 'MixMSE-AT'].iloc[:, 2:].to_numpy()

    res_list = [res_new, res_pct, res_pctat, res_rfat]

    fig, axs = plt.subplots(nrows, ncols, squeeze=False, figsize=(5 * ncols, 5 * nrows))

    for col_j, loss_id in enumerate(loss_ids[1:]):
        if loss_id == 0:
            continue
        for row_i, (x_metric, y_metric) in enumerate([(mx1, my1), (mx2, my2)]):
            scatter_ft_results(ax=axs[row_i, col_j], res_list=res_list,
                               x_metric=x_metric, y_metric=y_metric,
                               lines=lines, diff=diff,
                               loss_ids=(0, loss_id))

            axs[row_i, col_j].set_xlabel(METRIC_TITLES[metric_to_id_dict[x_metric]])
            axs[row_i, col_j].set_ylabel(METRIC_TITLES[metric_to_id_dict[y_metric]])

        axs[0, col_j].set_title(LOSS_NAMES[loss_id])

    fig.tight_layout()
    fig.show()

    fig.savefig(f"images/cluster_results/{fig_fname}_{'diff' if diff else ''}_methods.pdf")

    # legend_fig = create_legend(ax=axs[0, 0])
    # legend_fig.show()
    # legend_fig.savefig(f"images/cluster_results/{fig_fname}_methods_{'diff' if diff else ''}_legend.pdf")


def scatter_all(path, csv_fname,
                lines=False,
                mx1='acc', my1='anfr',
                mx2='robacc', my2='rnfr',
                loss_ids=(0, 1, 2, 3),
                fig_fname='clusters',
                diff=False,
                color_quadrants=False,
                zoom=False):

    nrows, ncols = 1, 3 if zoom else 2

    # todo: accrocchio
    # df = pd.read_csv(join(path, csv_fname)).rename(columns={'Unnamed: 0': 'Models'})
    df = pd.read_csv(join(path, csv_fname), index_col=[0, 1], skipinitialspace=True)
    new_indexes = df.index.get_level_values(0).unique()[np.array([3, 12, 1, 4, 7, 6, 11, 5, 14, 2]) - 1]
    df = df.reindex(new_indexes, level=0)
    df = df.reset_index()

    # TODO: qui aggiungo altre metriche custom basate su combinazione delle altre
    df['mean'] = df[['NFR', 'R-NFR']].mean(axis=1)
    df['cleanerr'] = 100 - df['Acc']
    df['roberr'] = 100 - df['RobAcc']

    # model_pairs = df['Models'].unique()
    # old_models = [mp.split('old-')[-1].split('_new-')[0] for mp in model_pairs]
    # new_models = [mp.split('old-')[-1].split('_new-')[0] for mp in model_pairs]

    # TODO: qui bisogna dirgli esattamente quali colonne prendere, perchè se aggiungo altra roba si rompe il codice
    res_new = df.loc[df['Loss'] == 'new'].iloc[:, 3:].to_numpy()
    res_pct = df.loc[df['Loss'] == 'PCT'].iloc[:, 3:].to_numpy()
    res_pctat = df.loc[df['Loss'] == 'PCT-AT'].iloc[:, 3:].to_numpy()
    res_rfat = df.loc[df['Loss'] == 'MixMSE-AT'].iloc[:, 3:].to_numpy()

    res_list = [res_new, res_pct, res_pctat, res_rfat]

    fig, axs = plt.subplots(nrows, ncols, squeeze=False, figsize=(10 * ncols, 5 * nrows))

    for col_j, (x_metric, y_metric) in enumerate([(mx1, my1), (mx2, my2)]):

        scatter_ft_results(axs[0, col_j], res_list,
                           x_metric, y_metric, loss_ids, lines, diff)
        xlabel = METRIC_TITLES[metric_to_id_dict[x_metric]]
        ylabel = METRIC_TITLES[metric_to_id_dict[y_metric]]

        # if diff:
        #     xlabel = f"$\Delta$ {xlabel}"
        #     ylabel = f"$\Delta$ {ylabel}"


        axs[0, col_j].set_xlabel(xlabel)
        axs[0, col_j].set_ylabel(ylabel)

    if color_quadrants:
        # for ax in [axs[0, 0], axs[0, 1], axins]:
            # xmin, xmax = ax.get_xlim()
            # ymin, ymax = ax.get_ylim()
            # major_ticks = np.arange(math.floor(xmin) - 1, math.ceil(xmax) + 1, 10)
            # minor_ticks = np.arange(math.floor(ymin) - 1, math.ceil(ymax) + 1, 2)

        fill_quadrants(axs[0, 0])
        fill_quadrants(axs[0, 1])


    axs[0, 0].grid('on', linestyle='dashed')
    axs[0, 1].grid('on', linestyle='dashed')
    # set_grid(axs[0, 0], major_delta=5, linestyle_maj='dotted')
    # set_grid(axs[0, 1], major_delta=5, linestyle_maj='dotted')

    # # Inplot interno
    # axs[0, 1].set_xlim(xmax=10)
    # axs[0, 1].set_ylim(ymin=-80)
    #
    # xstart, ystart = .1, .01    # inset axes....
    # xend, yend = .85, .55

    # Inplot esterno
    # axs[0, 1].set_xlim(xmax=10)
    # axs[0, 1].set_ylim(ymin=-80)

    if zoom:
        xstart, ystart = .02, .02  # inset axes....

        xend, yend = 0.6, 0.55
        xdim = xend - xstart
        ydim = yend - ystart

        # xdim, ydim = 0.5, 0.5


        # axins = axs[0, 1].inset_axes([xstart, ystart,
        #                               xdim, ydim])

        axins = axs[0, 2]

        scatter_ft_results(axins, res_list,
                           x_metric, y_metric,
                           loss_ids, lines, diff)
        # subregion of the original image
        x1, x2, y1, y2 = -8, 1, -3, 6
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        # axins.xaxis.set_ticks_position('top')
        # axins.yaxis.set_ticks_position('right')

        # axins.tick_params(labelbottom=False, labeltop=False,
        #                   labelleft=False, labelright=False)
        # axs[0, 0].tick_params(labelbottom=False, labeltop=True)
        # axs[0, 1].tick_params(labelbottom=False, labeltop=True)

        axs[0, 1].indicate_inset_zoom(axins, edgecolor="black")

        if color_quadrants:
            fill_quadrants(axins)
        set_grid(axins, major_delta=1, linestyle_maj='dotted')

    fig.tight_layout()
    fig.show()

    # # Accrocchi per fare legend personalizzata
    # _, ax = plt.subplots(1,1)
    # for i, marker in enumerate(ALL_MARKERS):
    #     ax.scatter(0, 0, marker=marker, color='tab:grey', label=df['Models'].unique()[i])
    # legend_fig = create_legend(ax=ax, figsize=(15, 1), ncol=7)
    # # legend_fig = create_legend(ax=axs[0, 0], figsize=(11, 0.5))
    # legend_fig.tight_layout()
    # legend_fig.show()

    # _, ax = plt.subplots(1, 1)
    # for i, loss_color in enumerate(COLORS):
    #     ax.scatter(0,0, marker='o', color=loss_color, label=LOSS_NAMES[i])

    # legend_fig = create_legend(ax=ax, figsize=(11, 0.5))
    legend_fig = create_legend(ax=axs[0, 0], figsize=(11, 0.5))
    legend_fig.tight_layout()
    legend_fig.show()

    # fig.savefig(f"images/cluster_results/all_{fig_fname}{'_diff' if diff else ''}.pdf")
    fig.savefig(f"images/cluster_results/all_{fig_fname}{'_diff' if diff else ''}_err.pdf",
                bbox_inches=mtransforms.Bbox([[0, 0], [0.5, 1]]).transformed(
                    (fig.transFigure - fig.dpi_scale_trans))
                )
    fig.savefig(f"images/cluster_results/all_{fig_fname}{'_diff' if diff else ''}_nfr.pdf",
                bbox_inches=mtransforms.Bbox([[0.5, 0], [1, 1]]).transformed(
                    (fig.transFigure - fig.dpi_scale_trans))
                )
    legend_fig.savefig(f"images/cluster_results/all_{fig_fname}{'_diff' if diff else ''}_legend.pdf")

    # fig.savefig(f"images/cluster_results/all_{fig_fname}{'_diff' if diff else ''}.png", dpi=300)
    # legend_fig.savefig(f"images/cluster_results/all_{fig_fname}{'_diff' if diff else ''}_legend.png", dpi=300)


if __name__ == '__main__':
    PATH = r'results\day-30-03-2023_hr-10-01-01_PIPELINE_50k_3models'
    CSV_FNAME = 'model_results_test_with_val_criteria-S-NFR.csv'
    lines = False
    diff = True
    loss_ids = (0, 1, 2, 3)
    color_quadrants = False
    zoom = False

    # mx1, my1 = 'cleanerr', 'anfr',
    # mx2, my2 = 'roberr', 'rnfr'

    # mx1, my1 = 'acc', 'snfr',
    # mx2, my2 = 'robacc', 'snfr'

    # mx1, my1 = 'robacc', 'acc'#, 'robacc',
    # mx2, my2 = 'rnfr', 'anfr'#, 'rnfr'

    # BEST
    mx1, my1 = 'roberr', 'cleanerr'  # , 'robacc',
    mx2, my2 = 'rnfr', 'anfr'  # , 'rnfr'

    # mx1, my1 = 'acc', 'custom',
    # mx2, my2 = 'robacc', 'custom'

    fig_fname = 'scatter_plot_metrics'

    scatter_all(path=PATH, csv_fname=CSV_FNAME, lines=lines,
                loss_ids=loss_ids,
                mx1=mx1, my1=my1,
                mx2=mx2, my2=my2,
                diff=diff,
                fig_fname=fig_fname,
                color_quadrants=color_quadrants,
                zoom=zoom)

    # scatter_methods(path=PATH, csv_fname=CSV_FNAME, lines=lines,
    #             mx1='acc', my1='anfr',
    #             mx2='robacc', my2='rnfr',
    #             diff=diff)

    # scatter_models(path=PATH, csv_fname=CSV_FNAME, lines=True)
    print("")



