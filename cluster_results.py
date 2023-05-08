import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from utils.utils import join
import math

import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['text.usetex'] = True

path = r'results\day-30-03-2023_hr-10-01-01_PIPELINE_50k_3models'
csv_fname = 'model_results_test_with_val_criteria-S-NFR.csv'

df = pd.read_csv(join(path, csv_fname)).rename(columns={'Unnamed: 0': 'Models'})

idxs = {'acc': 0, 'robacc': 1,
        'anfr': 2, 'rnfr': 3,
        'bnfr': 4, 'snfr': 5}

metric_titles = ['Clean Accuracy', 'Robust Accuracy',
                 'ANFR', 'RNFR',
                 'BNFR', 'SNFR']
metric_titles = [metric_title + ' (%)' for metric_title in metric_titles]
loss_names = ['baseline', 'PCT', 'PCT-AT', 'RF-AT']


res_new = df.loc[df['Loss'] == 'new'].iloc[:, 2:].to_numpy()
res_pct = df.loc[df['Loss'] == 'PCT'].iloc[:, 2:].to_numpy()
res_pctat = df.loc[df['Loss'] == 'PCT-AT'].iloc[:, 2:].to_numpy()
res_rfat = df.loc[df['Loss'] == 'MixMSE-AT'].iloc[:, 2:].to_numpy()

res_list = [res_new, res_pct, res_pctat, res_rfat]

nrows = 1
ncols = 2

fig, axs = plt.subplots(nrows, ncols, squeeze=False, figsize=(5*ncols, 5*nrows))

loss_ids = [0, 1, 2, 3]
colors = ['grey', 'red', 'blue', 'green']
markers = ['o', 'v', '^', 'D']

for col_j, (x_metric, y_metric) in enumerate([('acc', 'robacc'), ('anfr', 'rnfr')]):
    # x_metric = 'anfr'
    # y_metric = 'rnfr'

    xmin, xmax = math.inf, 0
    ymin, ymax = math.inf, 0

    for loss_id in loss_ids:
        if 0 in loss_ids:     # if baseline is present
            if loss_id > 0:
                for k in range(res_new.shape[0]):
                    axs[0, col_j].plot([res_new[k, idxs[x_metric]], res_list[loss_id][k, idxs[x_metric]]],
                                       [res_new[k, idxs[y_metric]], res_list[loss_id][k, idxs[y_metric]]],
                                       linestyle='dashed', linewidth=0.3,
                                       alpha=0.7,
                                       color=colors[loss_id])

        res_x = res_list[loss_id][:, idxs[x_metric]]
        res_y = res_list[loss_id][:, idxs[y_metric]]

        xmin_i, xmax_i = res_x.min(), res_x.max()
        ymin_i, ymax_i = res_y.min(), res_y.max()

        xmin = min(xmin, xmin_i)
        xmax = max(xmax, xmax_i)
        ymin = min(ymin, ymin_i)
        ymax = max(ymax, ymax_i)

        axs[0, col_j].scatter(res_x, res_y,
                              alpha=0.7, marker=markers[loss_id], s=40,
                              color=colors[loss_id], label=loss_names[loss_id])

    # axs[0, 0].scatter(res_new[:, 0], res_pctat[:, 0], alpha=0.7,
    #                   marker='v', color='blue', label='PCT-AT')
    # axs[0, 0].scatter(res_new[:, 0], res_rfat[:, 0], alpha=0.7,
    #                   marker='D', color='green', label='RF-AT')

    # axs[0, 0].plot([0, 100], [0, 100], 'k--')

    axs[0, col_j].set_xlabel(metric_titles[idxs[x_metric]])
    axs[0, col_j].set_ylabel(metric_titles[idxs[y_metric]])

    margin_perc = 0.05
    x_margin = math.ceil((xmax - xmin)*margin_perc)
    y_margin = math.ceil((ymax - ymin)*margin_perc)
    axs[0, col_j].set_xlim(xmin - x_margin, xmax + x_margin)
    axs[0, col_j].set_ylim(ymin - y_margin, ymax + y_margin)

fig.tight_layout()
fig.show()

fig_fname = "clusters"

fig.savefig(f"images/cluster_results/{fig_fname}.pdf")

# create legend
h, l = axs[0, 0].get_legend_handles_labels()
legend_dict = dict(zip(l, h))
legend_fig = plt.figure(figsize=(10, 0.5))

legend_fig.legend(legend_dict.values(), legend_dict.keys(), loc='center',
                  ncol=len(legend_dict.values()), frameon=False)
legend_fig.tight_layout()
legend_fig.show()

# axs[0, col_j].legend(loc='best')


legend_fig.savefig(f"images/cluster_results/{fig_fname}_legend.pdf")
print("")



