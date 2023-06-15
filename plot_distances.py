import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils.eval import compute_nflips, compute_common_nflips
from utils.visualization import create_legend


import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import scienceplots
plt.style.use('science')
# plt.style.use(['science','ieee'])
mpl.rcParams['font.size'] = 15

OLD_IDS = [1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 5, 5, 6]
NEW_IDS = [4, 7, 4, 5, 7, 2, 4, 5, 6, 7, 7, 4, 7, 7]
EPS = 0.03

def plot_distance_churn_distrib(ax, dold, old_corr_clean, new_corr_clean,
                                    old_corr_adv, new_corr_adv):
    anf_mask = compute_nflips(old_preds=old_corr_clean, new_preds=new_corr_clean, indexes=True)
    rnf_mask = compute_nflips(old_preds=old_corr_adv, new_preds=new_corr_adv, indexes=True)
    print("--------------------")
    print(f"ANF: {anf_mask.mean()}")
    print(f"RNF: {rnf_mask.mean()}")
    anf_mask, rnf_mask, bnf_mask = compute_common_nflips(clean_nf_idxs=anf_mask, advx_nf_idxs=rnf_mask, indexes=True)
    print(f"BNF: {bnf_mask.mean()}")

    ax.axvline(x=1, linestyle='dashed', color='k', label='$\\epsilon$')

    COLORS = ['g', 'r', 'b']
    MARKERS = ['^', 'v', 'o']
    HATCHES = ['//', '\\\\', 'oo']
    LABELS = ["$ANF \setminus BNF$", '$RNF \setminus BNF$', '$BNF$']
    alpha = 0.6

    for i, (label, nf) in enumerate(zip(LABELS, (anf_mask, rnf_mask, bnf_mask))):
        LABELS[i] = f"{label} = {nf.mean()}"

    for k, mask in enumerate([anf_mask, rnf_mask, bnf_mask]):
        ax.hist(dold[mask],
                color=COLORS[k], label=LABELS[k], hatch=HATCHES[k],
                alpha=alpha)

def main():
    file_path = 'results/distance_results/base_distances.gz'
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    nrows, ncols = 2, 7
    fdim = 3
    fig, axs = plt.subplots(nrows, ncols, figsize=(fdim*ncols, fdim*nrows))

    for plot_i, (old_id, new_id) in enumerate(zip(OLD_IDS, NEW_IDS)):
        i, j = plot_i // ncols, plot_i % ncols
        ax = axs[i, j]

        dold = data['distances'][old_id - 1] / EPS
        m_old_name = data['model_names'][old_id - 1]

        old_corr_clean = data['clean'][old_id - 1]
        new_corr_clean = data['clean'][new_id - 1]
        old_corr_adv = data['adv'][old_id - 1]
        new_corr_adv = data['adv'][new_id - 1]

        plot_distance_churn_distrib(ax, dold, old_corr_clean, new_corr_clean,
                                    old_corr_adv, new_corr_adv)

        # ax.legend(loc='best')
        ax.set_title(f"$M_{old_id} \\rightarrow M_{new_id}$")
        ax.set_ylabel('count')
        # ax.set_xlabel(f"$d(x, M_{old_id})$")
        # ax.set_xlabel(f"$d(x, M)/eps$")

        ax.set_ylim(0, 50)
        ax.set_xlim(0, 5)

        ax.set_xticks(list(range(5)))
        ax.set_xticklabels(['0'] + [f"${i}\\epsilon$" if i > 1 else "$\\epsilon$" for i in range(1, 5)])


        print("")

    fig.tight_layout()
    legend_fig = create_legend(axs[0, 0])
    fig.show()
    legend_fig.show()

    fig_fname = 'images/distances/distances'
    fig.savefig(f"{fig_fname}.pdf")
    legend_fig.savefig(f"{fig_fname}_legend.pdf")

    print("")


def main_ft():
    file_paths = [f"results/distance_results/base_distances{s}.gz"
                  for s in ['', '_PCT_ft', '_PCT-AT_ft', '_MixMSE-AT_ft']]

    sel_ids = list(range(len(OLD_IDS)))
    data_list = []
    for file_path in file_paths:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        data_list.append(data)

    nrows, ncols = len(sel_ids), 4
    fdim = 6
    fig, axs = plt.subplots(nrows, ncols, figsize=(fdim*ncols, fdim*nrows))

    loss_names = ('baseline', 'PCT', 'PCT-AT', 'RF-AT')

    for j, (loss_name, data) in enumerate(zip(loss_names, data_list)):
        print("---------------------------")
        print(f"Loss: {loss_name}")
        # cycle for model pairs old-new
        for i, (old_id, new_id) in enumerate(zip(OLD_IDS, NEW_IDS)):
            ax = axs[i, j]
            # cycle for new, PCT, PCT-AT e RF-AT
            dold = data_list[0]['distances'][old_id - 1] / EPS

            old_corr_clean = data_list[0]['clean'][old_id - 1]
            new_corr_clean = data['clean'][new_id - 1]
            old_corr_adv = data_list[0]['adv'][old_id - 1]
            new_corr_adv = data['adv'][new_id - 1]

            plot_distance_churn_distrib(ax, dold, old_corr_clean, new_corr_clean,
                                        old_corr_adv, new_corr_adv)
            ax.legend()

            # ax.legend(loc='best')
            if j == 0:
                ax.set_ylabel(f"$M_{old_id} \\rightarrow M_{new_id}$")
            # ax.set_xlabel(f"$d(x, M_{old_id})$")
            # ax.set_xlabel(f"$d(x, M)/eps$")

            # ax.set_ylim(0, 50)
            # ax.set_xlim(0, 5)

            ax.set_xticks(list(range(5)))
            ax.set_xticklabels(['0'] + [f"${i}\\epsilon$" if i > 1 else "$\\epsilon$" for i in range(1, 5)])

    for j, loss_name in enumerate(loss_names):
        axs[0, j].set_title(loss_name)

    fig.tight_layout()
    legend_fig = create_legend(axs[0, 0])
    fig.show()
    legend_fig.show()

    fig_fname = 'images/distances/distances_ftuned'
    fig.savefig(f"{fig_fname}.pdf")
    legend_fig.savefig(f"{fig_fname}_legend.pdf")

    print("")

if __name__ == '__main__':
    main_ft()