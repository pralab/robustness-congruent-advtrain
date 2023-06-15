from utils.utils import my_load, my_save, \
    get_model_info, select_group
from utils.eval import compute_nflips, compute_common_nflips
from utils.data import get_cifar10_dataset
from utils.visualization import imshow, create_legend, set_grid
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import scienceplots
plt.style.use('science')
# plt.style.use(['science','ieee'])
mpl.rcParams['font.size'] = 15


NF_NAMES = ['ANF', 'RNF', 'BNF']


def get_nf_matrices(data, old_ids, new_ids):
    anf_matrix = np.zeros(shape=(len(old_ids), 2000))
    rnf_matrix = np.zeros(shape=(len(old_ids), 2000))
    bnf_matrix = np.zeros(shape=(len(old_ids), 2000))

    for i, (old_id, new_id) in enumerate(zip(old_ids, new_ids)):
        # m_old_name = data['model_names'][old_id - 1]
        # m_new_name = data['model_names'][new_id - 1]

        old_corr_clean = data['clean'][old_id - 1]
        new_corr_clean = data['clean'][new_id - 1]
        old_corr_adv = data['adv'][old_id - 1]
        new_corr_adv = data['adv'][new_id - 1]

        anf_mask = compute_nflips(old_preds=old_corr_clean, new_preds=new_corr_clean, indexes=True)
        rnf_mask = compute_nflips(old_preds=old_corr_adv, new_preds=new_corr_adv, indexes=True)
        anf_mask, rnf_mask, bnf_mask = compute_common_nflips(clean_nf_idxs=anf_mask, advx_nf_idxs=rnf_mask,
                                                             indexes=True)

        anf_matrix[i, :] = anf_mask.to_numpy()
        rnf_matrix[i, :] = rnf_mask.to_numpy()
        bnf_matrix[i, :] = bnf_mask.to_numpy()
    nf_matrices = anf_matrix, rnf_matrix, bnf_matrix
    return nf_matrices


def compute_flip_counts(nf_matrices):
    anf_matrix, rnf_matrix, bnf_matrix = nf_matrices
    anf_sample_count = anf_matrix.sum(axis=0)
    anfs_idxs = np.flip(anf_sample_count.argsort())
    anf_sample_count = anf_sample_count[anfs_idxs]

    rnf_sample_count = rnf_matrix.sum(axis=0)
    rnfs_idxs = np.flip(rnf_sample_count.argsort())
    rnf_sample_count = rnf_sample_count[rnfs_idxs]

    bnf_sample_count = bnf_matrix.sum(axis=0)
    bnfs_idxs = np.flip(bnf_sample_count.argsort())
    bnf_sample_count = bnf_sample_count[bnfs_idxs]

    sample_counts = (anf_sample_count, rnf_sample_count, bnf_sample_count)
    idxs = (anfs_idxs, rnfs_idxs, bnfs_idxs)

    return sample_counts, idxs


def plot_frequent_flips(sample_counts, fig_path=None):
    # anf_sample_count, rnf_sample_count, bnf_sample_count = sample_counts
    alpha = 0.7
    linewidth = 2
    fdim = 5
    fig, ax = plt.subplots(figsize=(fdim, fdim))

    linestyles = ['dotted', 'dashed', 'dashdot']

    for i, (nf_name, nf_sample_count) in enumerate(zip(NF_NAMES, sample_counts)):
        # Non Zero values
        nz_nf = (nf_sample_count != 0).sum()

        # xmax = max(nz_anf, nz_rnf, nz_bnf)
        xmax = 1000
        x = np.arange(xmax)
        ax.plot(x, nf_sample_count[:xmax],
                linestyle=linestyles[i], linewidth=linewidth,
                label=f"{nf_name}, {nz_nf} NZ values", alpha=alpha)
        ax.legend(loc='best')
        ax.set_xlabel('\# sample')
        ax.set_ylabel('\# occurrences')

    # ax.set_xscale('log')

    fig.show()

    if fig_path is not None:
        fig.savefig(fig_path)

        ax.set_xscale('log')
        fig.show()
        fig.savefig(fig_path.replace('.', '_log.'))


def plot_distance_vs_flipfreq(distances, sample_counts, idxs,
                              old_ids, new_ids, group_id=None,
                              fig_fname=None):
    old_ids, new_ids = select_group(old_ids, new_ids, group_id)
    distances = np.array([list(distances[i]) for i in old_ids])

    eps = 0.031
    n_eps = 3
    nz_slots = 10e3
    log_scale = True

    linewidth = 1.2
    linewidth_v = 1.5
    c_min_max = '#A6D0DD'
    c_std = '#146C94'
    c_mean = '#FF6969'

    fdim = 5
    nrows, ncols = 2, 3
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fdim*ncols, fdim*nrows), squeeze=False)

    id_dist = 1
    id_count = 0

    # DISTANCE DISTRIB VS NSAMPLES (sorted according plot above)
    for j, (nf_sample_count, nf_idxs, nf_name) in enumerate(zip(sample_counts, idxs, NF_NAMES)):
        # Sort the distance value according to the frequency of NF
        nf_sorted_distances = distances[:, nf_idxs]
        # Before <nz_values> element we have all samples that flip at least in 1 over 14 model update configurations
        nz_values = np.where(nf_sample_count == 0)[0][0]
        # counts, idxs, n_samples = np.unique(nf_sample_count, return_index=True, return_counts=True)

        nf_dist_mean = nf_sorted_distances.mean(axis=0)
        nf_dist_std = nf_sorted_distances.std(axis=0)
        nf_dist_min = nf_sorted_distances.min(axis=0)
        nf_dist_max = nf_sorted_distances.max(axis=0)

        data = np.stack([nf_dist_mean,
                         nf_dist_std,
                         nf_dist_min,
                         nf_dist_max,
                         nf_sample_count]
                        )

        df = pd.DataFrame(data=data.T, columns=['mean', 'std', 'min', 'max', 'count'])
        # df = df[:nz_values*nz_slots]
        # df = df[:nz_slots]
        x = np.arange(df.shape[0])
        # ax = axs[1, j]



        ##############################
        # PLOT COUNTS
        ##############################

        axs[id_count, j].plot(x, df['count'],
                linestyle='solid', linewidth=linewidth,
                label=f"{nf_name}, {nz_values} NZ values")

        # Plot limit line of flipping samples in both rows
        axs[id_count, j].axvline(x=nz_values, linestyle='dashed', linewidth=linewidth_v,
                   color='grey', label=f"{nz_values} NF")

        if log_scale:
            axs[id_count, j].set_xscale('log')
            axs[id_count, j].set_xlim(xmin=1e-1)

        axs[id_count, j].set_ylim(ymax=10)
        axs[id_count, j].grid(which='major', alpha=0.7, zorder=-100)



        ##############################
        # PLOT DISTANCES
        ##############################

        # ax.plot(1/nf_dist_mean, color='b', alpha=0.7)
        # sns.tsplot(data=df.T.values, ax=ax)
        window = max(5, nz_values // 10)

        def roll(df, window=10):
            first_value = df[0]
            df = df.rolling(window, closed='left', min_periods=1).mean()
            df[0] = first_value
            return df

        # todo: qui per il roll basta un for su tutte le colonne tranne 'count'
        axs[id_dist, j].fill_between(x,
                        roll(df['max'], window),
                        roll(df['min'], window),
                        color=c_min_max, alpha=0.5)
        axs[id_dist, j].fill_between(x,
                        roll(df['mean'], window) + roll(df['std'], window),
                        roll(df['mean'], window) - roll(df['std'], window),
                        color=c_std, alpha=0.7)
        axs[id_dist, j].plot(x, roll(df['mean']), color=c_mean, linewidth=linewidth)

        # Plot limit line of flipping samples in both rows
        axs[id_dist, j].axvline(x=nz_values, linestyle='dashed', linewidth=linewidth_v,
                   color='grey', label=f"last NF")

        axs[id_dist, j].axhline(y=eps, linestyle='dotted', linewidth=linewidth_v,
                   color='red', label=f"$\\epsilon$")

        # REFINEMENT OF AXIS
        axs[id_dist, j].set_ylim(0, 0.1)
        axs[id_dist, j].set_xlabel('x (sorted for NF frequency)')
        yticks = np.arange(0, (n_eps + 1)*eps, eps)
        ytickslabels = ['0'] + [f"${i}\\epsilon$" if i > 1 else "$\\epsilon$" for i in range(1, (n_eps + 1))]
        axs[id_dist, j].set_yticks(yticks)
        axs[id_dist, j].set_yticklabels(ytickslabels)
        if log_scale:
            axs[id_dist, j].set_xscale('log')
            axs[id_dist, j].set_xlim(xmin=1e-1)
        # axs[id_dist, j].legend(loc='best')

        # xtickslabels = ['0'] + [f"${i}\\epsilon$" if i > 1 else "$\\epsilon$" for i in range(1, (n_eps + 1))]


        axs[id_dist, j].grid(which='major', alpha=0.7, zorder=-100)

        print("")



    for j, nf_name in enumerate(NF_NAMES):
        axs[0, j].set_title(nf_name)

    axs[id_count, 0].set_ylabel('\# occurrences')
    axs[id_dist, 0].set_ylabel('avg distance to model\'s boundary')

    fig.tight_layout()
    fig.show()

    legend_fig = create_legend(axs[1, 0], figsize=(10, 0.5))
    legend_fig.show()

    if log_scale:
        fig_fname = fig_fname.replace('.pdf', '_log.pdf')

    if isinstance(fig_fname, str):
        fig.savefig(fig_fname)
        legend_fig.savefig(fig_fname.replace('.pdf', '_legend.pdf'))

    print("")


def show_flipped_images(nf_idxs, nf_counts=None, n_images=10, fig_path=None):
    test_dataset = get_cifar10_dataset(train=False, shuffle=False, num_samples=2000)
    fdim = 3
    nrows, ncols = len(nf_idxs), n_images
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fdim*ncols, fdim*nrows))
    for i, (nf_idx, nf_count) in enumerate(zip(nf_idxs, nf_counts)):
        for j in range(n_images):
            idx = nf_idx[j]
            img = test_dataset[idx][0]
            imshow(img, ax=axs[i, j])

            axs[i, j].axes.get_xaxis().set_ticks([])
            axs[i, j].axes.get_yaxis().set_ticks([])
            axs[i, j].set_title(f"count = {int(nf_count[j])}")

        axs[0, 0].set_ylabel(NF_NAMES[i])
    fig.tight_layout()
    fig.show()

    if fig_path is not None:
        fig.savefig(fig_path)


# def split_archit(nf_matrices, nf_idxs, nf_counts=None, fig_path=None):
#     fdim = 3
#     nrows, ncols = len(nf_idxs), n_images
#     fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fdim * ncols, fdim * nrows))
#     for i, (nf_idx, nf_count) in enumerate(zip(nf_idxs, nf_counts)):
#         for j in range(n_images):
#             idx = nf_idx[j]
#             if nf_counts is not None:
#                 axs[i, j].set_title(f"count = {int(nf_count[j])}")
#
#     axs[0, 0].set_ylabel('ANF')
#     axs[1, 0].set_ylabel('RNF')
#     axs[2, 0].set_ylabel('BNF')
#     fig.tight_layout()
#     fig.show()
#
#     if fig_path is not None:
#         fig.savefig(fig_path)


def main():
    path = 'results/distance_results/base_distances.gz'
    fcounts_path = 'images/sample_wise_analysis/counts.pdf'
    fdist_vs_freq_path = 'images/sample_wise_analysis/dist_vs_freq.pdf'
    fimg_path = 'images/sample_wise_analysis/images.pdf'

    old_ids = [1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 5, 5, 6]
    new_ids = [4, 7, 4, 5, 7, 2, 4, 5, 6, 7, 7, 4, 7, 7]



    data = my_load(path)
    nf_matrices = get_nf_matrices(data, old_ids, new_ids)

    sample_counts, idxs = compute_flip_counts(nf_matrices)

    # Model specific analysis
    models_info = get_model_info(path='models/model_info/cifar10/Linf')

    plot_distance_vs_flipfreq(distances=data['distances'], sample_counts=sample_counts,
                              idxs=idxs, old_ids=old_ids, new_ids=new_ids,
                              group_id=None, fig_fname=fdist_vs_freq_path)

    # plot_frequent_flips(sample_counts, fig_path=fcounts_path)
    #
    # # todo: Qui sarebbe carino mostrare classe vera (predetta da old) e classe dopo
    # #  magari ANF sono più plausibili di RNF
    # show_flipped_images(nf_idxs=idxs,
    #                     nf_counts=sample_counts,
    #                     n_images=10,
    #                     fig_path=fimg_path)



    print("")



if __name__ == '__main__':
    main()
