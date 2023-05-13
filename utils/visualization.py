from torchvision.transforms import Normalize
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from secml.utils import fm
import os
import pandas as pd
import torch
import math
import pickle
from matplotlib.patches import Rectangle
from utils.utils import str_to_hps

class InvNormalize(Normalize):
    def __init__(self, normalizer):
        inv_mean = [-mean / std for mean, std in list(zip(normalizer.mean, normalizer.std))]
        inv_std = [1 / std for std in normalizer.std]
        super().__init__(inv_mean, inv_std)

def _tensor_to_show(img, transforms=None):
    if transforms is not None:
        for transform in transforms.transforms:
            if isinstance(transform, Normalize):
                normalizer = transform
                break
        inverse_transform = InvNormalize(normalizer)
        img = inverse_transform(img)

    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    return npimg

def imshow(img, transforms=None, figsize=(10, 20), path=None):
    npimg = _tensor_to_show(img, transforms)
    plt.figure(figsize=figsize)
    plt.imshow(npimg, interpolation=None)
    if path is not None:
        plt.savefig(path)



def show_batch(x, transforms=None, figsize=(10, 20)):
    imshow(make_grid(x.cpu().detach(), nrow=5),
           transforms=transforms, figsize=figsize)
    plt.axis('off')
    plt.show()

def show_loss_from_csv_to_filefig(csv_path, fig_path):
    df = pd.read_csv(csv_path, index_col='epoch')
    df.plot(kind='line')
    plt.savefig(fig_path)

    print("")


def my_plot_decision_regions(model, samples, targets, device='cpu',
                             flipped_samples=None, adv_flipped_samples=None,
                             adv_correct=None,
                             ax=None, n_grid_points=100,
                             fname=None,  x_adv=None,
                             eps=1):
    min = torch.min(samples, axis=0)[0] - 1
    max = torch.max(samples, axis=0)[0] + 1
    min = min*0
    max = min+1

    n_points_per_dim = math.floor(math.sqrt(n_grid_points))
    x = np.linspace(min[0], max[0], n_points_per_dim)
    y = np.linspace(min[1], max[1], n_points_per_dim)
    xx, yy = np.meshgrid(x, y)
    map_points = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])
    model.to(device)
    map_points = map_points.to(device)
    model.eval()
    outs = model(map_points)
    preds = outs.argmax(axis=1)
    Z = preds.reshape(xx.shape)

    # COLOR_LIST = ['r', 'g', 'y', 'b', 'c'] #todo: se idx eccede lista problem
    # color_fun = lambda idx: COLOR_LIST[idx]
    # tz = Z.numpy().astype(int)
    # color_regions_idxs = list(map(color_fun, tz))
    # ts = list(targets.numpy().astype(int))
    # color_samples_idxs = list(map(color_fun, ts))
    cmap = plt.cm.hsv

    if ax is None:
        fig, ax = plt.subplots()
        title = ('Decision Regions')

    levels = [-1] + Z.unique().tolist()
    color_list = ['r', 'g', 'b', 'y']
    ax.contourf(xx, yy, Z.cpu(),
                # cmap=cmap,
                levels=levels,
                colors=color_list,
                alpha=0.3)

    c_list = [color_list[t] for t in targets]

    if flipped_samples is None:
        alpha_idx = 1
        s_idx = 40

        # ax.scatter(samples.cpu().numpy()[:, 0], samples.numpy()[:, 1],
        #            c=targets.cpu().int().numpy(), alpha=0.7,
        #            cmap=cmap, s=20, edgecolors='k')
    else:
        alpha_idx = flipped_samples*1
        alpha_idx[~flipped_samples] = 0.7
        s_idx = flipped_samples*80
        s_idx[~flipped_samples] = 40
        x_list = samples.cpu().numpy()[:, 0]
        y_list = samples.numpy()[:, 1]


    if x_adv is not None:
        ax.scatter(x_adv.numpy()[:, 0], x_adv.numpy()[:, 1],
                   c=c_list, alpha=alpha_idx, cmap=cmap,
                   s=s_idx, edgecolors='k', marker='v')
        print("")

    ax.scatter(x_list, y_list,
               c=c_list, alpha=alpha_idx, cmap=cmap,
               s=s_idx, edgecolors='k')

    if adv_correct is not None:
        colors = np.array(['k'] * adv_correct.shape[0])
        colors[adv_correct] = 'r'

        # fig, ax = plt.subplots(1, 1)
        # ax.scatter(samples.numpy()[:, 0], samples.numpy()[:, 1])

        for i, (x, y) in enumerate(samples.numpy()):
            linestyle = '--'
            linewidth = 1
            if adv_flipped_samples is not None:
                if adv_flipped_samples[i]:
                    linestyle = '-'
                    linewidth = 2
            color = colors[i]
            ax.add_patch(Rectangle((x-eps, y-eps), eps*2, eps*2,
                                   edgecolor=color,
                                   linestyle=linestyle,
                                   linewidth=linewidth,
                                   facecolor='none'))







    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # ax.axis('equal')
    # ax.set_xticks(())
    # ax.set_yticks(())
    # ax.legend()
    # fig.show()

    # if fname is not None:
    #     plt.savefig(f'images/{fname}.png')
    # plt.show()


def plot_loss(loss, ax, window=20):
    loss_df = pd.DataFrame(loss)

    if len(loss['tot']) < window*10:
        window = 1
    
    if isinstance(window, int):
        loss_df = loss_df.rolling(window).mean()

    loss_df.plot(ax=ax)
    ax.legend(fontsize=15)
    ax.set_xlabel('iterations')


def show_hps_behaviour(root, fig_path=None, axs=None):
    results_list = []
    for path, dirs, files in os.walk(root):
        if 'results_last.gz' in files:
            with open(os.path.join(path, 'results_last.gz'), 'rb') as f:
                results = pickle.load(f)

            if "\\" in path:
                hps = str_to_hps(path.split('\\')[-1])
            else:
                hps = str_to_hps(path.split('/')[-1])
            hps['results'] = results
            
            results_list.append(hps)

    n_rows = 1
    n_cols = 4
    
    if axs is None:
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*5))

    betas = [r['beta'] for r in results_list]
    results_list = [x for _, x in sorted(zip(betas, results_list))]

    alphas, betas = [], []
    accs, rob_accs = [], []
    nfrs, rob_nfrs = [], []
    for result_dict in results_list:
        alpha = result_dict['alpha']
        beta = result_dict['beta']
        alphas.append(alpha)
        betas.append(beta)

        # extract performances on clean data
        results = result_dict['results']
        acc = results['clean']['new_acc']
        old_acc = results['clean']['old_acc']
        orig_acc = results['clean']['orig_acc']
        nfr = results['clean']['nfr']
        orig_nfr = results['clean']['orig_nfr']

        # extract performances on advx data
        rob_acc = results['advx']['new_acc']
        rob_old_acc = results['advx']['old_acc']
        rob_orig_acc = results['advx']['orig_acc']
        rob_nfr = results['advx']['nfr']
        rob_orig_nfr = results['advx']['orig_nfr']

        accs.append(acc)
        rob_accs.append(rob_acc)
        nfrs.append(nfr)
        rob_nfrs.append(rob_nfr)

    # Accuracy
    axs[0].axhline(y=old_acc, color='k', linestyle='--', label='Acc(M0)')
    axs[0].axhline(y=orig_acc, color='r', linestyle='--', label='Acc(M1)')
    axs[0].plot(betas, accs, color='b', marker='o', label='Acc(M1+)')
    axs[0].set_xlabel('Beta')
    axs[0].set_title('Accuracy')
    axs[0].set_ylim(0.8, 1)
    axs[0].legend()

    # Robust Accuracy
    axs[1].axhline(y=rob_old_acc, color='k', linestyle='--', label='Rob-Acc(M0)')
    axs[1].axhline(y=rob_orig_acc, color='r', linestyle='--', label='Rob-Acc(M1)')
    axs[1].plot(betas, rob_accs, color='b', marker='o', label='Rob-Acc(M1+)')
    axs[1].set_xlabel('Beta')
    axs[1].set_title('Robust Accuracy')
    axs[1].set_ylim(0.5, 0.7)
    axs[1].legend()

    # Negative Flips
    axs[2].axhline(y=orig_nfr, color='r', linestyle='--', label='NFR(M1)')
    axs[2].plot(betas, nfrs, color='b', marker='o', label='NFR(M1+)')
    axs[2].set_xlabel('Beta')
    axs[2].set_title('Negative Flips')
    axs[2].set_ylim(0, 0.08)
    axs[2].legend()

    # Robust Negative Flips
    axs[3].axhline(y=rob_orig_nfr, color='r', linestyle='--', label='Rob-NFR(M1)')
    axs[3].plot(betas, rob_nfrs, color='b', marker='o', label='Rob-NFR(M1+)')
    axs[3].set_xlabel('Beta')
    axs[3].set_title('Robust Negative Flips')
    axs[3].set_ylim(0, 0.08)
    axs[3].legend()
    
    if (fig_path is not None) and (axs is not None):
        fig.show()
        fig.savefig(fig_path)
    print("")

###############################
# ANDROID
###############################

def plot_results_android(results, ax, i=0):
    # ax[0, i].plot(result['f1'], color='blue', marker='o', label='F1')
    # ax[0, i].plot(result['prec'], color='green', marker='*', label='Precision')
    # ax[0, i].plot(result['rec'], color='red', marker='s', label='Recall')
    ax[0, i].plot(results[i]['tpr'], color='red', marker='o', label='TPR')
    ax[0, i].plot(results[i]['old_tpr'], color='red', marker='*', linestyle='dashed', label='old-TPR')
    # ax[0, i].plot(result['fpr'], color='red', marker='o', label='FPR')
    # ax[0, i].plot(result['old_fpr'], color='red', marker='*', linestyle='dashed', label='old-FPR')
    tnr = 1 - np.array(results[i]['fpr'])
    old_tnr = [(1 - x) if x is not None else None for x in results[i]['old_fpr']]
    ax[0, i].plot(tnr, color='green', marker='o', label='TNR')
    ax[0, i].plot(old_tnr, color='green', marker='*', linestyle='dashed', label='old-TNR')

    nfr_pos = np.array([math.nan] + results[i]['nfr_pos'][1:])
    nfr_neg = np.array([math.nan] + results[i]['nfr_neg'][1:])
    nfr_mean = (nfr_neg + nfr_pos)/2
    nfr_tot = np.array([math.nan] + results[i]['nfr_tot'][1:])

    # ax[1, i].plot(nfr_pos, color='red', marker='v', label='NFR-mw')
    # ax[1, i].plot(nfr_neg, color='green', marker='^', label='NFR-gw')
    ax[1, i].plot(nfr_mean, color='green', marker='^', label='NFR-mean')
    ax[1, i].plot(nfr_tot, color='blue', linestyle='dashed', marker='+', label='NFR-tot')

    # Plot differences
    if i > 0:
        old_nfr_pos = np.array([math.nan] + results[0]['nfr_pos'][1:])
        old_nfr_neg = np.array([math.nan] + results[0]['nfr_neg'][1:])
        old_nfr_tot = np.array([math.nan] + results[0]['nfr_tot'][1:])
        nfr_pos = nfr_pos - old_nfr_pos
        nfr_neg = nfr_neg - old_nfr_neg
        nfr_mean = nfr_mean - (old_nfr_pos + old_nfr_neg)/2
        nfr_tot = nfr_tot - old_nfr_tot
        # ax[2, i].plot(nfr_pos, color='red', marker='v', label='NFR-mw')
        # ax[2, i].plot(nfr_neg, color='green', marker='^', label='NFR-gw')
        ax[2, i].plot(nfr_mean, color='green', marker='^', label='NFR-mean')
        ax[2, i].plot(nfr_tot, color='blue', linestyle='dashed', marker='+', label='NFR-tot')

    ax[2, i].axhline(y=0, color='k', linestyle='dashed')

    # ax[row, 2].plot(result['pfrs_pos'], color='red', marker='>', label='PFR-mw')
    # ax[row, 2].plot(result['pfrs_neg'], color='green', marker='<', label='PFR-gw')
    # ax[row, 0].set_ylabel(f"C = {result['C']}")

    # titles = ['Performances (%)',
    #           'Negative Flip Rate (%)',
    #           'Positive Flip Rate (%)']
    titles = ['Performances (%)',
              'Negative Flip Rate (%)',
              'NFR(i) - NFR(0)']

    sw = results[i]['sample_weight'] if results[i]['sample_weight'] is not None else 'None'
    for j, title in enumerate(titles):
        ax[j, 0].set_ylabel(title)
        ax[j, i].set_xlabel('Updates')
        ax[j, i].set_xticks(np.arange(start=0, stop=len(results[i]['f1']), step=3))
        ax[j, i].legend()

    ax[0, i].set_title(f'sample weight = {sw}')
    ax[0, i].set_ylim(0.6, 1)
    ax[1, i].set_ylim(0, 0.02)


def plot_sequence_results_android(results_path,
                              fig_fname,
                              title=None):

    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    # for res in results:
    #     if res['C'] == 0.01:
    #         result = res
    #         break

    n_rows = 3
    fig, ax = plt.subplots(n_rows, len(results),
                           figsize=(5 * len(results), 5 * n_rows),
                           squeeze=False)
    for i in range(len(results)):
        plot_results_android(results, ax, i)

    # fig, ax = plt.subplots(1, 2, figsize=(10, 5), squeeze=False)
    # plot_android_result(result, ax)

    title = fig_fname if title is None else title
    fig.suptitle(title)
    fig.tight_layout()
    fig.show()
    fig.savefig(f"images/android/{fig_fname}.pdf")


def fill_quadrants(ax, xlim=(), ylim=()):

    color_q_list = ['orange', 'tomato', 'blue', 'green']
    hatch_list = ['//', '\\\\', '||', '--']
    # hatch_list = ['++']*4
    ij_list = [(1, 1), (0, 1), (0, 0), (1, 0)]
    quadrant_labels = ["$Q_1$", "$Q_2$", "$Q_3$", "$Q_4$"]
    alpha = .2
    zorder = -100

    if len(xlim) == 0:
        xlim = ax.get_xlim()
    if len(ylim) == 0:
        ylim = ax.get_ylim()

    for q, (color_q, hatch, (i, j), q_label) in enumerate(zip(color_q_list,
                                      hatch_list,
                                      ij_list,
                                      quadrant_labels)):
        if q in (1, 3):
            ax.add_patch(Rectangle((0, 0), xlim[i], ylim[j],
                                   color=color_q,
                                   fill=True,
                                   # hatch=hatch,
                                   alpha=alpha,
                                   # label=q_label,
                                   zorder=zorder)
                         )


def create_legend(ax, figsize=(10, 0.5)):
    # create legend
    h, l = ax.get_legend_handles_labels()
    legend_dict = dict(zip(l, h))
    legend_fig = plt.figure(figsize=figsize)

    legend_fig.legend(legend_dict.values(), legend_dict.keys(), loc='center',
                      ncol=len(legend_dict.values()), frameon=False)
    legend_fig.tight_layout()

    return legend_fig


def set_grid(ax, major_delta, minor_delta=None,
             alpha_maj=0.7, alpha_min=0.4,
             linewidth_maj=.5, linewidth_min=.2,
             linestyle_maj='dashdot', linestyle_min='dotted'
             ):
    # Save axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xmin = (xlim[0] - xlim[0] % major_delta) - major_delta
    xmax = (xlim[1] - xlim[1] % major_delta) + major_delta
    ymin = (ylim[0] - ylim[0] % major_delta) - major_delta
    ymax = (ylim[1] - ylim[1] % major_delta) + major_delta

    major_ticks = np.arange(min(xmin, ymin), max(xmax, ymax), major_delta)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.grid(which='major', alpha=alpha_maj,
            linewidth=linewidth_maj,
            linestyle=linestyle_maj,
            zorder=-100)

    if minor_delta is not None:
        minor_ticks = np.arange(min(xmin, ymin), max(xmax, ymax), minor_delta)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(minor_ticks, minor=True)
        ax.grid(which='minor', alpha=alpha_min,
                linewidth=linewidth_min,
                linestyle=linestyle_min,
                zorder=-100)

        # Restore axis limits (with different ticks it changes visualization)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def plot_axis_lines(ax, linestyle='solid', color='grey',
                    alpha=0.99, zorder=-100):
    ax.hlines(y=0, xmin=-100, xmax=100,
              linestyle=linestyle, color=color, alpha=alpha, zorder=zorder)
    ax.vlines(x=0, ymin=-100, ymax=100,
              linestyle=linestyle, color=color, alpha=alpha, zorder=zorder)

def remove_ticklabels(ax):
    ax.set_xticklabels(['']*len(ax.get_xticks()))
    ax.set_yticklabels([''] * len(ax.get_yticks()))