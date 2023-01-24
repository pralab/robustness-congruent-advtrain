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
                             flipped_samples=None, ax=None, n_grid_points=100,
                             fname=None):
    min = torch.min(samples, axis=0)[0] - 1
    max = torch.max(samples, axis=0)[0] + 1
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


    if ax is None:
        fig, ax = plt.subplots()
        title = ('Decision Regions')
    ax.contourf(xx, yy, Z.cpu(), cmap=plt.cm.coolwarm, alpha=0.3)
    if flipped_samples is None:
        ax.scatter(samples.cpu().numpy()[:, 0], samples.numpy()[:, 1],
                   c=targets.cpu().int().numpy(), alpha=0.7,
                   cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    else:
        alpha_idx = flipped_samples*1
        alpha_idx[~flipped_samples] = 0.5
        s_idx = flipped_samples*60
        s_idx[~flipped_samples] = 20
        x_list = samples.cpu().numpy()[:, 0]
        y_list = samples.numpy()[:, 1]
        c_list = targets.cpu().int().numpy()

        # EDGE_COLOR_LIST = ['none', 'k']
        # edge_fun = lambda idx: EDGE_COLOR_LIST[idx]
        # edge_list = (flipped_samples*1).to_numpy()
        # edge_list_idx = list(map(edge_fun, edge_list))

        ax.scatter(x_list, y_list,
                   c=c_list, alpha=alpha_idx, cmap=plt.cm.coolwarm,
                   s=s_idx, edgecolors='k')

    # ax.set_xticks(())
    # ax.set_yticks(())
    # ax.legend()
    if fname is not None:
        plt.savefig(f'images/{fname}.png')
    # plt.show()


def plot_loss(loss, ax, window=20):
    loss_df = pd.DataFrame(loss)

    if isinstance(window, int):
        loss_df = loss_df.rolling(window).mean()

    loss_df.plot(ax=ax)
    ax.legend(fontsize=15)
    ax.set_xlabel('iterations')



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
        nfr_mean = nfr_mean - (old_nfr_pos - old_nfr_neg)/2
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
