from secml.data.loader import CDLRandomBlobs
from secml.ml.features import CNormalizerMinMax
from secml.ml.classifiers import CClassifierPyTorch
from secml.figure import CFigure
from torch import nn
import torch
from utils.trainer import train_epoch, pc_train_epoch, adv_train_epoch, adv_pc_train_epoch
import matplotlib.pyplot as plt
import numpy as np
from utils.visualization import my_plot_decision_regions, plot_loss
from utils.utils import set_all_seed, rotate
from utils.eval import get_ds_outputs, evaluate_acc, compute_nflips, \
    compute_pflips, correct_predictions, get_pct_results
from utils.custom_loss import MyCrossEntropyLoss, PCTLoss, MixedPCTLoss
from utils.models_simple import MyLinear, MLP
from adv_lib.attacks.auto_pgd import apgd

import os

if os.name != 'nt':
    import matplotlib
    matplotlib.use('macosx')

from torch.utils.data import DataLoader, TensorDataset

# todo: can go to utils.data
def blobs_to_tensor_ds(n_features, centers, cluster_std,
                       n_samples, random_state, batch_size, transform=None):
    ds = CDLRandomBlobs(n_features=n_features,
                        centers=centers,
                        cluster_std=cluster_std,
                        n_samples=n_samples,
                        random_state=random_state).load()
    X = torch.Tensor(ds.X.tolist())
    # X_min, X_max = X.min(), X.max()
    # new_X_min, new_X_max = 0.1, 0.9
    # X = (X - X_min) / (X_max - X_min) * (new_X_max - new_X_min) + new_X_min

    Y = torch.Tensor(ds.Y.tolist())
    Y = Y.type(torch.int64)
    ds = TensorDataset(X, Y)
    ds_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    return X, Y, ds, ds_loader

def make_adv_ds_loader(model, ds_loader, device, eps=0.2, n_steps=20):
    model.to(device)
    model.eval()

    X_adv, Y = [], []
    X = []
    for batch_idx, (x, y) in enumerate(ds_loader):
        x, y = x.to(device), y.to(device)
        advx = apgd(model, x, y,
                    eps=eps, norm=float('inf'), n_iter=n_steps)
        X_adv.append(advx)
        Y.append(y)

        X.append(x)

    X_adv = torch.cat(X_adv)
    Y = torch.cat(Y)

    X = torch.cat(X)

    adv_ds = TensorDataset(X_adv, Y)
    adv_ds_loader = DataLoader(adv_ds, batch_size=ds_loader.batch_size, shuffle=False)

    return adv_ds_loader, X_adv

def demo_train(model_class, input_size, output_size, train_loader,
               lr=1e-3, n_epochs=1, device='cpu', old_model=None,
               loss_fn=None, adv_tr=False,
               eps=1, seed=0):
    n_iter = 5
    set_all_seed(seed)
    model = model_class(input_size=input_size, output_size=output_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    set_all_seed(seed)
    if adv_tr:
        if loss_fn is None:
            loss_fn = MyCrossEntropyLoss()
            for epoch in range(n_epochs):
                adv_train_epoch(model=model, device=device, train_loader=train_loader,
                                optimizer=optimizer, epoch=epoch, loss_fn=loss_fn,
                                eps=eps, n_steps=n_iter)
        else:
            for epoch in range(n_epochs):
                adv_pc_train_epoch(model=model, old_model=old_model, device=device,
                                   train_loader=train_loader, optimizer=optimizer,
                                   epoch=epoch, loss_fn=loss_fn,
                                   eps=eps, n_steps=n_iter)
    else:
        if loss_fn is None:
            loss_fn = MyCrossEntropyLoss()
            for epoch in range(n_epochs):
                train_epoch(model=model, device=device, train_loader=train_loader,
                            optimizer=optimizer, epoch=epoch, loss_fn=loss_fn)
        else:
            assert old_model is not None, 'Adv PC Training need the instance of the old model'
            for epoch in range(n_epochs):
                pc_train_epoch(model=model, device=device, train_loader=train_loader,
                            optimizer=optimizer, epoch=epoch, loss_fn=loss_fn)

    return model, loss_fn


def train_plot(model_class, centers, cluster_std=1., theta=0.,
               n_samples_per_class=100, n_test_sample_per_class=100,
               n_epochs=5, batch_size=1, lr=1e-3, eps=0.5,
               alpha=1, beta=5, eval_trainset=True,
               diff_model_init=False, diff_trset_init=False,
               adv_tr=False, ax=False,
               random_state=999):

    if not isinstance(alpha, list):
        alpha = [alpha]
    if not isinstance(beta, list):
        beta = [beta]

    assert len(alpha) == len(beta)

    set_all_seed(random_state)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_features = 2  # number of features
    n_points_per_dim = 1e5
    n_samples = n_samples_per_class * len(centers)  # number of samples
    n_test_samples = n_test_sample_per_class * len(centers)

    if ax is None:
        n_plot_x = 1
        n_plot_y = 1 + len(alpha)
        fig, ax = plt.subplots(n_plot_x, n_plot_y,
                               figsize=(n_plot_y*5, n_plot_x*5),
                               squeeze=True)

    ###################################
    # DATA PREPARATION
    ###################################

    # Distinguish between old and new train set
    train_ds = {}
    X_tr, Y_tr = {}, {}
    tr_loader = {}
    for ds_i, ds in enumerate(['old', 'new']):
        random_state_trsets = random_state + 1 + ds_i \
            if diff_trset_init else random_state + 1
        theta_i = theta if ds_i == 0 else -theta
        set_all_seed(random_state)
        X_tr[ds], Y_tr[ds], train_ds[ds], tr_loader[ds] = blobs_to_tensor_ds(n_features=n_features,
                                                                             centers=rotate(centers, theta_i),
                                                                             cluster_std=cluster_std,
                                                                             n_samples=n_samples,
                                                                             random_state=random_state_trsets,
                                                                             batch_size=batch_size)
    if eval_trainset:
        # Set test set equal to the train set to check how the optimization goes
        ds_loader = tr_loader['new']
        X, Y = X_tr['new'], Y_tr['new']
    else:
        set_all_seed(random_state)
        X, Y, ds, ds_loader = blobs_to_tensor_ds(n_features=n_features,
                                                 centers=centers,
                                                 cluster_std=cluster_std,
                                                 n_samples=n_test_samples,
                                                 random_state=random_state,
                                                 batch_size=batch_size)

    ###################################
    # TRAINING
    ###################################

    # Training baseline model from skratch
    random_state_model = random_state
    old_model, old_loss_fn = demo_train(model_class=model_class,
                                        input_size=n_features, output_size=len(centers),
                                        lr=lr, n_epochs=n_epochs, device=device,
                                        train_loader=tr_loader['old'], adv_tr=adv_tr,
                                        eps=eps, seed=random_state_model)

    old_correct = correct_predictions(old_model, ds_loader, device)
    adv_ds_loader, old_X_adv = make_adv_ds_loader(old_model, ds_loader, device, eps=eps)
    old_correct_adv = correct_predictions(old_model, adv_ds_loader, device)
    old_rob_acc = old_correct_adv.numpy().mean()

    # print(f"Clean Acc: {old_correct.numpy().mean()}")
    # print(f"Rob Acc: {old_correct_adv.numpy().mean()}")

    random_state_model = random_state + 2 if diff_model_init \
        else random_state
    for j, (alpha_j, beta_j) in enumerate(zip(alpha, beta)):
        if alpha_j is None:
            loss_fn = None
        else:
            old_outputs = get_ds_outputs(old_model, tr_loader['new'], device)
            loss_fn = PCTLoss(old_outputs, alpha1=alpha_j, beta1=beta_j)

        new_model, new_loss_fn = demo_train(model_class=model_class,
                                            input_size=n_features, output_size=len(centers),
                                            lr=lr, n_epochs=n_epochs, device=device,
                                            train_loader=tr_loader['new'],
                                            old_model=old_model,
                                            loss_fn=loss_fn, adv_tr=adv_tr,
                                            eps=eps, seed=random_state_model)

        # new_metrics = compute_metrics(new_model, ds_loader, device, old_correct)
        # new_acc, diff_acc, nf_idxs, nfr, pf_idxs, pfr = new_metrics['new_acc'], \
        #                                                 new_metrics['diff_acc'], \
        #                                                 new_metrics['nf_idxs'], \
        #                                                 new_metrics['nfr'], \
        #                                                 new_metrics['pf_idxs'], \
        #                                                 new_metrics['pfr']

        # Performances on clean data
        new_metrics = get_pct_results(new_model=new_model, ds_loader=ds_loader,
                                      old_correct=old_correct,
                                      device=device)
        new_acc = new_metrics['new_acc']
        old_acc = new_metrics['old_acc']
        diff_acc = new_metrics['diff_acc']
        nf_idxs = new_metrics['nf_idxs']
        nfr = new_metrics['nfr']
        pf_idxs = new_metrics['pf_idxs']
        pfr = new_metrics['pfr']

        # Performances on adversarial data
        adv_ds_loader, X_adv = make_adv_ds_loader(new_model, ds_loader, device, eps=eps)
        adv_new_metrics = get_pct_results(new_model=new_model, ds_loader=adv_ds_loader,
                                          old_correct=old_correct_adv,
                                          device=device)
        adv_new_acc = adv_new_metrics['new_acc']
        adv_new_correct = adv_new_metrics['new_correct']
        adv_old_acc = adv_new_metrics['old_acc']
        adv_diff_acc = adv_new_metrics['diff_acc']
        adv_nf_idxs = adv_new_metrics['nf_idxs']
        adv_nfr = adv_new_metrics['nfr']
        adv_pf_idxs = adv_new_metrics['pf_idxs']
        adv_pfr = adv_new_metrics['pfr']

        if j == 0:
            # Plot testing set  and NFs withing decision regions
            my_plot_decision_regions(model=old_model, samples=X, targets=Y,
                                     device=device, flipped_samples=nf_idxs,
                                     adv_correct=~old_correct_adv, x_adv=old_X_adv,
                                     ax=ax[0], eps=eps,
                                     n_grid_points=n_points_per_dim)
            ax[0].set_xlabel(f"Acc: {old_acc * 100:.2f}%\n"
                                f"Rob: {old_rob_acc * 100:.2f}%")

        my_plot_decision_regions(model=new_model, samples=X, targets=Y,
                                 device=device, flipped_samples=nf_idxs,
                                 adv_flipped_samples=adv_nf_idxs,
                                 adv_correct=~adv_new_correct, x_adv=X_adv,
                                 ax=ax[j + 1], eps=eps,
                                 n_grid_points=n_points_per_dim)

        # ax[j + 1].set_xlabel(f"Acc: {new_acc * 100:.2f}%"
        #                         f"({'+' if diff_acc>=0 else ''}{diff_acc * 100:.2f}%)\n"
        #                         f"NF: {nf_idxs.sum()} ({nfr * 100:.2f}%), "
        #                         f"PF: {pf_idxs.sum()} ({pfr * 100:.2f}%)")

        # ax[j + 1].set_xlabel(f"Acc: {new_acc * 100:.2f}%"
        #                         f"({'+' if diff_acc>=0 else ''}{diff_acc * 100:.2f}%)"\
        #                         f"NF: {nf_idxs.sum()} ({nfr * 100:.2f}%)\n"
        #                         f"Rob: {adv_new_acc * 100:.2f}%"
        #                         f"({'+' if adv_diff_acc>=0 else ''}{adv_diff_acc * 100:.2f}%)"\
        #                         f"adv-NF: {adv_nf_idxs.sum()} ({adv_nfr * 100:.2f}%)")

        ax[j + 1].set_xlabel(f"Acc: {new_acc * 100:.2f}%, " \
                             f"NF: {nf_idxs.sum()}\n"
                             f"Rob: {adv_new_acc * 100:.2f}%, " \
                             f"adv-NF: {adv_nf_idxs.sum()}")


def main():
    random_state = 2
    # centers = np.array([[1, -1], [-1, -1], [-1, 1], [1, 1]])    # centers of the clusters
    centers = np.array([[.3, .3], [.7, .3], [.5, .7]])  # centers of the clusters
    delta = .25
    cluster_std = delta/3  # standard deviation of the clusters

    k = 1
    centers = centers / k
    cluster_std = cluster_std / k

    # alpha = [0,10,1,1,1,1,1]
    # beta = [10,0,1,2,5,10,100]
    alpha = [None, 1, 1, 1, 1]
    beta = [None, 1, 2, 5, 10]

    lr = 1e-2
    ft_lr = 1e-3
    n_epochs = 10
    n_samples_per_class = 50
    batch_size = 10
    n_test_samples_per_class = 5
    eps = 0.05


    eval_trainset = False
    diff_model_init = True
    diff_trset_init = True
    # adv_tr = True
    model_class = MyLinear
    theta = 10

    model_name = 'linear' if model_class is MyLinear else 'mlp'

    fname = 'churn_plot2D'
    # f"complete_plot_samples-{n}_MLP_{random_state}" #'churn_plot_rotation_drift'
    #f"churn_plot_nsamples_tr-{eval_trainset}-{n}_m-{model_name}_alpha-{alpha}_beta-{beta}"


    n_plot_x = 2
    n_plot_y = 1 + len(alpha)
    fig, ax = plt.subplots(n_plot_x, n_plot_y,
                           figsize=(n_plot_y*5, n_plot_x*5),
                           squeeze=False)
    for i, adv_tr in enumerate([False, True]):
        train_plot(model_class=model_class, centers=centers,
                   cluster_std=cluster_std, theta=theta,
                   n_samples_per_class=n_samples_per_class,
                   n_test_sample_per_class=n_test_samples_per_class,
                   n_epochs=n_epochs, batch_size=batch_size,
                   lr=lr, eps=eps,
                   alpha=alpha, beta=beta,
                   eval_trainset=eval_trainset,
                   diff_model_init=diff_model_init, diff_trset_init=diff_trset_init,
                   adv_tr=adv_tr, ax=ax[i],
                   random_state=random_state)

        print("")

    for j in range(n_plot_y):
        if j == 0:
            ax[0, j].set_title('Old model')
        elif j == 1:
            ax[0, j].set_title('New model')
        else:
            ax[0, j].set_title(f"alpha={alpha[j - 1]}, beta={beta[j - 1]}")

    ax[0, 0].set_ylabel("PCT")
    ax[1, 0].set_ylabel("PCT-AT")

    fig.tight_layout()
    fig.show()
    if fname is not None:
        fig.savefig(f'images/demo_2D/{fname}.pdf')



if __name__ == '__main__':
    main()

