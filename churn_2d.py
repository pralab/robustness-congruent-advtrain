from secml.data.loader import CDLRandomBlobs
from secml.ml.features import CNormalizerMinMax
from secml.ml.classifiers import CClassifierPyTorch
from secml.figure import CFigure
from torch import nn
import torch
from utils.trainer import train_epoch, pc_train_epoch
import matplotlib.pyplot as plt
import numpy as np
from utils.visualization import my_plot_decision_regions, plot_loss
from utils.utils import set_all_seed, rotate
from utils.eval import get_ds_outputs, evaluate_acc, compute_nflips, compute_pflips, correct_predictions
from utils.custom_loss import MyCrossEntropyLoss, PCTLoss, MixedPCTLoss
from utils.models_simple import MyLinear, MLP

from torch.utils.data import DataLoader, TensorDataset


def main(model_class, centers, cluster_std=1., theta=0., n_samples_per_class=100,
         n_epochs=5, n_ft_epochs=5, batch_size=1, lr=1e-3, ft_lr=1e-3,
         mixed_loss=False, only_nf=False, alpha=1, beta=5, eval_trainset=True,
         diff_model_init=False, diff_trset_init=False,
         show_losses=False,
         fname=None, random_state=999):

    if not isinstance(alpha, list):
        alpha = [alpha]
    if not isinstance(beta, list):
        beta = [beta]

    assert len(alpha) == len(beta)

    lsel = 2 if show_losses else 1

    n_plot_x = 3 * lsel    #len(diff_trset_init) * 2     # include loss plots
    n_plot_y = 2 + len(alpha)
    fig, ax = plt.subplots(n_plot_x, n_plot_y,
                           figsize=(n_plot_y*5, n_plot_x*5),
                           squeeze=False)

    set_all_seed(random_state)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_features = 2  # number of features

    n_points_per_dim = 1e5

    n_samples = n_samples_per_class * len(centers)  # number of samples

    ###################################
    # DATA PREPARATION
    ###################################

    train_ds = {}
    X_tr, Y_tr = {}, {}
    tr_loader = {}
    for ds_i, ds in enumerate(['old', 'new']):
        random_state_trsets = random_state + 1 + ds_i \
            if diff_trset_init else random_state + 1
        theta_i = theta if ds_i == 0 else -theta
        train_ds[ds] = CDLRandomBlobs(n_features=n_features,
                                centers=rotate(centers, theta_i),
                                cluster_std=cluster_std,
                                n_samples=n_samples,
                                random_state=random_state_trsets).load()

        X_tr[ds] = torch.Tensor(train_ds[ds].X.tolist())
        Y_tr[ds] = torch.Tensor(train_ds[ds].Y.tolist())
        train_ds[ds] = TensorDataset(X_tr[ds], Y_tr[ds])
        tr_loader[ds] = DataLoader(train_ds[ds], batch_size=batch_size, shuffle=False)

    if eval_trainset:
        ds_loader = tr_loader['new']
        X, Y = X_tr['new'], Y_tr['new']
    else:
        test_ds = CDLRandomBlobs(n_features=n_features,
                                  centers=centers,
                                  cluster_std=cluster_std,
                                  n_samples=n_samples,
                                  random_state=random_state).load()
        X_ts = torch.Tensor(test_ds.X.tolist())
        Y_ts = torch.Tensor(test_ds.Y.tolist())
        test_ds = TensorDataset(X_ts, Y_ts)
        ts_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        ds_loader = ts_loader
        X, Y = X_ts, Y_ts


    ###################################
    # TRAINING
    ###################################

    # Training baseline model from skratch
    random_state_model = random_state
    set_all_seed(random_state_model)
    old_model = model_class(input_size=n_features, output_size=len(centers))
    old_optimizer = torch.optim.SGD(old_model.parameters(), lr=lr, momentum=0.9)
    old_loss_fn = MyCrossEntropyLoss()
    set_all_seed(random_state_model)
    for epoch in range(n_epochs):
        train_epoch(model=old_model, device=device, train_loader=tr_loader['old'],
                    optimizer=old_optimizer, epoch=epoch, loss_fn=old_loss_fn)

    old_correct = correct_predictions(old_model, ds_loader, device)
    old_acc = old_correct.cpu().numpy().mean()

    # Standard Finetuning
    random_state_model = random_state + 1 if diff_model_init \
        else random_state
    set_all_seed(random_state_model)
    new_model = model_class(input_size=n_features, output_size=len(centers))
    new_loss_fn = MyCrossEntropyLoss()
    #new_model.load_state_dict(old_model.state_dict())
    new_optimizer = torch.optim.SGD(new_model.parameters(), lr=ft_lr, momentum=0.9)
    for epoch in range(n_epochs):
        train_epoch(model=new_model, device=device, train_loader=tr_loader['new'],
                    optimizer=new_optimizer, epoch=epoch, loss_fn=new_loss_fn)
    new_correct = correct_predictions(new_model, ds_loader, device)
    nf_idxs = compute_nflips(old_correct, new_correct, indexes=True)
    pf_idxs = compute_pflips(old_correct, new_correct, indexes=True)
    new_acc = new_correct.cpu().numpy().mean()
    diff_acc = new_acc - old_acc
    pfr = pf_idxs.mean()
    nfr = nf_idxs.mean()
    idxs = nf_idxs

    for i, (mixed_loss, only_nf) in enumerate([[False, None],
                                              [True, False],
                                               [True, True]]):
        if mixed_loss:
            if only_nf:
                ylabel = 'Mix MSE Loss (NF)'
            else:
                ylabel = 'Mix MSE Loss'
        else:
            ylabel = 'PCT Loss'

        ax[lsel * i, 0].set_ylabel(ylabel)
        # Plot testing set  and NFs withing decision regions
        my_plot_decision_regions(old_model, X, Y, device, idxs, ax[lsel * i, 0],
                                 n_grid_points=n_points_per_dim)
        ax[lsel * i, 0].set_xlabel(f"Acc: {old_acc*100:.2f}%")



        my_plot_decision_regions(new_model, X, Y, device, idxs, ax[lsel * i, 1],
                                 n_grid_points=n_points_per_dim)
        ax[lsel * i, 1].set_xlabel(f"Acc: {new_acc * 100:.2f}%"
                                f"({'+' if diff_acc >= 0 else ''}{diff_acc * 100:.2f}%)\n"
                                f"NF: {nf_idxs.sum()} ({nfr * 100:.2f}%), "
                                f"PF: {pf_idxs.sum()} ({pfr * 100:.2f}%)")
        if show_losses:
            plot_loss(old_loss_fn.loss_path, ax=ax[lsel * i + 1, 0])
            plot_loss(new_loss_fn.loss_path, ax=ax[lsel * i + 1, 1])


        random_state_model = random_state + 2 if diff_model_init \
            else random_state
        for j, (alpha_j, beta_j) in enumerate(list(zip(alpha, beta))):
            # PCT Finetuning
            set_all_seed(random_state_model)
            pct_model = model_class(input_size=n_features, output_size=len(centers))
            pct_model.load_state_dict(new_model.state_dict())
            pct_optimizer = torch.optim.SGD(pct_model.parameters(), lr=ft_lr, momentum=0.9)

            if not mixed_loss:
                old_outputs = get_ds_outputs(old_model, tr_loader['new'], device)
                pct_loss_fn = PCTLoss(old_outputs, alpha1=alpha_j, beta1=beta_j)
            else:
                old_outputs = get_ds_outputs(old_model, tr_loader['new'], device)
                new_outputs = get_ds_outputs(new_model, tr_loader['new'], device)
                pct_loss_fn = MixedPCTLoss(old_outputs, new_outputs,
                                           alpha1=alpha_j, beta1=beta_j,
                                           only_nf=only_nf)

            for epoch in range(n_ft_epochs):
                pc_train_epoch(pct_model, device, tr_loader['new'],
                               pct_optimizer, epoch, pct_loss_fn)

            pct_correct = correct_predictions(pct_model, ds_loader, device)
            pct_nf_idxs = compute_nflips(old_correct, pct_correct, indexes=True)
            pct_pf_idxs = compute_pflips(old_correct, pct_correct, indexes=True)
            pct_acc = pct_correct.cpu().numpy().mean()
            pct_diff_acc = pct_acc - old_acc
            pct_pfr = pct_pf_idxs.mean()
            pct_nfr = pct_nf_idxs.mean()
            # pct_idxs = nf_idxs if pct_nfr < nfr else pct_nf_idxs
            pct_idxs = pct_nf_idxs
            my_plot_decision_regions(model=pct_model, samples=X, targets=Y,
                                     device=device, flipped_samples=pct_idxs,
                                     ax=ax[lsel * i, j + 2],
                                     n_grid_points=n_points_per_dim)
            ax[lsel * i, j + 2].set_xlabel(f"Acc: {pct_acc * 100:.2f}%"
                                   f"({'+' if pct_diff_acc>=0 else ''}{pct_diff_acc * 100:.2f}%)\n"
                                   f"NF: {pct_nf_idxs.sum()} ({pct_nfr * 100:.2f}%), "
                                   f"PF: {pct_pf_idxs.sum()} ({pct_pfr * 100:.2f}%)")
            if show_losses:
                plot_loss(pct_loss_fn.loss_path, ax=ax[lsel * i + 1, j + 2])


    for i in range(n_plot_x):
        for j in range(n_plot_y):
            if j == 0:
                ax[i, j].set_title('Old model')
            elif j == 1:
                ax[i, j].set_title('New model')
            else:
                if i == 1:
                    ax[i, j].set_title(f"MixMSE finetuned model\n"
                                       f"beta={beta[j-2]}")
                elif i == 2:
                    ax[i, j].set_title(f"MixMSE (NF) finetuned model\n"
                                       f"beta={beta[j-2]}")
                else:
                    ax[i, j].set_title(f"PCT finetuned model\n"
                                       f"(alpha={alpha[j - 2]}, beta={beta[j - 2]})")

    fig.tight_layout()
    # fig.suptitle(title)
    if fname is not None:
        fig.savefig(f'images/{fname}.pdf')
    fig.show()
    # fig_tr.show()

    print("")

if __name__ == '__main__':
    random_state = 999
    for random_state in np.arange(995, 999):
        centers = np.array([[1, -1], [-1, -1], [-1, 1], [1, 1]]) # centers of the clusters
        #centers = np.array([[1, -1], [-1, -1]])
        cluster_std = 0.6  # standard deviation of the clusters

        alpha = [0.1, 0.5, 1]
        beta = [1, 2, 5, 10]
        alpha = beta


        lr = 1e-3
        ft_lr = 1e-3
        n_epochs = 10
        n_ft_epochs = 10
        batch_size = 10
        n = 100

        mixed_loss = True
        only_nf = True

        eval_trainset = False
        diff_model_init = True
        diff_trset_init = False
        show_losses = True
        model_class = MLP
        theta = 0

        model_name = 'linear' if model_class is MyLinear else 'mlp'

        fname = f"complete_plot_samples-{n}_MLP_{random_state}" #'churn_plot_rotation_drift'
        #f"churn_plot_nsamples_tr-{eval_trainset}-{n}_m-{model_name}_alpha-{alpha}_beta-{beta}"

        main(model_class=model_class, centers=centers,
             cluster_std=cluster_std, theta=theta, n_samples_per_class=n,
             n_epochs=n_epochs, n_ft_epochs=n_ft_epochs, batch_size=batch_size,
             lr=lr, ft_lr=ft_lr, mixed_loss=mixed_loss, only_nf=only_nf,
             alpha=alpha, beta=beta,
             eval_trainset=eval_trainset,
             diff_model_init=diff_model_init, diff_trset_init=diff_trset_init,
             show_losses=show_losses,
             fname=fname, random_state=random_state)

    print("")
