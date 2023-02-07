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
from utils.eval import get_ds_outputs, evaluate_acc, compute_nflips, compute_pflips, correct_predictions
from utils.custom_loss import MyCrossEntropyLoss, PCTLoss, MixedPCTLoss
from utils.models_simple import MyLinear, MLP

from torch.utils.data import DataLoader, TensorDataset

# todo: can go to utils.data
def blobs_to_tensor_ds(n_features, centers, cluster_std,
                       n_samples, random_state, batch_size):
    ds = CDLRandomBlobs(n_features=n_features,
                             centers=centers,
                             cluster_std=cluster_std,
                             n_samples=n_samples,
                             random_state=random_state).load()
    X = torch.Tensor(ds.X.tolist())
    Y = torch.Tensor(ds.Y.tolist())
    ds = TensorDataset(X, Y)
    ds_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    return X, Y, ds, ds_loader

def demo_train(model_class, input_size, output_size, train_loader,
               lr=1e-3, n_epochs=1, device='cpu', old_model=None, loss_fn=None, adv_tr=False,
               seed=0):

    set_all_seed(seed)
    model = model_class(input_size=input_size, output_size=output_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    set_all_seed(seed)
    if adv_tr:
        if loss_fn is None:
            loss_fn = MyCrossEntropyLoss()
            for epoch in range(n_epochs):
                adv_train_epoch(model=model, device=device, train_loader=train_loader,
                               optimizer=optimizer, epoch=epoch, loss_fn=loss_fn)
        else:
            for epoch in range(n_epochs):

                adv_pc_train_epoch(model=model, old_model=old_model, device=device,
                                   train_loader=train_loader, optimizer=optimizer,
                                   epoch=epoch, loss_fn=loss_fn)
    else:
        if loss_fn is None:
            loss_fn = MyCrossEntropyLoss()
            for epoch in range(n_epochs):
                train_epoch(model=model, device=device, train_loader=train_loader,
                            optimizer=optimizer, epoch=epoch, loss_fn=loss_fn)
        else:
            for epoch in range(n_epochs):
                pc_train_epoch(model=model, device=device, train_loader=train_loader,
                            optimizer=optimizer, epoch=epoch, loss_fn=loss_fn)




    return model, loss_fn

# todo: can go to utils.evaluate BUT TO BE RENAMED
def compute_metrics(model, ds_loader, device, old_correct=None):
    metrics = {}

    metrics['new_correct'] = correct_predictions(model, ds_loader, device)
    metrics['new_acc'] = metrics['new_correct'].cpu().numpy().mean()
    if old_correct is not None:
        metrics['old_acc'] = old_correct.cpu().numpy().mean()
        metrics['nf_idxs'] = compute_nflips(old_correct, metrics['new_correct'], indexes=True)
        metrics['pf_idxs'] = compute_pflips(old_correct, metrics['new_correct'], indexes=True)
        metrics['diff_acc'] = metrics['new_acc'] - metrics['old_acc']
        metrics['pfr'] = metrics['pf_idxs'].mean()
        metrics['nfr'] = metrics['nf_idxs'].mean()
        metrics['idxs'] = metrics['nf_idxs']

    return metrics

def train_plot(model_class, centers, cluster_std=1., theta=0., n_samples_per_class=100,
               n_epochs=5, batch_size=1, lr=1e-3,
               alpha=1, beta=5, eval_trainset=True,
               diff_model_init=False, diff_trset_init=False,
               show_losses=False, adv_tr=False,
               fname=None, random_state=999):

    if not isinstance(alpha, list):
        alpha = [alpha]
    if not isinstance(beta, list):
        beta = [beta]

    assert len(alpha) == len(beta)

    lsel = 2 if show_losses else 1
    n_plot_x = lsel    #len(diff_trset_init) * 2     # include loss plots
    n_plot_y = 1 + len(alpha)
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

    # Distinguish between old and new train set
    train_ds = {}
    X_tr, Y_tr = {}, {}
    tr_loader = {}
    for ds_i, ds in enumerate(['old', 'new']):
        random_state_trsets = random_state + 1 + ds_i \
            if diff_trset_init else random_state + 1
        theta_i = theta if ds_i == 0 else -theta
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
        X, Y, ds, ds_loader = blobs_to_tensor_ds(n_features=n_features,
                                                 centers=rotate(centers, theta_i),
                                                 cluster_std=cluster_std,
                                                 n_samples=n_samples,
                                                 random_state=random_state_trsets,
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
                                        seed=random_state_model)

    old_metrics = compute_metrics(old_model, ds_loader, device)
    old_correct = old_metrics['new_correct']    # the function considers "new" the model you pass as argument
    old_acc = old_metrics['new_acc']
    #
    # my_plot_decision_regions(model=old_model, samples=X, targets=Y,
    #                          device=device, flipped_samples=None,
    #                          ax=ax[0, 0],
    #                          n_grid_points=n_points_per_dim)
    # ax[0, 0].set_xlabel(f"Acc: {old_acc * 100:.2f}%")


    # my_plot_decision_regions(new_model, X, Y, device, new_metrics['idxs'], ax[lsel, 1],
    #                          n_grid_points=n_points_per_dim)
    # ax[lsel, 1].set_xlabel(f"Acc: {new_acc * 100:.2f}%"
    #                         f"({'+' if diff_acc >= 0 else ''}{diff_acc * 100:.2f}%)\n"
    #                         f"NF: {nf_idxs.sum()} ({nfr * 100:.2f}%), "
    #                         f"PF: {pf_idxs.sum()} ({pfr * 100:.2f}%)")

    if show_losses:
        plot_loss(old_loss_fn.loss_path, ax=ax[lsel + 1, 0])


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
                                            loss_fn=loss_fn,
                                            seed=random_state_model)

        new_metrics = compute_metrics(new_model, ds_loader, device, old_correct)
        new_acc, diff_acc, nf_idxs, nfr, pf_idxs, pfr = new_metrics['new_acc'], \
                                                        new_metrics['diff_acc'], \
                                                        new_metrics['nf_idxs'], \
                                                        new_metrics['nfr'], \
                                                        new_metrics['pf_idxs'], \
                                                        new_metrics['pfr']

        if j == 0:
            # Plot testing set  and NFs withing decision regions
            my_plot_decision_regions(model=old_model, samples=X, targets=Y,
                                     device=device, flipped_samples=None,
                                     ax=ax[0, 0],
                                     n_grid_points=n_points_per_dim)
            ax[0, 0].set_xlabel(f"Acc: {old_acc * 100:.2f}%")

        my_plot_decision_regions(model=new_model, samples=X, targets=Y,
                                 device=device, flipped_samples=nf_idxs,
                                 ax=ax[0, j + 1],
                                 n_grid_points=n_points_per_dim)
        ax[0, j + 1].set_xlabel(f"Acc: {new_acc * 100:.2f}%"
                                   f"({'+' if diff_acc>=0 else ''}{diff_acc * 100:.2f}%)\n"
                                   f"NF: {nf_idxs.sum()} ({nfr * 100:.2f}%), "
                                   f"PF: {pf_idxs.sum()} ({pfr * 100:.2f}%)")
        if show_losses:
            plot_loss(new_loss_fn.loss_path, ax=ax[1, j + 1])


    for i in range(n_plot_x):
        for j in range(n_plot_y):
            if j == 0:
                ax[i, j].set_title('Old model')
            elif j == 1:
                ax[i, j].set_title('New model')
            else:
                ax[i, j].set_title(f"PCT finetuned model\n"
                                   f"(alpha={alpha[j - 1]}, beta={beta[j - 1]})")

    fig.tight_layout()
    fig.show()
    if fname is not None:
        fig.savefig(f'images/{fname}.pdf')

    # fig_tr.show()

    print("")

def main():
    random_state = 0
    centers = np.array([[1, -1], [-1, -1], [-1, 1], [1, 1]]) # centers of the clusters
    #centers = np.array([[1, -1], [-1, -1]])
    cluster_std = 0.6  # standard deviation of the clusters

    # alpha = [0,10,1,1,1,1,1]
    # beta = [10,0,1,2,5,10,100]
    alpha = [None, 1, 1, 1, 1]
    beta = [None, 1, 2, 5, 10]

    lr = 1e-3
    ft_lr = 1e-3
    n_epochs = 10
    n_ft_epochs = 10
    batch_size = 10
    n_samples_per_class = 50


    eval_trainset = True
    diff_model_init = True
    diff_trset_init = True
    show_losses = False
    adv_tr = True
    model_class = MyLinear
    theta = 10

    model_name = 'linear' if model_class is MyLinear else 'mlp'

    fname = None
    # f"complete_plot_samples-{n}_MLP_{random_state}" #'churn_plot_rotation_drift'
    #f"churn_plot_nsamples_tr-{eval_trainset}-{n}_m-{model_name}_alpha-{alpha}_beta-{beta}"

    train_plot(model_class=model_class, centers=centers,
         cluster_std=cluster_std, theta=theta, n_samples_per_class=n_samples_per_class,
         n_epochs=n_epochs, batch_size=batch_size,
         lr=lr,
         alpha=alpha, beta=beta,
         eval_trainset=eval_trainset,
         diff_model_init=diff_model_init, diff_trset_init=diff_trset_init,
         show_losses=show_losses, adv_tr=adv_tr,
         fname=fname, random_state=random_state)

    print("")



if __name__ == '__main__':
    main()

