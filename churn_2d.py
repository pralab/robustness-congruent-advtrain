from secml.data.loader import CDLRandomBlobs
from secml.ml.features import CNormalizerMinMax
from secml.ml.classifiers import CClassifierPyTorch
from secml.figure import CFigure
from torch import nn
import torch.nn.functional as F
import torch
from utils.trainer import train_epoch, pc_train_epoch
import matplotlib.pyplot as plt
import numpy as np
from utils.visualization import my_plot_decision_regions
from utils.utils import set_all_seed
from utils.eval import get_ds_outputs, evaluate_acc, compute_nflips, compute_pflips, correct_predictions
from utils.custom_loss import PCTLoss

from torch.utils.data import DataLoader, TensorDataset



def main(model_class, n_samples_per_class=100, n_epochs=5, n_ft_epochs=5,
         lr=1e-3, ft_lr=1e-3, alpha=1,
         beta=5, eval_trainset=True, fname='decision_regions'):

    if not isinstance(n_samples_per_class, list):
        n_samples_per_class = [n_samples_per_class]
    if not isinstance(alpha, list):
        alpha = [alpha]
    if not isinstance(beta, list):
        beta = [beta]

    assert len(alpha) == len(beta)

    n_plot_x = len(n_samples_per_class)
    n_plot_y = 2 + len(alpha)

    fig, ax = plt.subplots(n_plot_x, n_plot_y,
                           figsize=(n_plot_y*5, n_plot_x*5),
                           squeeze=False)

    for i, n_samples_per_class_i in enumerate(n_samples_per_class):

        ax[i, 0].set_ylabel(f"{n_samples_per_class_i} samples")

        random_state = 999
        set_all_seed(random_state)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        n_features = 2  # number of features
        centers = np.array([[1, -1], [-1, -1], [-1, 1], [1, 1]]) # centers of the clusters
        # centers = np.array([[0, 0], [0, 2], [-1, -1], [1, -1]])  # centers of the clusters
        centers = centers*2
        n_points_per_dim = 1e5

        n_samples = n_samples_per_class_i * len(centers)  # number of samples
        cluster_std = 1  # standard deviation of the clusters

        # Prepare data
        train_ds = CDLRandomBlobs(n_features=n_features,
                                centers=centers,
                                cluster_std=cluster_std,
                                n_samples=n_samples,
                                random_state=random_state).load()

        X_tr = torch.Tensor(train_ds.X.tolist())
        Y_tr = torch.Tensor(train_ds.Y.tolist())
        train_ds = TensorDataset(X_tr, Y_tr)
        tr_loader = DataLoader(train_ds, batch_size=10, shuffle=False)

        old_model = model_class(input_size=n_features, output_size=len(centers))
        optimizer = torch.optim.SGD(old_model.parameters(), lr=lr, momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()

        if eval_trainset:
            ds_loader = tr_loader
            X, Y = X_tr, Y_tr
        else:
            test_ds = CDLRandomBlobs(n_features=n_features,
                                      centers=centers,
                                      cluster_std=cluster_std,
                                      n_samples=n_samples,
                                      random_state=random_state+1).load()
            X_ts = torch.Tensor(test_ds.X.tolist())
            Y_ts = torch.Tensor(test_ds.Y.tolist())
            test_ds = TensorDataset(X_ts, Y_ts)
            ts_loader = DataLoader(test_ds, batch_size=10, shuffle=False)
            ds_loader = ts_loader
            X, Y = X_ts, Y_ts


        ###################################
        # TRAINING
        ###################################

        # Training baseline model from skratch
        for epoch in range(n_epochs):
            train_epoch(model=old_model, device=device, train_loader=tr_loader,
                        optimizer=optimizer, epoch=epoch, loss_fn=loss_fn)

        old_correct = correct_predictions(old_model, ds_loader, device)
        old_acc = old_correct.numpy().mean()

        # Standard Finetuning
        set_all_seed(random_state)
        new_model = model_class(input_size=n_features, output_size=len(centers))
        new_model.load_state_dict(old_model.state_dict())
        optimizer = torch.optim.SGD(new_model.parameters(), lr=ft_lr, momentum=0.9)
        for epoch in range(n_ft_epochs):
            train_epoch(model=new_model, device=device, train_loader=tr_loader,
                        optimizer=optimizer, epoch=epoch, loss_fn=loss_fn)
        new_correct = correct_predictions(new_model, ds_loader, device)
        nf_idxs = compute_nflips(old_correct, new_correct, indexes=True)
        pf_idxs = compute_pflips(old_correct, new_correct, indexes=True)
        new_acc = new_correct.numpy().mean()
        diff_acc = new_acc - old_acc
        pfr = pf_idxs.mean()
        nfr = nf_idxs.mean()
        idxs = nf_idxs

        my_plot_decision_regions(old_model, X, Y, device, idxs, ax[i, 0],
                                 n_grid_points=n_points_per_dim)
        ax[i, 0].set_xlabel(f"Acc: {old_acc*100:.2f}%")

        my_plot_decision_regions(new_model, X, Y, device, idxs, ax[i, 1],
                                 n_grid_points=n_points_per_dim)
        ax[i, 1].set_xlabel(f"Acc: {new_acc * 100:.2f}%"
                        f"({'+' if diff_acc >= 0 else ''}{diff_acc * 100:.2f}%)\n"
                        f"NF: {nf_idxs.sum()} ({nfr * 100:.2f}%), "
                        f"PF: {pf_idxs.sum()} ({pfr * 100:.2f}%)")


        for j, (alpha_j, beta_j) in enumerate(list(zip(alpha, beta))):
            # PCT Finetuning
            set_all_seed(random_state)
            pct_model = model_class(input_size=n_features, output_size=len(centers))
            pct_model.load_state_dict(new_model.state_dict())
            optimizer = torch.optim.SGD(pct_model.parameters(), lr=ft_lr, momentum=0.9)
            old_outputs = get_ds_outputs(old_model, tr_loader, device)
            loss_fn = PCTLoss(old_outputs, alpha1=alpha_j, beta1=beta_j)
            for epoch in range(n_ft_epochs):
                pc_train_epoch(pct_model, device, tr_loader, optimizer, epoch, loss_fn)

            pct_correct = correct_predictions(pct_model, ds_loader, device)
            pct_nf_idxs = compute_nflips(old_correct, pct_correct, indexes=True)
            pct_pf_idxs = compute_pflips(old_correct, pct_correct, indexes=True)
            pct_acc = pct_correct.numpy().mean()
            pct_diff_acc = pct_acc - old_acc
            pct_pfr = pct_pf_idxs.mean()
            pct_nfr = pct_nf_idxs.mean()
            # pct_idxs = nf_idxs if pct_nfr < nfr else pct_nf_idxs
            pct_idxs = pct_nf_idxs
            my_plot_decision_regions(pct_model, X, Y,
                                     device, pct_idxs, ax[i, j + 2],
                                     n_grid_points=n_points_per_dim)
            ax[i, j + 2].set_xlabel(f"Acc: {pct_acc * 100:.2f}%"
                                   f"({'+' if pct_diff_acc>=0 else ''}{pct_diff_acc * 100:.2f}%)\n"
                                   f"NF: {pct_nf_idxs.sum()} ({pct_nfr * 100:.2f}%), "
                                   f"PF: {pct_pf_idxs.sum()} ({pct_pfr * 100:.2f}%)")

    for j in range(n_plot_y):
        if j == 0:
            ax[0, j].set_title('Old model')
        elif j == 1:
            ax[0, j].set_title('Standard finetuned model')
        else:
            ax[0, j].set_title(f"PCT finetuned model\n"
                               f"(alpha={alpha[j-2]}, beta={beta[j-2]})")


    # fig.suptitle(title)
    fig.savefig(f'images/{fname}.pdf')
    fig.show()

    print("")

class MLP(nn.Module):
    """Model with input size (-1, 28, 28) for MNIST 10-classes dataset."""

    def __init__(self, input_size=2, output_size=3, hidden_units=10):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_units)
        self.fc = nn.Linear(hidden_units, output_size)

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.fc(x)
        return x

class MyLinear(nn.Module):
    """Model with input size (-1, 28, 28) for MNIST 10-classes dataset."""


    def __init__(self, input_size=2, output_size=3):
        super(MyLinear, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        return x

if __name__ == '__main__':
    alpha = [0.1, 0.5, 1]
    beta = [1, 2, 5]
    lr = 1e-3
    ft_lr = 1e-3
    n_epochs = 5
    n_ft_epochs = 5
    n = [100, 500, 1000]
    eval_trainset = False

    for model_class in [MyLinear, MLP]:
        model_name = 'linear' if model_class is MyLinear else 'mlp'
        main(model_class=model_class, n_samples_per_class=n,
             n_epochs=n_epochs, n_ft_epochs=n_ft_epochs,
             lr=lr, ft_lr=ft_lr, alpha=alpha, beta=beta,
             eval_trainset=eval_trainset,
             fname=f"churn_plot_nsamples_tr-{eval_trainset}-{n}_m-{model_name}_alpha-{alpha}_beta-{beta}")
        print("")
