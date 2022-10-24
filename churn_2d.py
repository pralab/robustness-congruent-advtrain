from secml.data.loader import CDLRandomBlobs
from secml.ml.features import CNormalizerMinMax
from secml.ml.classifiers import CClassifierPyTorch
from secml.figure import CFigure
from torch import nn
import torch
from utils.trainer import train_epoch
import matplotlib.pyplot as plt
import numpy as np
from utils.visualization import my_plot_decision_regions
from utils.utils import set_all_seed

from torch.utils.data import DataLoader, TensorDataset



def main():
    random_state = 999
    set_all_seed(random_state)

    n_features = 2  # number of features
    centers = np.array([[1, -1], [-1, -1], [-1, 1], [1, 1]]) # centers of the clusters
    centers = centers*2
    
    n_samples = 100 * len(centers)  # number of samples
    cluster_std = 0.8  # standard deviation of the clusters

    train_ds = CDLRandomBlobs(n_features=n_features,
                            centers=centers,
                            cluster_std=cluster_std,
                            n_samples=n_samples,
                            random_state=random_state).load()

    X = torch.Tensor(train_ds.X.tolist())
    Y = torch.Tensor(train_ds.Y.tolist())
    train_ds = TensorDataset(X, Y)
    tr_loader = DataLoader(train_ds, batch_size=10, shuffle=False)


    old_model = Net(input_size=n_features, output_size=len(centers))
    optimizer = torch.optim.SGD(old_model.parameters(), lr=0.001, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

 
    for epoch in range(1):
        train_epoch(model=old_model, device='cuda:0', train_loader=tr_loader,
                    optimizer=optimizer, epoch=epoch, loss_fn=loss_fn)
    
    new_model = Net(input_size=n_features, output_size=len(centers))
    new_model.load_state_dict(old_model.state_dict())
    for epoch in range(3):
        train_epoch(model=new_model, device='cuda:0', train_loader=tr_loader,
                    optimizer=optimizer, epoch=epoch, loss_fn=loss_fn)



    fname ='decision_regions_best'
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    title = ('Decision Regions')
    my_plot_decision_regions(old_model, X, Y, ax[0], n_grid_points=1000)
    my_plot_decision_regions(new_model, X, Y, ax[1], n_grid_points=1000)
    ax[0].set_title('Old model')
    ax[1].set_title('New model')
    fig.suptitle('Decision regions')
    plt.savefig(f'images/{fname}.png')


class Net(nn.Module):
    """Model with input size (-1, 28, 28) for MNIST 10-classes dataset."""

    def __init__(self, input_size=2, output_size=3):
        super(Net, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        return x



if __name__ == '__main__':
    main()