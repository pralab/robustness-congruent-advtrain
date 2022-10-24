from torchvision.transforms import Normalize
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from secml.utils import fm
import os
import pandas as pd
import torch
import math

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


def show_loss(csv_path, fig_path):
    df = pd.read_csv(csv_path, index_col='epoch')
    df.plot(kind='line')
    plt.savefig(fig_path)

    print("")


def my_plot_decision_regions(model, samples, targets, ax=None, n_grid_points=100, fname='decision_regions'):
    min = torch.min(samples, axis=0)[0] - 1
    max = torch.max(samples, axis=0)[0] + 1
    n_points_per_dim = math.floor(math.sqrt(n_grid_points))
    x = np.linspace(min[0], max[0])
    y = np.linspace(min[1], max[1])
    xx, yy = np.meshgrid(x, y)
    map_points = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])
    model.eval()
    outs = model.cuda()(map_points.cuda())
    preds = outs.argmax(axis=1)
    Z = preds.reshape(xx.shape)

    if ax is None:
        fig, ax = plt.subplots()
        title = ('Decision Regions')
    ax.contourf(xx, yy, Z.cpu(), cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(samples.cpu().numpy()[:, 0], samples.numpy()[:, 1], 
                c=targets.cpu().int().numpy(), cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.legend()    
    if fname is not None:
        plt.savefig(f'images/{fname}.png')
    # plt.show()

