from torchvision.transforms import Normalize
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from secml.utils import fm
import os
import pandas as pd

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

def imshow(img, transforms=None, figsize=(10, 20)):
    npimg = _tensor_to_show(img, transforms)
    plt.figure(figsize=figsize)
    plt.imshow(npimg, interpolation=None)


def show_batch(x, transforms=None, figsize=(10, 20)):
    imshow(make_grid(x.cpu().detach(), nrow=5),
           transforms=transforms, figsize=figsize)
    plt.axis('off')
    plt.show()


def show_loss(root, fname):
    csv_path = fm.join(root, f"{fname}.csv")
    fig_path = fm.join(root, f"{fname}.pdf")
    df = pd.read_csv(fm.join(root, f"{fname}.csv"), index_col='iter')

    df.plot(x='iter', kind='line')
    plt.savefig(fig_path)

    print("")
