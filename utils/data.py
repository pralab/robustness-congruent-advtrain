import pickle
import os
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import CIFAR10, MNIST
import torch
import math
from adv_lib.attacks.auto_pgd import apgd
from robustbench.utils import load_model
from utils.utils import set_all_seed
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.datasets import ImageNet
from sklearn.model_selection import train_test_split
import json
import torchvision

# from visualization import imshow, InvNormalize
# from utils import MODEL_NAMES

from scipy.sparse import vstack, csr_matrix
import numpy as np

# class MyTensorDataset(Dataset):
#     def __init__(self, ds_path, transforms=None):
#         self.ds_path = ds_path
#         self.transforms = transforms
#         all_samples = os.listdir(self.ds_path)
#         self.n_samples = len(all_samples)

#     def __len__(self):
#         return self.n_samples

#     def __getitem__(self, index):
#         file_path = os.path.join(self.ds_path, f"{str(index).zfill(10)}.gz")
#         with open(file_path, 'rb') as f:
#             data = pickle.load(f)
#         x, y = data[0], data[1]
#         if self.transforms is not None:
#             x = self.transforms(x)
#         return x, y


class MyTensorDataset(Dataset):
    """
    This retrieve the saved adversarial examples in the form of <sample_id>.gz
    """
    def __init__(self, ds_path, transforms=None):
        self.ds_path = ds_path
        self.transforms = transforms
        all_samples = os.listdir(self.ds_path)
        all_samples.remove('fname_to_target.json')
        self.n_samples = len(all_samples)
        with open(os.path.join(ds_path, 'fname_to_target.json')) as json_file:
            self.fname_to_target = json.load(json_file)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        fname = f"{str(index).zfill(10)}.png"
        file_path = os.path.join(self.ds_path, fname)
        x = torchvision.io.read_image(file_path)/255.
        y = self.fname_to_target[fname]
        if self.transforms is not None:
            x = self.transforms(x)
        return x, y


# def get_dataset_from_file(ds_name,
#     root='data/2ksample_250steps_100batchsize_bancoprova/advx_trset'):
#     """
#     Ora come ora è una porcheria ma questo deve pigliarmi il dataset di advx
#     """
    
#     model_name = ds_name
#     ds_name = f"advx_WB_{ds_name}"

#     dir_name = os.path.join(root, ds_name)

#     with open(f"{dir_name}.gz", 'rb') as f:
#         data = pickle.load(f)

#     norm = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     inv_norm = InvNormalize(norm)
#     base_ds = get_cifar10_dataset()
#     x0 = base_ds[2][0][None, :]
#     y0 = torch.Tensor([base_ds[2][1]])

#     from torch.utils.data import DataLoader
#     dataloader = DataLoader(base_ds, batch_size=1)
#     data = iter(dataloader).next()
#     x0,y0 = data[0], data[1]

#     model=load_model(model_name=model_name, dataset='cifar10', threat_model='Linf')
#     # model.cuda()

#     eps=0.03
#     n_iter=250
#     xadv = apgd(model, inputs=x0, 
#             labels=y0, eps=eps, norm=float('inf'), n_iter=n_iter)

#     for i, img in enumerate([x0[0], xadv[0]]):
#         imshow(img, path=f'images/{i}.png')

#     if not os.path.isdir(dir_name):
#         os.mkdir(dir_name)

#     for i, x in enumerate(data):
#         file_path = os.path.join(dir_name, f"{str(i).zfill(10)}.gz")
#         with open(file_path, 'wb') as f:
#             pickle.dump(x, f)
#     print("")


def get_cifar10_dataset(data_dir='datasets/Cifar10', train=True, 
                        shuffle=False, num_samples=None,
                        normalize=False, download=True,
                        random_seed=0):
    transform_list = [ToTensor()]
    if normalize:
        transform_list.append(
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    transform = Compose(transform_list)

    dataset = CIFAR10(root=data_dir, train=train,
                      download=download, transform=transform)
    set_all_seed(random_seed)
    if shuffle is True:
        indexes = torch.randperm(len(dataset))
    else:
        indexes = torch.arange(0, len(dataset))

    if num_samples is not None:
        indexes = indexes[:min(len(dataset), num_samples)]
    dataset = torch.utils.data.Subset(dataset, indexes)

    return dataset



def get_mnist_dataset(data_dir='datasets/MNIST', train=True, 
                        shuffle=False, num_samples=None,
                        normalize=False, download=True):
    transform_list = [ToTensor()]
    if normalize:
        transform_list.append(
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    transform = Compose(transform_list)

    dataset = MNIST(root=data_dir, train=train,
                      download=download, transform=transform)

    if shuffle is True:
        indexes = torch.randperm(len(dataset))
    else:
        indexes = torch.arange(0, len(dataset))

    if num_samples is not None:
        indexes = indexes[:min(len(dataset), num_samples)]
    dataset = torch.utils.data.Subset(dataset, indexes)

    return dataset


def get_imagenet_dataset(data_dir='datasets/imagenet/imagenet_val',
                         normalize=True, 
                         num_train_samples=45000,
                         train_size=0.8,
                         random_seed=0):
    if normalize:
        transforms = Compose([Resize(256), CenterCrop(224), ToTensor(),
                              Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])
    else:
        transforms = Compose([Resize(256), CenterCrop(224), ToTensor()])

    dataset = ImageNet(data_dir, split="val", transform=transforms)
    targets = torch.tensor(dataset.targets)

    _tr_idxs, ts_idxs = train_test_split(torch.arange(len(dataset)), 
                                        train_size=num_train_samples, 
                                        random_state=random_seed,
                                        shuffle=True, 
                                        stratify=targets)

    _train_dataset = torch.utils.data.Subset(dataset, _tr_idxs)
    _train_dataset.targets = targets[_tr_idxs]

    test_dataset = torch.utils.data.Subset(dataset, ts_idxs)
    test_dataset.targets = targets[ts_idxs]

    # Obtain validation set as 20% of train set
    tr_idxs, val_idxs = train_test_split(torch.arange(len(_train_dataset)),
                                        train_size=int(num_train_samples * train_size), 
                                        random_state=random_seed,
                                        shuffle=True, 
                                        stratify=_train_dataset.targets)
    train_dataset = torch.utils.data.Subset(_train_dataset, tr_idxs)
    train_dataset.targets = _train_dataset.targets[tr_idxs]

    validation_dataset = torch.utils.data.Subset(_train_dataset, val_idxs)
    validation_dataset.targets = _train_dataset.targets[val_idxs]

    return train_dataset, validation_dataset, test_dataset

def split_train_valid(dataset, train_size=0.8):
    if train_size is None:
        return dataset, dataset
    indexes = torch.arange(0, len(dataset))
    num_samples_train = math.ceil(len(dataset) * train_size)
    train_dataset = torch.utils.data.Subset(dataset, indexes[:num_samples_train])
    val_dataset = torch.utils.data.Subset(dataset, indexes[num_samples_train:])
    return train_dataset, val_dataset


##########################
# ANDROID
##########################

def ds_stack(X: list, y:list,
             start: int = 0, n_months: int = 12):
    X_stack = vstack(X[start: start + n_months])
    y_stack = np.hstack(y[start: start + n_months])

    idx = 0
    idxs = []
    for x in X[start: start + n_months]:
        idx += x.shape[0]
        idxs.append(idx)
    #
    # X_unstack, y_unstack = ds_unstack(X_stack, y_stack, idxs)
    #
    # s = 0
    # for i in range(n_months):
    #     s += (X_unstack[i] != X[start: start + n_months][i]).nnz
    #     s += (y_unstack[i] != y[start: start + n_months][i]).sum()
    # assert s == 0

    return X_stack, y_stack, idxs

def ds_unstack(X: csr_matrix, y: np.ndarray, idxs: list, preds: list):
    X_unstack = [X[0 if i == 0 else idxs[i-1]: idxs[i]] for i in range(len(idxs))]
    y_unstack = np.split(y, idxs)[:-1]
    preds_unstack = np.split(preds, idxs)[:-1]
    return X_unstack, y_unstack, preds_unstack

def sort_df(df, b = None, by='NFR(Sum)'):
    if b is None:
        # df = df.sort_values(by='Acc(FT)', ascending=False).drop_duplicates(['Models ID', 'Loss', 'NFR(FT)']+other_cols)
        # df = df.sort_values(by='NFR(FT)').drop_duplicates(['Models ID', 'Loss']+other_cols).sort_index()
        df = df.sort_values(by=by, ascending=True).drop_duplicates(
            ['Loss', 'AT'])

    else:
        df = df.loc[df['Hparams'].str.endswith(f"b={b}")]

    return df


if __name__ == '__main__':
    pass
    # model_name = MODEL_NAMES[0]
    #
    # for model_name in MODEL_NAMES:
    #     print(model_name)
    #     dataset = get_dataset_from_file(model_name)
    # print("")
