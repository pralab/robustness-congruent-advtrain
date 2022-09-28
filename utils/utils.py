import torch
import numpy as np
import random
import argparse
import logging
from secml.utils import fm

from typing import Tuple, Optional
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# ordinati dalla leaderboard su github (a fine pagina)


MODEL_NAMES = ['Standard', #81
'Engstrom2019Robustness', #53
'Rice2020Overfitting', #44
'Zhang2020Attacks', #43
'Rade2021Helper_R18_ddpm', #30
'Addepalli2021Towards_WRN34', #25
'Carmon2019Unlabeled', #23
'Hendrycks2019Using', #18
'Kang2021Stable', #6
'Gowal2020Uncovering_70_16_extra', #3
'Gowal2021Improving_70_16_ddpm_100m' #2
]

# MODEL_NAMES = ['Addepalli2021Towards_WRN34',
# 'Chan2020Jacobian',
# 'Cui2020Learnable_34_20',
# 'Engstrom2019Robustness',
# 'Hendrycks2019Using',
# 'Jang2019Adversarial',
# 'Kang2021Stable']

# MODEL_NAMES = ['Rebuffi2021Fixing_70_16_cutmix_extra',
# 'Gowal2020Uncovering_70_16_extra',
# 'Rebuffi2021Fixing_70_16_cutmix_ddpm',
# 'Gowal2021Improving_28_10_ddpm_100m',
# 'Rade2021Helper_extra',
# 'Sehwag2021Proxy_ResNest152',
# 'Dai2021Parameterizing',
# 'Rebuffi2021Fixing_28_10_cutmix_ddpm',
# 'Sehwag2021Proxy',
# 'Zhang2020Geometry',
# 'Addepalli2021Towards_WRN34',
# 'Rade2021Helper_R18_extra',
# 'Rebuffi2021Fixing_R18_ddpm',
# 'Wu2020Adversarial',
# 'Pang2020Boosting',
# 'Rice2020Overfitting',
# 'Cui2020Learnable_34_10',
# 'Addepalli2021Towards_RN18',
# 'Andriushchenko2020Understanding',
# 'Wong2020Fast']
# todo: aggiungere funzioni per scegliere il tipo di ordinamento e selezionare quanti e quali modelli

advx_fname = lambda model_name: f'advx_WB_{model_name}.gz'
preds_fname = lambda model_name: f"{model_name}_predictions.csv"
PERF_FNAME = 'rob_acc_table.csv'
NFLIPS_FNAME = 'neg_flips_table.csv'
OVERALL_RES_FNAME = 'overall_results_table.csv'

ADVX_DIRNAME_DEFAULT = 'advx'
custom_dirname = lambda dirname, ft_models=False, tr_set=False: f"{dirname}{'_ft' if ft_models else ''}{'_trset' if tr_set else ''}"
PREDS_DIRNAME_DEFAULT = 'predictions'
RESULTS_DIRNAME_DEFAULT = 'results'
FINETUNING_DIRNAME_DEFAULT = 'finetuned_models'

COLUMN_NAMES = ['True', 'Clean'] + MODEL_NAMES

def set_all_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-seed', default=0, type=int)
#     parser.add_argument('-n_examples', default=5, type=int)
#     parser.add_argument('-eps', default=0.1, type=float)
#     parser.add_argument('-n_steps', default=10,  type=int)
#     parser.add_argument('-n_models', default=10, type=int)
#     parser.add_argument('-batch_size', default=2, type=int)
#     parser.add_argument('-root', default='data', type=str)
#     parser.add_argument('-exp_name', default='exp', type=str)
#     args = parser.parse_args()
#     return args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', default=0, type=int)
    parser.add_argument('-n_examples', default=20, type=int)
    parser.add_argument('-eps', default=0.03, type=float)
    parser.add_argument('-n_steps', default=5,  type=int)
    parser.add_argument('-n_models', default=5, type=int)
    parser.add_argument('-batch_size', default=5, type=int)
    parser.add_argument('-root', default='data', type=str)
    parser.add_argument('-exp_name', default='exp', type=str)
    args = parser.parse_args()
    return args

def init_logger(root):
    logger = logging.getLogger('progress')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(fm.join(root, 'progress.log'))
    # formatter_file = logging.Formatter('%(asctime)s - %(message)s')
    formatter = logging.Formatter('[%(asctime)s] %(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
                                  '%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    streamhandler = logging.StreamHandler()
    streamhandler.setFormatter(fh)
    logger.addHandler(streamhandler)
    return logger

def save_params(local_items, dirname):
    s = ''
    for k, v in local_items:
        s += f"{k}: {v}\n"

    with open(fm.join(dirname, "info.txt"), 'w') as f:
        f.write(s)



def load_train_set(
        n_examples: Optional[int] = None,
        data_dir: str = './data') -> Tuple[torch.Tensor, torch.Tensor]:
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root=data_dir,
                            train=True,
                            transform=transform,
                            download=True)

    batch_size = 100
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=False, num_workers=0)


    x_tr, y_tr = [], []
    for i, (x, y) in enumerate(train_loader):
        x_tr.append(x)
        y_tr.append(y)
        if n_examples is not None and batch_size * i >= n_examples:
            break
    x_tr_tensor = torch.cat(x_tr)
    y_tr_tensor = torch.cat(y_tr)

    if n_examples is not None:
        x_tr_tensor = x_tr_tensor[:n_examples]
        y_tr_tensor = y_tr_tensor[:n_examples]

    return x_tr_tensor, y_tr_tensor