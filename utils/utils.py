import torch
import numpy as np
import random
import argparse
import logging
# from secml.utils import fm

from typing import Tuple, Optional
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import pandas as pd

import pickle
import json
import os

# ordinati dalla leaderboard su github (a fine pagina)

# #ordinati per la robustness misurata su 2k sample advx
# MODEL_NAMES = [
# 'Zhang2020Attacks',
# 'Rice2020Overfitting',
# 'Rade2021Helper_R18_ddpm',
# 'Hendrycks2019Using',
# 'Addepalli2021Towards_WRN34',
# 'Carmon2019Unlabeled',
# ]

cifar10_id = 'cifar10'
imagenet_id = 'imagenet'

EPS = {}
EPS[cifar10_id] = 8/255
EPS[imagenet_id] = 4/255

N_MAX_ADVX = {}
N_MAX_ADVX[cifar10_id] = 2000
N_MAX_ADVX[imagenet_id] = 5000


MODEL_NAMES = {}
MODEL_NAMES_SHORT = {}
MODEL_NAMES_LONG_SHORT_DICT = {}


#####################################################################
# CIFAR-10
#####################################################################
# Usati per finetuning e AT dove robustness non era strettamente crescente
MODEL_NAMES[cifar10_id] = [
'Standard',
'Engstrom2019Robustness',
'Rice2020Overfitting',
'Zhang2020Attacks',
'Hendrycks2019Using',
'Rade2021Helper_R18_ddpm',
'Addepalli2021Towards_WRN34',
'Carmon2019Unlabeled',
'Kang2021Stable',
'Gowal2020Uncovering_70_16_extra',
'Gowal2021Improving_70_16_ddpm_100m'
]

MODEL_NAMES_SHORT[cifar10_id] = ['Std.',
'Engstrom',
'Rice',
'Zhang',
'Hendrycks',
'Rade',
'Addep.',
'Carmon',
'Kang',
'Gowal2020',
'Gowal2021'
]


MODEL_NAMES_LONG_SHORT_DICT[cifar10_id] = {k: v for k, v in zip(MODEL_NAMES[cifar10_id],
                                                    MODEL_NAMES_SHORT[cifar10_id])}

#####################################################################
# IMAGENET
#####################################################################

MODEL_NAMES[imagenet_id] = [
'Salman2020Do_R18',                 # 52.95 / 25.32 (R18)
'Engstrom2019Robustness',           # 62.56 / 29.22 (R50)
'Chen2024Data_WRN_50_2',            # 68.76 / 40.60 (WR50-2)
'Liu2023Comprehensive_ConvNeXt-B',  # 76.02 / 55.82 (ConvNeXt-B)
'Liu2023Comprehensive_Swin-L',      # 78.92 / 59.56 (Swin-L)
]

# MODEL_NAMES = ['Standard', #81
# 'Engstrom2019Robustness', #53
# 'Rice2020Overfitting', #44
# 'Zhang2020Attacks', #43
# 'Rade2021Helper_R18_ddpm', #30
# 'Addepalli2021Towards_WRN34', #25
# 'Carmon2019Unlabeled', #23
# 'Hendrycks2019Using', #18
# 'Kang2021Stable', #6
# 'Gowal2020Uncovering_70_16_extra', #3
# 'Gowal2021Improving_70_16_ddpm_100m' #2
# ]

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
ADVX_IMAGENET_DIRNAME_DEFAULT = 'advx-imagenet'
custom_dirname = lambda dirname, ft_models=False, tr_set=False: f"{dirname}{'_ft' if ft_models else ''}{'_trset' if tr_set else ''}"
PREDS_DIRNAME_DEFAULT = 'predictions'
RESULTS_DIRNAME_DEFAULT = 'results'
FINETUNING_DIRNAME_DEFAULT = 'finetuned_models'
FT_DEBUG_FOLDER_DEFAULT = 'ft_debug'

COLUMN_NAMES = lambda model_names=MODEL_NAMES: ['True', 'Clean'] + model_names


def select_group(old_ids, new_ids, group_id=None):
    """
    group_id = 0 -> select old_ids 1 and 2 (5 settings)
    group_id = 1 -> select old_ids 3 (5 settings)
    group_id = 2 -> select old_ids 4, 5 and 6 (4 settings)

    """
    if group_id == 0:
        old_ids, new_ids = old_ids[:5], new_ids[:5]
    elif group_id == 1:
        old_ids, new_ids = old_ids[5:-4], new_ids[5:-4]
    elif group_id == 2:
        old_ids, new_ids = old_ids[-4:], new_ids[-4:]

    return old_ids, new_ids

def get_model_info(path='../models/model_info/cifar10\Linf',
                   id_select=tuple(range(1, 8))):
    # All info
    models_info_df = None
    for id, model_name in enumerate(MODEL_NAMES):
        if id not in id_select:
            continue
        with open(join(path, f"{model_name}.json")) as f:
            model_info = json.load(f)
        # model_info['id'] = model_name
        if models_info_df is None:
            models_info_df = pd.DataFrame(columns=model_info.keys())
            models_info_df.index.name = 'id'
        models_info_df.loc[model_name] = model_info

    return models_info_df


def get_chosen_ftmodels_path(csv_path, loss_name='MixMSE-AT'):
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df = df[df.columns[:3]]
    df = df.loc[df['Loss'] == loss_name]

    path_list = []
    for i in range(df.shape[0]):
        path_list.append(join(df.iloc[i, 0], df.iloc[i, 1], df.iloc[i, 2]))

    return path_list


def my_load(path, format='rb'):
    with open(path, format) as f:
        object = pickle.load(f)
    return object

def my_save(object, path, format='wb'):
    with open(path, format) as f:
        pickle.dump(object, f)


def model_pairs_str_to_ids(model_pair_str):
    old_id, new_id = (int(i) for i in model_pair_str.split('old-')[1].split('_new-'))
    return old_id, new_id

def join(*args):
    path = os.path.join(*args).replace('\\', '/')
    return path

def set_all_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def used_memory_percentage(device):
    return torch.cuda.memory_allocated(device) / torch.cuda.get_device_properties(device).total_memory

def str_to_hps(s):
    alpha, beta = s.split('_')
    hps = {}
    hps['alpha'] = float(alpha.split('-')[-1])
    hps['beta'] = float(beta.split('-')[-1])
    return hps
# Default
# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-seed', default=0, type=int)
#     parser.add_argument('-n_examples', default=20, type=int)
#     parser.add_argument('-n_tr_examples', default=20, type=int)
#     parser.add_argument('-eps', default=0.03, type=float)
#     parser.add_argument('-n_steps', default=5,  type=int)
#     parser.add_argument('-n_models', default=5, type=int)
#     parser.add_argument('-batch_size', default=5, type=int)
#     parser.add_argument('-root', default='data', type=str)
#     parser.add_argument('-exp_name', default='exp', type=str)
#     parser.add_argument('-exp_ft_name', default='exp_ft', type=str)
#     parser.add_argument('-cuda_id', default=0, type=int)
#     # Finetuning parameters
#     parser.add_argument('-lr', default=1e-1, type=float)
#     parser.add_argument('-epochs', default=10, type=int)
#     parser.add_argument('-gamma1', default=1, type=float)
#     parser.add_argument('-gamma2', default=0, type=float)
#     args = parser.parse_args()
#     return args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', default=0, type=int)
    parser.add_argument('-n_examples', default=200, type=int)
    parser.add_argument('-n_tr_examples', default=500, type=int)
    parser.add_argument('-eps', default=0.03, type=float)
    parser.add_argument('-n_steps', default=250,  type=int)
    parser.add_argument('-n_models', default=11, type=int)
    parser.add_argument('-batch_size', default=100, type=int)
    parser.add_argument('-root', default='data', type=str)
    parser.add_argument('-exp_name', default='exp', type=str)
    parser.add_argument('-exp_ft_name', default='exp_ft', type=str)
    parser.add_argument('-cuda_id', default=0, type=int)
    # Finetuning parameters
    parser.add_argument('-lr', default=1e-1, type=float)
    parser.add_argument('-epochs', default=100, type=int)
    parser.add_argument('-gamma1', default=1, type=float)
    parser.add_argument('-gamma2', default=0, type=float)
    args = parser.parse_args()
    return args

def init_logger(root, fname='progress', level=logging.DEBUG):
    logger = logging.getLogger(fname)
    logger.setLevel(level)

    fh = logging.FileHandler(join(root, f'{fname}.log'))
    # formatter_file = logging.Formatter('%(asctime)s - %(message)s')
    formatter = logging.Formatter('[%(asctime)s] %(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
                                  '%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    streamhandler = logging.StreamHandler()
    streamhandler.setFormatter(fh)
    logger.addHandler(streamhandler)
    return logger

def save_params(local_items, dirname, fname):
    s = ''
    for k, v in local_items:
        s += f"{k}: {v}\n"

    with open(join(dirname, f"{fname}.txt"), 'w') as f:
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

def rotate(x, theta):
    theta = np.radians(theta)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    center = x.mean(axis=0)
    x = x - center
    x_rot = R.dot(x.T).T
    x_rot = x_rot + center

    # import matplotlib.pyplot as plt
    #
    # fig, ax = plt.subplots()
    # ax.scatter(x[:, 0], x[:, 1])
    #
    # ax.scatter(x_rot[:, 0], x_rot[:, 1])
    # fig.show()
    return x_rot


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    models_info = get_model_info()
    print("")