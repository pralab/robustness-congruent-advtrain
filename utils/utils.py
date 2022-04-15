import torch
import numpy as np
import random
import argparse

# ordinati per autoattack robust accuracy da leaderboard cifar10 linf
MODEL_NAMES = ['Kang2021Stable',
               'Rebuffi2021Fixing_70_16_cutmix_extra',
               'Gowal2021Improving_70_16_ddpm_100m']

def set_all_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', default=0, type=int)
    parser.add_argument('-n_examples', default=30, type=int)
    parser.add_argument('-epsilon_list', default=[8/255], nargs="+", type=int)
    parser.add_argument('-n_steps', default=10,  type=int)
    parser.add_argument('-n_models', default=3, type=int)
    parser.add_argument('-batch_size', default=2, type=int)
    args = parser.parse_args()
    return args

