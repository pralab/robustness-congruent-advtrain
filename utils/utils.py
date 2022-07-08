import torch
import numpy as np
import random
import argparse
import logging
from secml.utils import fm

# ordinati per autoattack robust accuracy da leaderboard cifar10 linf, (l'ultimo è il più top)
MODEL_NAMES = ['Standard',
               'Sehwag2020Hydra',
               'Gowal2020Uncovering_28_10_extra']
# todo: aggiungere funzioni per scegliere il tipo di ordinamento e selezionare quanti e quali modelli

advx_fname = lambda model_name: f'advx_WB_{model_name}.gz'
preds_fname = lambda model_name: f"{model_name}_predictions.csv"
PERF_FNAME = 'performance_table.csv'
NFLIPS_FNAME = 'neg_flips_table.csv'

ADVX_DIRNAME_DEFAULT = 'advx'
PREDS_DIRNAME_DEFAULT = 'predictions'
RESULTS_DIRNAME_DEFAULT = 'results'

COLUMN_NAMES = ['True', 'Clean'] + MODEL_NAMES

def set_all_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', default=0, type=int)
    parser.add_argument('-n_examples', default=5, type=int)
    parser.add_argument('-eps', default=0.1, type=float)
    parser.add_argument('-n_steps', default=10,  type=int)
    parser.add_argument('-n_models', default=10, type=int)
    parser.add_argument('-batch_size', default=2, type=int)
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