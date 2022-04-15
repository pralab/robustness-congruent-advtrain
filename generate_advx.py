import torch
from robustbench import load_cifar10
from robustbench.utils import load_model
from robustbench.data import load_cifar10
from utils.data import get_cifar10_dataset
from torch.utils.data import DataLoader, RandomSampler
from utils.utils import set_all_seed, MODEL_NAMES, parse_args

from secml.utils import fm
from datetime import datetime
import gzip
import pickle

import numpy as np
import foolbox as fb
import time

args = parse_args()

SEED = args.seed
N_EXAMPLES = args.n_examples
EPSILON_LIST = args.epsilon_list
N_STEPS = args.n_steps
N_MODELS = args.n_models

# SEED = 0
# N_EXAMPLES = 5
# EPSILON_LIST = [8/255]
# N_STEPS = 1
# N_MODELS = 3

ROOT = 'data'
exp_folder_name = fm.join(ROOT, 'exp_prova')   # datetime.now().strftime("day-%d-%m-%Y hr-%H-%M-%S")
advx_folder = fm.join(exp_folder_name, 'advx')
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

set_all_seed(SEED)

# ------ LOAD CIFAR10 ------ #
x_test, y_test = load_cifar10(n_examples=N_EXAMPLES, data_dir='datasets/Cifar10')


if not fm.folder_exist(advx_folder):
    fm.make_folder(advx_folder)

for i, model_name in enumerate(MODEL_NAMES):
    if i >= N_MODELS:
        break

    # ------ LOAD MODEL ------ #
    print(f"Loading {model_name}")
    model = load_model(model_name=model_name, dataset='cifar10', threat_model='Linf')
    # model is already in eval mode

    # ------ COMPUTE ADVX ------ #
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    attack = fb.attacks.LinfPGD(steps=N_STEPS)

    # todo: fare un po' di debug degli attacchi, logger, verbose ecc
    start = time.time()
    # lista esterna sono gli epsilon, lista interna sono i sample
    _, advs, success = attack(fmodel, x_test, y_test, epsilons=EPSILON_LIST)
    end = time.time()
    print('Robust accuracy: {:.1%}'.format(1 - success.float().mean()))
    print(f"Took {end - start:.2f} s")

    # prendo advs[0] perchè sto usando un solo epsilon
    data = {'advx': advs[0], 'success': success}

    # ------ SAVE ADVX ------ #
    with open(fm.join(advx_folder, f'{model_name}.gz'), 'wb') as f:
        pickle.dump(data, f)


print("")