import torch
from robustbench import load_cifar10
from robustbench.utils import load_model
from robustbench.data import load_cifar10
from utils.data import get_cifar10_dataset
from torch.utils.data import DataLoader, RandomSampler
from utils.utils import set_all_seed, MODEL_NAMES, parse_args
import math
from secml.utils import fm
from datetime import datetime
import gzip
import pickle
import pandas as pd
import numpy as np
import foolbox as fb
import time


args = parse_args()

SEED = args.seed
N_EXAMPLES = args.n_examples
N_MODELS = args.n_models
BATCH_SIZE = args.batch_size

# SEED = 0
# N_EXAMPLES = 5
# N_MODELS = 3
# BATCH_SIZE = 2

ROOT = 'data'
exp_folder_name = fm.join(ROOT, 'exp_prova')
advx_folder = fm.join(exp_folder_name, 'advx')
predictions_folder = fm.join(exp_folder_name, 'predictions')

COLUMNS_NAMES = ['Clean Acc'] + MODEL_NAMES[:N_MODELS]

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

set_all_seed(SEED)

# ------ ASSERTIONS AND FOLDER MANAGEMENT ------ #
assert fm.folder_exist(exp_folder_name), 'You must run generate_advx first'

if not fm.folder_exist(predictions_folder):
    fm.make_folder(predictions_folder)

with open(fm.join(advx_folder, f'{MODEL_NAMES[0]}.gz'), 'rb') as f:
    data = pickle.load(f)
advx, _ = data['advx'], data['success']
assert N_EXAMPLES == advx.shape[0], 'number of clean samples different from number of saved advx'


# ------ LOAD CIFAR10 ------ #
# todo: qui rischio di prendere dimensioni diverse dagli advx salvati
x_test, y_test = load_cifar10(n_examples=N_EXAMPLES, data_dir='datasets/Cifar10')

# predictions_matrix = np.zeros((N_MODELS, N_MODELS+1, N_EXAMPLES))

for i, model_name_i in enumerate(MODEL_NAMES[:N_MODELS]):
    predictions_df = pd.DataFrame(columns=COLUMNS_NAMES)

    # ------ LOAD MODEL ------ #
    print(f"Loading {model_name_i}")
    model = load_model(model_name=model_name_i, dataset='cifar10', threat_model='Linf')
    # model is already in eval mode


    # evaluate model_i with every set of clean samples and advx optimized on model_j
    for j, column_name in enumerate(COLUMNS_NAMES):
        if j == 0:
            x = x_test
        else:
            # ------ LOAD ADVX ------ #
            with open(fm.join(advx_folder, f'{column_name}.gz'), 'rb') as f:
                data = pickle.load(f)
            x, _ = data['advx'], data['success']

        preds = []
        with torch.no_grad():
            for batch_i in range(math.ceil(N_EXAMPLES/BATCH_SIZE)):
                start_i = batch_i * BATCH_SIZE
                end_i = start_i + BATCH_SIZE

                out = model(x[start_i:end_i])
                pred = torch.argmax(out, axis=1)
                preds.extend(pred.tolist())
        predictions_df[column_name] = preds
        predictions_df.to_csv(fm.join(predictions_folder, f"{model_name_i}_predictions.csv"))



print("")