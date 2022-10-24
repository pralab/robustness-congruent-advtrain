from copyreg import pickle
from pyexpat import model
from statistics import mode
from secml.utils import fm
from utils.utils import init_logger, set_all_seed, MODEL_NAMES, save_params
from robustbench.utils import load_model
import torch
from robustbench.data import load_cifar10
from generate_advx import generate_advx
from save_predictions import save_predictions, save_trainset_predictions
from evaluate import evaluate_pipeline
from positive_congruent_finetuning import finetuning_pipeline
from utils.utils import parse_args
from datetime import datetime
import robustbench.model_zoo as mzoo
from utils.utils import load_train_set
from utils.eval import predict
import pandas as pd

from utils.data import get_cifar10_dataset

SEED = 0
N_EXAMPLES = 2000
BATCH_SIZE = 500
ROOT = 'data'
EXP_FOLDER_NAME = 'data/2ksample_250steps_100batchsize_bancoprova'
EXP_FT_FOLDER_NAME = fm.join(EXP_FOLDER_NAME, 'finetuning')
DEVICE = f'cuda:0' if torch.cuda.is_available() else 'cpu'
model_names = MODEL_NAMES[:3]

set_all_seed(SEED)


# ------ LOAD CIFAR10 ------ #
test_dataset = get_cifar10_dataset(train=False, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

id=0
dirname = f'ft_prova_0_1ep_nofreeze'
# for dirname in fm.listdir(EXP_FT_FOLDER_NAME):
exp_ft_path = fm.join(EXP_FT_FOLDER_NAME, dirname)
model_path = fm.join(exp_ft_path, 'finetuned_models', f'{model_names[id]}.pt')
model = load_model(model_name=model_names[id], dataset='cifar10', threat_model='Linf')

model.load_state_dict(torch.load(model_path))

preds_old_df = pd.read_csv(fm.join(EXP_FOLDER_NAME, 'predictions', f'{model_names[id]}_predictions.csv'))
preds_old_df = preds_old_df['Clean']

preds_df = pd.read_csv(fm.join(EXP_FOLDER_NAME, 'predictions', f'{model_names[id]}_predictions.csv'))
preds_df = preds_df['Clean']

preds_ft, _ = predict(model, x=x_test, batch_size=BATCH_SIZE, device=DEVICE)
preds_ft_df = pd.Series(preds_ft)
y_true = pd.Series(y_test.tolist())

acc0 = (preds_old_df==y_true).mean()
acc1 = (preds_df==y_true).mean()
acc1_ft = (preds_ft_df==y_true).mean()
churn = ((preds_old_df==y_true) & (preds_df!=y_true)).mean()
churn_ft = ((preds_old_df==y_true) & (preds_ft_df!=y_true)).mean()

print(f"{dirname}\n" \
f"Original:\nA0: {acc0}, A1: {acc1}, C: {churn}\n" \
f"Finetuned:\nA0: {acc0}, A1: {acc1_ft}, C: {churn_ft}")

print("")



print("")
