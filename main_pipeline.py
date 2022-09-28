from secml.utils import fm
from utils.utils import init_logger, set_all_seed, MODEL_NAMES, save_params
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

args = parse_args()
SEED = args.seed
N_EXAMPLES = args.n_examples
#Attack constants
EPSILON = args.eps  #32/255
N_STEPS = args.n_steps  #30
# N_MODELS = args.n_models
#Predictions constants
BATCH_SIZE = args.batch_size    #10
ROOT = args.root    #'data'

EXP_FOLDER_NAME = 'data/banco_prova_POCHIsamples'
# EXP_FOLDER_NAME = 'data/banco_prova_2ksamples'
# EXP_FOLDER_NAME = fm.join(ROOT, args.exp_name)
# EXP_FOLDER_NAME = fm.join(ROOT, f'pipeline {datetime.now().strftime("day-%d-%m-%Y hr-%H-%M-%S")}')
# EXP_FOLDER_NAME = fm.join(ROOT, f'{args.exp_name}_{datetime.now().strftime("day-%d-%m-%Y hr-%H-%M-%S")}')
EXP_FOLDER_NAME = EXP_FOLDER_NAME.replace(' ', '_')


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
assert DEVICE != 'cpu'


if not fm.folder_exist(EXP_FOLDER_NAME):
    fm.make_folder(EXP_FOLDER_NAME)
# create logger
logger = init_logger(EXP_FOLDER_NAME)

set_all_seed(SEED)

model_names = MODEL_NAMES

# save_params(locals().items(), EXP_FOLDER_NAME)
# logger.info('Parameters saved.')

# todo: vedere se differenziare N_EXAMPLES per train e test
# ------ LOAD CIFAR10 ------ #
x_test, y_test = load_cifar10(n_examples=N_EXAMPLES, data_dir='datasets/Cifar10')
# qui prendo un paio di sample dal train set di cifar10, poi vediamo se cambiarlo
x_train, y_train = load_train_set(n_examples=N_EXAMPLES, data_dir='datasets/Cifar10')


"""
STEP 1

Baseline evaluations:
- generate advx: take test set and compute all black box attacks on all models
- save_predictions: predict all models VS all generated advx
- evaluate: compute accuracy, robust accuracy, churn and robust churn and show results
"""

# logger.info("Generating ADVX")
# generate_advx(x=x_test, y=y_test, batch_size=BATCH_SIZE,
#               model_names=model_names,
#               eps=EPSILON, n_steps=N_STEPS,
#               exp_folder_name=EXP_FOLDER_NAME, logger=logger, device=DEVICE)

# logger.info("Computing predictions")
# save_predictions(model_names=model_names, x_test=x_test, y_test=y_test,
#                  batch_size=BATCH_SIZE, exp_folder_name=EXP_FOLDER_NAME,
#                  device=DEVICE, logger=logger)

# logger.info("Baseline Evaluations")
# evaluate_pipeline(model_names=model_names,
#                     exp_folder_name=EXP_FOLDER_NAME, logger=logger)


"""
STEP 2

- generate_advx: this time we generate advx from the training samples, that will be the baseline
  to reduce the robustness churn in the updated model (useful only for robust churn minimization)
- save_training_predictions: compute and save outputs of all models, 
  so that they are ready for churn-aware training. If advx=True load and predict also advx samples of the 
  training set, storing both clean outputs and advx outputs

- finetuning_pipeline: train the LAST LAYER of all models (starting from the 2nd), 
  considering 3 loss terms:
    - cross entropy for overall accuracy
    - regularizer for accuracy churn wrt to correct predictions of the previous model on clean sample 
    - regularizer for robustness churn wrt to correct predictions of the previous model on WB advx sample
"""

# codice per generare advx sul train set
# è importante usare stesso codice del test, così si ha lo stesso threat model
# logger.info("Computing Advx of train set")
# generate_advx(x=x_train, y=y_train, batch_size=BATCH_SIZE,
#               model_names=model_names,
#               eps=EPSILON, n_steps=N_STEPS,
#               exp_folder_name=EXP_FOLDER_NAME, logger=logger, device=DEVICE,
#               tr_set=True)

# logger.info("Computing outputs for train samples")
# save_trainset_predictions(model_names=model_names, x_train=x_train, y_train=y_train,
#                  batch_size=BATCH_SIZE, exp_folder_name=EXP_FOLDER_NAME,
#                  device=DEVICE, logger=logger, pred_advx=True)

logger.info("Positive Congruent Finetuning")
finetuning_pipeline(model_names=model_names, 
                    x_train=x_train, y_train=y_train,
                    batch_size=BATCH_SIZE, epochs=10,
                    exp_folder_name=EXP_FOLDER_NAME, logger=logger, device=DEVICE)


"""
STEP 3

- repeat STEP 1 considering the finetuned models and saving all in different folders *_ft
"""

# # questa run serve contro i modelli finetunati, per valutare robustezza
# logger.info("Generating ADVX")
# generate_advx(x_test=x_test, y_test=y_test, batch_size=BATCH_SIZE,
#               model_names=model_names,
#               eps=EPSILON, n_steps=N_STEPS,
#               exp_folder_name=EXP_FOLDER_NAME, logger=logger, device=DEVICE, ft_models=True)

# # logger.info("Computing predictions")
# save_predictions(model_names=model_names, x_test=x_test, y_test=y_test,
#                  batch_size=BATCH_SIZE, exp_folder_name=EXP_FOLDER_NAME,
#                  device=DEVICE, logger=logger, ft_models=True)

# # logger.info("Evaluations")
# evaluate_pipeline(model_names=model_names,
#                     exp_folder_name=EXP_FOLDER_NAME, logger=logger, ft_models=True)

print("")
