from secml.utils import fm
from utils.utils import init_logger, set_all_seed, MODEL_NAMES, save_params
import torch
from robustbench.data import load_cifar10
from generate_advx import generate_advx
from save_predictions import save_predictions
from evaluate import evaluate_pipeline

from utils.utils import parse_args
from datetime import datetime
import robustbench.model_zoo as mzoo
import logging

args = parse_args()

SEED = args.seed
<<<<<<< HEAD
N_EXAMPLES = args.n_examples
=======
N_EXAMPLES = 312 #args.n_examples    #50
>>>>>>> 146be9dd04e37ede9115366392e010c7b7a42240

#Attack constants
EPSILON = args.eps  #32/255
N_STEPS = args.n_steps  #30
# N_MODELS = args.n_models

#Predictions constants
BATCH_SIZE = args.batch_size    #10

ROOT = args.root    #'data'
EXP_FOLDER_NAME = 'data/2ksample_250steps_100batchsize_day-09-07-2022_hr-19-46-34'
# EXP_FOLDER_NAME = fm.join(ROOT, args.exp_name)
# EXP_FOLDER_NAME = fm.join(ROOT, f'pipeline {datetime.now().strftime("day-%d-%m-%Y hr-%H-%M-%S")}')
# EXP_FOLDER_NAME = fm.join(ROOT, f'{args.exp_name}_{datetime.now().strftime("day-%d-%m-%Y hr-%H-%M-%S")}')
EXP_FOLDER_NAME = EXP_FOLDER_NAME.replace(' ', '_')
if not fm.folder_exist(EXP_FOLDER_NAME):
    fm.make_folder(EXP_FOLDER_NAME)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
assert DEVICE != 'cpu'

# create logger
logger = init_logger(EXP_FOLDER_NAME)

<<<<<<< HEAD

set_all_seed(SEED)

model_names = MODEL_NAMES

save_params(locals().items(), EXP_FOLDER_NAME)
logger.info('Parameters saved.')

# ------ LOAD CIFAR10 ------ #
x_test, y_test = load_cifar10(n_examples=N_EXAMPLES, data_dir='datasets/Cifar10')



logger.info("Generating ADVX")
generate_advx(x_test=x_test, y_test=y_test, batch_size=BATCH_SIZE,
              model_names=model_names,
              eps=EPSILON, n_steps=N_STEPS,
              exp_folder_name=EXP_FOLDER_NAME, logger=logger, device=DEVICE)

logger.info("Computing predictions")
save_predictions(model_names=model_names, x_test=x_test, y_test=y_test,
                 batch_size=BATCH_SIZE, exp_folder_name=EXP_FOLDER_NAME,
                 device=DEVICE, logger=logger)

logger.info("Evaluations")
=======
set_all_seed(SEED)

model_names = MODEL_NAMES   #[:N_MODELS]

# ------ LOAD CIFAR10 ------ #
# x_test, y_test = load_cifar10(n_examples=N_EXAMPLES, data_dir='datasets/Cifar10')

# save_params(locals().items(), EXP_FOLDER_NAME)
# logger.info('Parameters saved.')

# logger.info("Generating ADVX")
# generate_advx(x_test=x_test, y_test=y_test,
#               model_names=model_names,
#               eps=EPSILON, n_steps=N_STEPS,
#               exp_folder_name=EXP_FOLDER_NAME, logger=logger, device=DEVICE)
#
# logger.info("Computing predictions")
# save_predictions(model_names=model_names, x_test=x_test, y_test=y_test,
#                  batch_size=BATCH_SIZE, exp_folder_name=EXP_FOLDER_NAME,
#                  device=DEVICE, logger=logger)

>>>>>>> 146be9dd04e37ede9115366392e010c7b7a42240
evaluate_pipeline(model_names=model_names,
                  exp_folder_name=EXP_FOLDER_NAME, logger=logger)


logger.info("Positive Congruent Finetuning")
for model_name in model_names:
    # ogni modello a partire dal secondo devo finetunarlo con 
    # constraint sui sample classificati correttamente
    # i modelli li salvo su una cartella 
    finetuning(model_name=model_name,
                exp_folder_name=EXP_FOLDER_NAME, logger=logger)

print("")



