from pickletools import optimize
from random import seed
from symbol import parameters
from turtle import forward
from utils.data import get_mnist_dataset, split_train_valid
from utils.utils import MODEL_NAMES, set_all_seed
from torch.utils.data import DataLoader
from robustbench.utils import load_model
from config import Config
from utils.trainer import train_epoch, pc_train_epoch, freeze_network, MLP, CNN
from utils.eval import evaluate_acc, correct_predictions, get_ds_outputs
from utils.eval import compute_nflips
from utils.custom_loss import PCTLoss
import torch
import logging
import sys
import os
import numpy as np

for d in [Config.MODELS_DIR, Config.LOGS_PATH]:
    if not os.path.exists(d):
        os.makedirs(d)


#todo: check folders

random_seed=0
epochs=5
batch_size=200
lr=1e-3
# gamma1=1e-3
momentum=0


set_all_seed(random_seed)
device = torch.device("cuda:0" if torch.cuda.is_available()
                      else "cpu")

model_path = os.path.join(Config.MODELS_DIR, 'mnist', 'baseline.pt')

# set up logging    
logger = logging.getLogger(f'train_logger')
logger.setLevel(logging.DEBUG)
stream = logging.StreamHandler(sys.stdout)
stream.setLevel(logging.DEBUG)
logger.addHandler(stream)
fh = logging.FileHandler(os.path.join(Config.LOGS_PATH, f"prove_MNIST.log"))
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

# Prepare data
train_dataset, val_dataset = split_train_valid(
    get_mnist_dataset(train=True, shuffle=False), train_size=0.8)
test_dataset = get_mnist_dataset(train=False, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Load initial model
logger.info(f"Loading M0")
model = CNN()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
for e in range(epochs):
    train_epoch(model, device, train_loader, optimizer, e, loss_fn=torch.nn.CrossEntropyLoss())
torch.save(model.state_dict(), model_path)

model.load_state_dict(torch.load(model_path))
old_outputs = get_ds_outputs(model, train_loader, device)   #▲needed for PCT finetuning

old_preds = {}
for ds, ds_name in list(zip([train_loader, test_loader], ["Train", "Test"])):
    old_preds[ds_name] = correct_predictions(model, ds, device)
    old_acc = torch.mean(old_preds[ds_name].float()).item()

    logger.info(f"{ds_name}")
    logger.info(f"Old Acc: {old_acc}")

for beta in np.arange(0, 1, 1e-1):
    alpha = 0 if beta==0 else 1e-1
    # for gamma1 in np.arange(1e-2, 1e-1, 1e-2):
    logger.info(f"---------- Alpha: {alpha}, Beta: {beta} -------------")
    set_all_seed(random_seed)
    pct_loss_fn = PCTLoss(old_outputs, alpha1=alpha, beta1=beta)
    model = CNN()
    model.load_state_dict(torch.load(model_path))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    freeze_network(model)
    for e in range(epochs):
        pc_train_epoch(model, device, train_loader, optimizer, e, pct_loss_fn)

    for ds, ds_name in list(zip([train_loader, test_loader], ["Train", "Test"])):
        logger.info(f">>>{ds_name}")
        ft_preds = correct_predictions(model, ds, device)
        ft_acc = torch.mean(ft_preds.float()).item()
        logger.info(f"FT Acc: {ft_acc}, FT Churn: {compute_nflips(old_preds[ds_name], ft_preds)}")
    logger.info("\n")

    print("")

        
        

