from utils.data import get_cifar10_dataset, split_train_valid
from utils.utils import MODEL_NAMES, set_all_seed, init_logger, save_params
from torch.utils.data import DataLoader
from robustbench.utils import load_model
from config import Config
from torchvision import models
from utils.trainer import train_epoch, pc_train_epoch, freeze_network, MLP
from utils.eval import evaluate_acc, correct_predictions, get_ds_outputs, get_pct_results
from utils.eval import compute_nflips
from utils.custom_loss import PCTLoss, MixedPCTLoss, MyCrossEntropyLoss
from utils.visualization import plot_loss
import torch
import logging
import sys
import os
import numpy as np
from datetime import datetime
import pickle
import matplotlib.pyplot as plt


def train_pct_model(model, new_model, old_model, 
                    train_loader, val_loader,
                    epochs, loss_fn, lr, random_seed, device,
                    alpha, beta, exp_dir, only_nf=False,
                    logger=None):

    set_all_seed(random_seed)
    if logger is not None:
        print = logger.debug

    #####################################
    # COMPUTE BASELINE OUTPUTS AND METRICS
    #####################################

    try:
        with open(os.path.join(exp_dir, 'baseline.gz'), 'rb') as f:
            train_outputs = pickle.load(f)
    except:
        train_outputs = {}
        train_outputs['old'] = get_ds_outputs(old_model, train_loader, device)   #needed for PCT finetuning
        train_outputs['new'] = get_ds_outputs(new_model, train_loader, device)
        
        with open(os.path.join(exp_dir, 'baseline.gz'), 'wb') as f:
            pickle.dump(train_outputs, f)

    old_outputs, new_outputs = train_outputs['old'], train_outputs['new']

    set_all_seed(random_seed)
    if loss_fn is MyCrossEntropyLoss:
        loss_fn = loss_fn()
    elif loss_fn is PCTLoss:
        loss_fn = loss_fn(old_outputs, alpha1=alpha, beta1=beta)
    elif loss_fn is MixedPCTLoss:
        loss_fn = loss_fn(old_outputs, new_outputs,
                                alpha1=alpha, beta1=beta,
                                only_nf=only_nf)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # freeze_network(model)
    model = new_model

    checkpoints_dir = os.path.join(exp_dir, 'checkpoints')
    if not os.path.isdir(checkpoints_dir):
        os.mkdir(checkpoints_dir)
    
    results = get_pct_results(new_model=model, ds_loader=val_loader, 
                            old_model=old_model, device=device)

    old_acc = results['old_acc']
    best_acc = results['new_acc']
    best_nfr = results['nfr']
    for e in range(epochs):
        pc_train_epoch(model, device, train_loader, optimizer, e, loss_fn)
        
        # evaluate on validation
        results = get_pct_results(new_model=model, ds_loader=val_loader, 
                        old_model=old_model, device=device)
        acc, nfr, pfr = results['new_acc'], results['nfr'], results['pfr']
        print(f"Epoch {e}, OldAcc: {old_acc*100:.3f}%, "\
                f"NewAcc: {acc*100:.3f}%, "\
                f"NFR: {nfr*100:.3f}%, "\
                f"PFR: {pfr*100:.3f}%")

        model_data = {
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_fn,
            'perf': {'acc': acc, 'nfr': nfr, 'pfr': pfr}
            }

        if acc > best_acc:
            torch.save(model_data, os.path.join(checkpoints_dir, f"best_acc.pt"))
        
        if nfr < best_nfr:
            torch.save(model_data, os.path.join(checkpoints_dir, f"best_nfr.pt"))
    
    torch.save(model_data, os.path.join(checkpoints_dir, f"last.pt"))




def print_perf(s0, oldacc, newacc, nfr, pfr):
    s = f"{s0}"\
    f"OldAcc: {oldacc*100:.3f}%\n"\
    f"NewAcc: {newacc*100:.3f}%\n"\
    f"NFR: {nfr*100:.3f}%\n"\
    f"PFR: {pfr*100:.3f}%"
    return s





if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available()
                else "cpu")

    random_seed=0
    model_id=5
    old_model_id=4
    epochs=1
    batch_size=200
    lr=1e-3
    root = 'results'

    for old_model_id in [1, 2, 3]:
        model_id = old_model_id + 1
        # exp_dir = f'{datetime.now().strftime("day-%d-%m-%Y_hr-%H-%M-%S")}_old-{old_model_id}_new-model_id'
        exp_dir = f"old-{old_model_id}_new-{model_id}"
        exp_dir = os.path.join(root, exp_dir)
        if not os.path.isdir(exp_dir):
            os.mkdir(exp_dir)

        logger = init_logger(exp_dir)

        loss_fn = MixedPCTLoss
        betas = [1, 2, 5]
        alphas = [0]*len(betas)
        only_nf = True

        save_params(locals().items(), exp_dir, 'info')



        #####################################
        # PREPARE DATA
        #####################################
        train_dataset, val_dataset = split_train_valid(
            get_cifar10_dataset(train=True, shuffle=False, num_samples=5000), train_size=0.8)
        test_dataset = get_cifar10_dataset(train=False, shuffle=False, num_samples=2000)
        # shuffle può essere messo a True se si valuta il vecchio modello 
        # on the fly senza usare output precalcolati
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        #####################################
        # GET MODELS
        #####################################
        old_model = load_model(MODEL_NAMES[old_model_id], dataset='cifar10', threat_model='Linf')
        new_model = load_model(MODEL_NAMES[model_id], dataset='cifar10', threat_model='Linf')

        base_results = get_pct_results(new_model=new_model, ds_loader=test_loader, 
                                        old_model=old_model,
                                        device=device)
        old_correct = base_results['old_correct']

        logger.info(print_perf("\n>>> Starting test perf \n",
            base_results['old_acc'], base_results['new_acc'], 
            base_results['nfr'], base_results['pfr']))

        for i, loss_name in enumerate(['PCT', 'MixMSE', 'MixMSE(NF)']):
            if i==0:
                loss_fn = PCTLoss
            elif i==1:
                loss_fn = MixedPCTLoss
                only_nf = False
            else:
                loss_fn = MixedPCTLoss
                only_nf = True
            
            exp_dir1 = os.path.join(exp_dir, loss_name)

            for alpha, beta in list(zip(alphas, betas)):
                exp_dir2 = os.path.join(exp_dir1, f"a-{alpha}_b-{beta}")
                if not os.path.isdir(exp_dir2):
                    os.makedirs(exp_dir2)
                
                logger.info(f"------- Alpha {alpha}, Beta: {beta} -------")
                #####################################
                # TRAIN POSITIVE CONGRUENT
                #####################################
                model = new_model   # make a copy
                train_pct_model(model=model, new_model=new_model, old_model=old_model,
                                train_loader=train_loader, val_loader=val_loader,
                                epochs=epochs, loss_fn=loss_fn, lr=lr, random_seed=random_seed, device=device,
                                alpha=alpha, beta=beta, only_nf=only_nf,
                                logger=logger, exp_dir=exp_dir2)
                
                #####################################
                # SAVE RESULTS
                #####################################
                model_eval = new_model
                model_fname = os.path.join(exp_dir2, 'checkpoints', 'last.pt')
                checkpoint = torch.load(model_fname)
                model_eval.load_state_dict(checkpoint['model_state_dict'])
                
                results = get_pct_results(new_model=model, ds_loader=test_loader, 
                                            old_correct=old_correct,
                                            device=device)
                results['loss'] = checkpoint['loss']
                results['orig_acc'] = base_results['new_acc']
                results['orig_nfr'] = base_results['new_nfr']
                results['orig_pfr'] = base_results['new_pfr']

                with open(os.path.join(exp_dir2, 'results.gz')) as f:
                    pickle.dump(results, f)


