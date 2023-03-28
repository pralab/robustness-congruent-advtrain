from utils.data import get_cifar10_dataset, split_train_valid
from utils.utils import MODEL_NAMES, set_all_seed, init_logger, save_params
from torch.utils.data import DataLoader
from robustbench.utils import load_model
from utils.trainer import pc_train_epoch, adv_pc_train_epoch, freeze_network
from utils.eval import get_ds_outputs, get_pct_results, compute_nflips
from utils.custom_loss import PCTLoss, MixedPCTLoss
import torch
import os
from datetime import datetime
import pickle
import math
import numpy as np
import argparse

# from confusion_matrix import find_candidate_model_pairs

parser = argparse.ArgumentParser()
parser.add_argument('-cuda', default=0, type=int)
parser.add_argument('-adv_tr', default=0, type=int)
parser.add_argument('-exp_name', default='exp', type=str)


def train_pct_pipeline():
    """
    this creates a folder exp_name with this structur inside 'results'
    exp_name
    |___ ...
    |___old-i_new-1 (MODEL PAIR LEVEL)
    |        |___PCT  (LOSS TYPE LEVEL)
    |        |    |___ ...
    |        |    |___a-1_b-5  (HYPERPARAMETERS LEVEL)
    |        |    |    |___checkpoints
    |        |    |    |        |____best_nfr.pt
    |        |    |    |        |____best_acc.pt
    |        |    |    |        |____last.pt
    |        |    |    |___baseline.gz  (contains dict with old/new outputs of tr set)
    |        |    |    |___results_best_acc.gz  (contains dict with ts results)
    |        |    |    |___results_best_nfr.gz
    |        |    |    |___results_last.gz
    |        |    |    |___val_perf_best_acc.gz  (contains dict with val results)
    |        |    |    |___val_perf_best_nfr.gz
    |        |    |    |___val_perf_last.gz
    |        |    |___ ...
    |        |    
    |        |___MixMSE
    |        |    |___ ...
    |        |    |___ ...
    |        |    |___ ...
    |        |    
    |        |___MixMSE(NF)
    |             |___ ...
    |             |___ ...
    |             |___ ...
    |___ ...
    |___ ...
    |___ ...
    """


    args = parser.parse_args()

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available()
                else "cpu")

    random_seed=0
    # old_model_ids=[1,2,3,4,5,6]
    # model_ids = [old_model_id + 1 for old_model_id in old_model_ids]
    # old_model_ids, model_ids = find_candidate_model_pairs()

    # old_model_ids = [1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 5, 5, 6]
    # model_ids = [4, 7, 4, 5, 7, 2, 4, 5, 6, 7, 7, 4, 7, 7]
    # old_model_ids = [2, 2, 3, 3, 3, 3, 3, 4, 5, 5, 6]
    # model_ids =     [5, 7, 2, 4, 5, 6, 7, 7, 4, 7, 7]
    old_model_ids = [3]
    model_ids = [2]

    trainable_layers = None 
    adv_training = bool(args.adv_tr)
    temporal = False
    n_tr = None  
    n_ts = None
    epochs=12
    batch_size=500
    lr=1e-3
    loss_names = ['PCT'] #, 'MixMSE']
    betas = [1, 2, 5, 10]
    alphas = [1, 1, 1, 1]
    exp_name = f"epochs-{epochs}_batchsize-{batch_size}_{args.exp_name}"
    # exp_name = "day-30-01-2023_hr-10-01-02_epochs-12_batchsize-500_ADV_TR"
    root = 'results'
    date = datetime.now().strftime("day-%d-%m-%Y_hr-%H-%M-%S")
    exp_path = os.path.join(root, f"{date}_{exp_name}")
    exp_path = os.path.join(root, exp_name)

    if not os.path.isdir(exp_path):
        os.mkdir(exp_path)

    save_params(locals().items(), exp_path, 'info')

    logger = init_logger(exp_path)

    #####################################
    # PREPARE DATA
    #####################################
    train_dataset, val_dataset = split_train_valid(
        get_cifar10_dataset(train=True, shuffle=False, num_samples=n_tr), train_size=0.8)
    test_dataset = get_cifar10_dataset(train=False, shuffle=False, num_samples=n_ts)
    # shuffle can be set to True if reference models are evaluated on the fly
    # without exploiting precomputed outputs
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #############################
    # MODEL PAIR LEVEL
    #############################
    for mid_i, (old_model_id, model_id) in enumerate(zip(old_model_ids, model_ids)):   
        model_pair_dir = f"old-{old_model_id}_new-{model_id}"
        model_pair_path = os.path.join(exp_path, model_pair_dir)

        if not os.path.isdir(model_pair_path):
            os.mkdir(model_pair_path)

        logger.info(f"------- MODELS: {old_model_id} -> {model_id} -------")
        try:
            # The architecture of the old model will be the same, I load it here
            old_model = load_model(MODEL_NAMES[old_model_id], dataset='cifar10', threat_model='Linf')         

            #############################
            # LOSS TYPE LEVEL
            #############################
            for i, loss_name in enumerate(loss_names):
                logger.info(f"------- LOSS: {loss_name} --------")
                loss_dir_path = os.path.join(model_pair_path, loss_name)      
                # NB: non mi serve fare un check per creare la cartella perchè lo faccio un livello dopo            
                
                if temporal and (old_model_id != old_model_ids[0]):
                    # Select as a reference the best previous models
                    # NB: if old_model_id is the first in the list it does not exist the finetuned version!
                    old_model_pair_dir = f"old-{old_model_ids[mid_i-1]}_new-{model_ids[mid_i-1]}"
                    old_model_pair_path = os.path.join(exp_path, old_model_pair_dir)
                    old_best_model_path_ftuned = select_model_from_validation(loss_dir_path=os.path.join(old_model_pair_path, loss_name),
                                                                        alphas=alphas, betas=betas)
                    ckpt = torch.load(old_best_model_path_ftuned)
                    old_model.load_state_dict(ckpt['model_state_dict'])
                
                # todo: si può spostare un livello indietro questo?
                model = load_model(MODEL_NAMES[model_id], dataset='cifar10', threat_model='Linf')

                # Starting test set performances
                logger.debug('Get baseline results')
                base_results = get_pct_results(new_model=model, ds_loader=test_loader, 
                                                old_model=old_model,
                                                device=device)
                old_correct = base_results['old_correct']

                logger.info(print_perf("\n>>> Starting test perf \n",
                    base_results['old_acc'], base_results['new_acc'], 
                    base_results['nfr'], base_results['pfr']))


                #############################
                # HYPERPARAMETERS LEVEL
                #############################
                for alpha, beta in zip(alphas, betas):
                    logger.info(f">>> Alpha {alpha}, Beta: {beta}")
                    params_dir = f"a-{alpha}_b-{beta}"
                    params_dir_path = os.path.join(loss_dir_path, params_dir)
                    if not os.path.isdir(params_dir_path):
                        os.makedirs(params_dir_path)                    

                    try:
                        logger.debug('Start training...')
                        #####################################
                        # TRAIN POSITIVE CONGRUENT
                        #####################################
                        model = load_model(MODEL_NAMES[model_id], dataset='cifar10', threat_model='Linf')
                        train_pct_model(model=model, old_model=old_model,
                                        train_loader=train_loader, val_loader=val_loader,
                                        epochs=epochs, loss_name=loss_name, lr=lr, random_seed=random_seed, device=device,
                                        alpha=alpha, beta=beta, trainable_layers=trainable_layers,
                                        adv_training=adv_training,
                                        logger=logger, exp_dir=params_dir_path)

                        
                        logger.debug('Evaluating finetuned model...')

                        #####################################
                        # SAVE RESULTS
                        #####################################
                        for tr_model_sel in ['last', 'best_acc', 'best_nfr']:
                            model_fname = os.path.join(params_dir_path, 'checkpoints', f"{tr_model_sel}.pt")
                            if os.path.exists(model_fname):
                                try:
                                    checkpoint = torch.load(model_fname)
                                    model.load_state_dict(checkpoint['model_state_dict'])
                                    
                                    results = get_pct_results(new_model=model, ds_loader=test_loader, 
                                                                old_correct=old_correct,
                                                                device=device)
                                    results['loss'] = checkpoint['loss'].loss_path
                                    results['orig_acc'] = base_results['new_acc']
                                    results['orig_nfr'] = base_results['nfr']
                                    results['orig_pfr'] = base_results['pfr']

                                    with open(os.path.join(params_dir_path, f"results_{tr_model_sel}.gz"), 'wb') as f:
                                        pickle.dump(results, f)
                                except Exception as e:
                                    logger.debug(f"Evaluation failed for {tr_model_sel}")


                    except Exception as e:
                        logger.debug('Training failed.')
                        logger.debug(e)
        except Exception as e:
            logger.debug(f"{model_pair_path} not computed.")
            logger.debug(e)

    logger.info("Pipeline completed :)")


def train_pct_model(model, old_model,
                    train_loader, val_loader,
                    epochs, loss_name, lr, random_seed, device, 
                    alpha, beta, exp_dir, trainable_layers=None, 
                    adv_training: bool = False, keep_best: str = 'both',
                    logger=None):
    """
    Function that performs Positive Congruent Training of a model with respect to a reference model.

    Args:
        model: instance of the model to be finetuned
        old_model: instance of the reference model from wich we want to reduce NFR
        loss_name: can be 'PCT', 'MixMSE' or 'MixMSE(NF)'
        train_loader: DataLoader for the training set
        val_loader: DataLoader for the validation set
        epochs: number of epochs
        alpha: set of pure distillation hyperparams (useful only for PCT)
        betas: set of PC distillation hyperparams (used by all three loss)
        exp_dir: path of the directory in which to save models, 
        train set baseline predictions, validation and test results.
        trainable_layers: if None train all layers, otherwise it trains 'trainable_layers' modules
            top-down from the last one.
            For example, with trainable_layers=3 will train only the last 3 modules.
        adv_training (bool): if True use adv_pc_train_epoch instead of pc_train_epoch.
        keep_best (str): set to ''acc'' to save and update a file 'best_acc.pt' inside exp_dir/checkpoints
            when the model reaches the HIGHEST accuracy encountered in the validation set.
            Set to ''nfr'' to save and update a file 'best_nfr.pt' inside the same folder when the model
            reaches the LOWEST negative flip rate encountered in the validation set.

        logger: if None use normal print, otherwise it output into a specified logger.

    """

    set_all_seed(random_seed)
    if logger is not None:
        print = logger.debug

    # Obtain outputs of old model (the reference model) and new model (the one that we train)
    try:
        # If already computed load them ...
        with open(os.path.join(exp_dir, 'baseline.gz'), 'rb') as f:
            train_outputs = pickle.load(f)
    except:
        # ... otherwise compute
        train_outputs = {}
        train_outputs['old'] = get_ds_outputs(old_model, train_loader, device).cpu()   #needed for PCT finetuning
        train_outputs['new'] = get_ds_outputs(model, train_loader, device).cpu() 
        
        with open(os.path.join(exp_dir, 'baseline.gz'), 'wb') as f:
            pickle.dump(train_outputs, f)

    old_outputs, new_outputs = train_outputs['old'].to(device), train_outputs['new'].to(device)

    set_all_seed(random_seed)
    
    # Define loss function based on loss_name
    # set mixmse to True when using MixMSE or the NF version as while adv training we need to keep
    # the new model before training and measure its outputs while training.
    # To save space and computation we avoid it for PCT.
    if loss_name == 'PCT':
        loss_fn = PCTLoss(old_output_clean=old_outputs, alpha1=alpha, beta1=beta)
        mixmse = False
    elif loss_name == 'MixMSE':
        loss_fn = MixedPCTLoss(output1=old_outputs, output2=new_outputs,
                                alpha1=alpha, beta1=beta,
                                only_nf=False)
        mixmse = True
    elif loss_name == 'MixMSE(NF)':
        loss_fn = MixedPCTLoss(output1=old_outputs, output2=new_outputs,
                                alpha1=alpha, beta1=beta,
                                only_nf=True)
        mixmse = True
    
    # Set the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Freeze layers accordingly to trainable_layers argument (see definition)
    if trainable_layers is not None:
        freeze_network(model, n_layer=trainable_layers)

    # Create a directory inside exp_path to save model checkpoints
    checkpoints_dir = os.path.join(exp_dir, 'checkpoints')
    if not os.path.isdir(checkpoints_dir):
        os.mkdir(checkpoints_dir)

    # Compute starting performances in validation set
    results = get_pct_results(new_model=model, ds_loader=val_loader, 
                            old_model=old_model, device=device)
    # Compute acc of old model
    # initialize the best acc and best nfr of the current model as the starting ones
    old_acc, best_acc, best_nfr = results['old_acc'], results['new_acc'], results['nfr']


    # Start the training loop...
    for e in range(epochs):
        if not adv_training:
            pc_train_epoch(model, device, train_loader, optimizer, e, loss_fn)
        else:
            adv_pc_train_epoch(model, old_model, device, train_loader, optimizer, e, loss_fn, mixmse)
            
        # check performance on validation
        val_results = get_pct_results(new_model=model, ds_loader=val_loader, 
                        old_model=old_model, device=device)
        acc, nfr, pfr = val_results['new_acc'], val_results['nfr'], val_results['pfr']        
        print(f"Epoch {e}, OldAcc: {old_acc*100:.3f}%, "\
                f"NewAcc: {acc*100:.3f}%, "\
                f"NFR: {nfr*100:.3f}%, "\
                f"PFR: {pfr*100:.3f}%")

        # Compact information to eventually save them
        model_data = {
            'epoch': e,
            'model_state_dict': model.cpu().state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_fn
            }

        # Save the model when highest accuracy on validation is reached
        if (acc > best_acc) and (keep_best in ('acc', 'both')):
            best_acc = acc
            torch.save(model_data, os.path.join(checkpoints_dir, f"best_acc.pt"))
            with open(os.path.join(exp_dir, 'val_perf_best_acc.gz'), 'wb') as f:
                pickle.dump(val_results, f)
        
        # Save the model when lowest NFR on validation is reached
        if (nfr < best_nfr) and (keep_best in ('nfr', 'both')):
            best_nfr = nfr
            torch.save(model_data, os.path.join(checkpoints_dir, f"best_nfr.pt"))
            with open(os.path.join(exp_dir, 'val_perf_best_nfr.gz'), 'wb') as f:
                pickle.dump(val_results, f)

    # Save the model after the last epoch
    torch.save(model_data, os.path.join(checkpoints_dir, f"last.pt"))
    with open(os.path.join(exp_dir, 'val_perf_last.gz'), 'wb') as f:
        pickle.dump(val_results, f)


def select_model_from_validation(loss_dir_path, alphas, betas):
    """
    todo: remove dependency on alphas an betas
    """
    paths, accs, nfrs = [], [], []
    for i, (alpha, beta) in enumerate(zip(alphas, betas)):
        params_dir_path = os.path.join(loss_dir_path, f"a-{alpha}_b-{beta}")
        for sel in ['best_nfr', 'best_acc', 'last']:
            fname = os.path.join(params_dir_path, f"val_perf_{sel}.gz")
            if os.path.exists(fname):
                with open(fname, 'rb') as f:
                    val_perf = pickle.load(f)
                paths.append(os.path.join(params_dir_path, f"checkpoints/{sel}.pt"))
                break
        
        accs.append(val_perf['acc'])
        nfrs.append(val_perf['nfr'])
    paths, accs, nfrs = np.array(paths), np.array(accs), np.array(nfrs)
    nfr_cond = np.where(nfrs==nfrs.min())
    paths, accs = paths[nfr_cond], accs[nfr_cond]
    acc_cond = np.where(accs==accs.max())[0][0]
    old_model_path_ftuned = paths[acc_cond].item()
    
    return old_model_path_ftuned



def print_perf(s0, oldacc, newacc, nfr, pfr):
    s = f"{s0}"\
    f"OldAcc: {oldacc*100:.3f}%\n"\
    f"NewAcc: {newacc*100:.3f}%\n"\
    f"NFR: {nfr*100:.3f}%\n"\
    f"PFR: {pfr*100:.3f}%"
    return s


if __name__ == '__main__':
    train_pct_pipeline()

