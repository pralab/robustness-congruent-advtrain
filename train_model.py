from utils.data import get_cifar10_dataset, split_train_valid, get_imagenet_dataset
from utils.trainer import pc_train_epoch, adv_pc_train_epoch, freeze_network, fgsm_attack
from utils.eval import get_ds_outputs, get_pct_results, compute_nflips, compute_pflips, compute_common_nflips, correct_predictions
from utils.custom_loss import PCTLoss, MixedPCTLoss
from utils.data import MyTensorDataset
from utils.visualization import show_hps_behaviour, plot_loss
import utils.utils as ut

from generate_advx import generate_advx, check_baseline
from manage_files import delete_advx_ts

import torch
from torch.utils.data import DataLoader
from robustbench.utils import load_model

import os
import pickle
import logging
import argparse
from copy import deepcopy
from datetime import datetime

import math
import numpy as np
import matplotlib.pyplot as plt

# from confusion_matrix import find_candidate_model_pairs

# class EarlyStopping:
#     def __init__(self, tolerance=5, min_delta=0):

#         self.tolerance = tolerance
#         self.min_delta = min_delta
#         self.counter = 0
#         self.early_stop = False

#     def __call__(self, train_loss, previous_train_loss):
#         if (previous_train_loss - train_loss) < self.min_delta:
#             self.counter +=1
#             if self.counter >= self.tolerance:  
#                 self.early_stop = True


def train_pct_model(model, old_model,
                    train_loader, val_loader,
                    epochs, loss_name, lr, 
                    random_seed, device, 
                    alpha, beta, exp_dir,
                    optim = 'sgd',
                    adv_training: bool = False,
                    ds_id=ut.cifar10_id,
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

    ut.set_all_seed(random_seed)
    # if logger is not None:
    #     logger.debug = print

    # Obtain outputs of old model (the reference model) and new model (the one that we train)
    if ds_id == ut.cifar10_id:
        try:
            # If already computed load them ...
            with open(os.path.join(exp_dir, 'baseline.gz'), 'rb') as f:
                train_outputs = pickle.load(f)
        except:
            # ... otherwise compute
            train_outputs = {}
            train_outputs['old'] = get_ds_outputs(old_model, train_loader, device).cpu()   #needed for PCT finetuning
            train_outputs['new'] = get_ds_outputs(model, train_loader, device).cpu() 
            os.makedirs(exp_dir, exist_ok=True)
            with open(os.path.join(exp_dir, 'baseline.gz'), 'wb') as f:
                pickle.dump(train_outputs, f)

        old_outputs, new_outputs = train_outputs['old'].to(device), train_outputs['new'].to(device)

    else:
        old_outputs, new_outputs = None, None   # TODO: modified as train_loader has shuffle=True

    # ut.set_all_seed(random_seed)

    # try:
    #     # If already computed load them ...
    #     with open(os.path.join(exp_dir, 'val_baseline.gz'), 'rb') as f:
    #         val_outputs = pickle.load(f)
    # except:
    #     # ... otherwise compute
    #     train_outputs = {}
    #     train_outputs['old'] = get_ds_outputs(old_model, train_loader, device).cpu()   #needed for PCT finetuning
    #     train_outputs['new'] = get_ds_outputs(model, train_loader, device).cpu() 
    #     os.makedirs(exp_dir, exist_ok=True)
    #     with open(os.path.join(exp_dir, 'baseline.gz'), 'wb') as f:
    #         pickle.dump(train_outputs, f)

    # old_outputs, new_outputs = train_outputs['old'].to(device), train_outputs['new'].to(device)

    # ut.set_all_seed(random_seed)
    
    # Define loss function based on loss_name
    # set mixmse to True when using MixMSE or the NF version as while adv training we need to keep
    # the new model before training and measure its outputs while training.
    # To save space and computation we avoid it for PCT.
    if loss_name == 'PCT':
        loss_fn = PCTLoss(old_output_clean=old_outputs, alpha=alpha, beta=beta)
        # val_loss_fn = PCTLoss(old_output=old_outputs, new_output=new_outputs,
        #                       alpha=alpha, beta=beta)
        mixmse = False
        new_model = None
    elif loss_name == 'MixMSE':
        loss_fn = MixedPCTLoss(old_output=old_outputs, new_output=new_outputs,
                                alpha=alpha, beta=beta,
                                only_nf=False)
        # val_loss_fn = MixedPCTLoss(old_output=old_outputs, new_output=new_outputs,
        #                            alpha=alpha, beta=beta, only_nf=False)
        if adv_training:
            new_model = deepcopy(model).to(device)
            new_model.eval()
        mixmse = True
    
    # Set the optimizer
    if optim == 'adam':
        logger.debug(f"Using Adam optimizer.")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        # set SGD by default
        logger.debug(f"Using SGD optimizer.")
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)


    # todo: rendere opzioni scheduler tramite parser
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
    #                                                        factor=0.1, patience=3, 
    #                                                        threshold=0.001, 
    #                                                        threshold_mode='abs')

    # Create a directory inside exp_path to save model checkpoints
    checkpoints_dir = os.path.join(exp_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    ut.set_all_seed(random_seed)
    # TODO: cosa ci faceva questo?
    # val_loader = DataLoader(val_loader.dataset, batch_size=train_loader.batch_size, shuffle=False)
    
    # best_val_loss = math.inf
    # best_epoch = 0
    # patience_lr = 5
    # min_lr = 1e-4
    # patience_early_stopping = 10
    # margin = 0.005
    # cnt = 0
    # running_lr = lr

    eps = ut.EPS[ds_id]
    # Start the training loop...
    for e in range(epochs):
        if not adv_training:
            pc_train_epoch(model=model, old_model=old_model, device=device, train_loader=train_loader, 
                    optimizer=optimizer, epoch=e, loss_fn=loss_fn, logger=logger)
            
        else:
            adv_pc_train_epoch(model=model, old_model=old_model, device=device, train_loader=train_loader, 
                    optimizer=optimizer, epoch=e, loss_fn=loss_fn, new_model=new_model, mixmse=mixmse,
                    eps=eps, logger=logger)
            
        model_data = {
            'epoch': e,
            'model_state_dict': model.cpu().state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_fn
            }
    
    torch.save(model_data, os.path.join(checkpoints_dir, f"last.pt"))
    #     # Evaluate loss on validation set
    #     logger.info('Checking loss on validation set...')
    #     full_val_loss = 0
    #     for i, (data, target) in enumerate(val_loader):
    #         data, target = data.to(device), target.to(device)
    #         if adv_training:
    #             data = fgsm_attack(x=data, target=target, model=model, epsilon=8/255)
            
    #         with torch.no_grad():
    #             out = model(data)
    #             old_out = old_model(data)
    #             if mixmse:
    #                 new_out = new_model(data)
    #                 val_loss = val_loss_fn(model_output=out, target=target, 
    #                                     old_output=old_out, new_output=new_out)
    #             else:
    #                 val_loss = val_loss_fn(model_output=out, target=target, old_output=old_out)
    #             full_val_loss += val_loss[0].item() / len(val_loader)
    #         logger.debug(f"[{i}/{len(val_loader)}] -  / "\
    #     f"tot:{val_loss[0]:.3f}, ce:{val_loss[1]:.3f}, dist: {val_loss[2]:.3f}, foc: {val_loss[3]:.3f}, ")

    #     # scheduler.step()
    #     # logger.debug(f"LR = {scheduler.get_last_lr()}")
    #     if (best_val_loss - full_val_loss <= margin):
    #         cnt += 1
    #         logger.debug(f"Increase counter. Loss not significantly improved.")
    #     else:
    #         best_val_loss = full_val_loss
    #         best_epoch = e
    #         cnt = 0
    #         model_data = {
    #             'epoch': e,
    #             'model_state_dict': model.cpu().state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'loss': loss_fn,
    #             'val_loss': val_loss_fn
    #             }
    #         torch.save(model_data, os.path.join(checkpoints_dir, f"last.pt"))
    #         logger.debug(f"New checkpoint. Best epoch: {best_epoch}. LR: {running_lr}, Best Val loss: {best_val_loss}")
        

    #     if cnt >= patience_lr:
    #         running_lr /= 2
    #         for g in optimizer.param_groups:
    #             g['lr'] = running_lr
    #         logger.debug(f"LR = {running_lr}")
    #     if cnt >= patience_early_stopping:
    #         logger.debug(f'Early stopping at epoch {e}')
    #         break
    # logger.debug(f"Training completed. Best epoch: {best_epoch}. LR: {running_lr}, Best Val loss: {best_val_loss}")
        # loss = np.array(loss_fn.loss_path['tot'])[-20:].mean()
        # scheduler.step(loss)
        # check performance on validation
        # val_results = get_pct_results(new_model=model, ds_loader=val_loader, 
        #                 old_model=old_model, device=device)
        # acc, nfr, pfr = val_results['new_acc'], val_results['nfr'], val_results['pfr']        
        # print(f"Epoch {e}, OldAcc: {old_acc*100:.3f}%, "\
        #         f"NewAcc: {acc*100:.3f}%, "\
        #         f"NFR: {nfr*100:.3f}%, "\
        #         f"PFR: {pfr*100:.3f}%")
        # if writer is not None:
        #     writer.add_scalar('Accuracy/validation', acc, e)
        #     writer.add_scalar('Accuracy/validation', acc, e)

        # Compact information to eventually save them


        # # Save the model when highest accuracy on validation is reached
        # if (acc > best_acc) and (keep_best in ('acc', 'both')):
        #     best_acc = acc
        #     torch.save(model_data, os.path.join(checkpoints_dir, f"best_acc.pt"))
        #     with open(os.path.join(exp_dir, 'val_perf_best_acc.gz'), 'wb') as f:
        #         pickle.dump(val_results, f)
        
        # # Save the model when lowest NFR on validation is reached
        # if (nfr < best_nfr) and (keep_best in ('nfr', 'both')):
        #     best_nfr = nfr
        #     torch.save(model_data, os.path.join(checkpoints_dir, f"best_nfr.pt"))
        #     with open(os.path.join(exp_dir, 'val_perf_best_nfr.gz'), 'wb') as f:
        #         pickle.dump(val_results, f)

    # Save the model after the last epoch
    # torch.save(model_data, os.path.join(checkpoints_dir, f"last.pt"))
    # with open(os.path.join(exp_dir, 'val_perf_last.gz'), 'wb') as f:
    #     pickle.dump(val_results, f)


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



def train_pct_pipeline(args):
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
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available()
                else "cpu")

    exp_name = f"{args.exp_name}"    
    date = datetime.now().strftime("day-%d-%m-%Y_hr-%H-%M-%S")
    exp_path = os.path.join(args.root, exp_name if not args.date else f"{date}_{exp_name}") 

    # exp_name = "prova_DEBUG_ADV"
    # exp_path = os.path.join(root, exp_name)

    os.makedirs(exp_path, exist_ok=True)

    if not args.test_only:
        ut.save_params(locals().items(), exp_path, 'info')

    logger_fname = f'train_{args.exp_name}-cuda_{args.cuda}' if not args.test_only else f'test_{args.exp_name}-cuda_{args.cuda}'
    logger = ut.init_logger(exp_path, fname=logger_fname, level=logging.DEBUG)

    #####################################
    # PREPARE DATA
    #####################################
    ut.set_all_seed(args.random_seed)
    ds_id = args.dataset
    logger.info(f"------- DATASET: {ds_id} -------") 
    if ds_id == ut.cifar10_id:
        train_dataset, val_dataset = split_train_valid(
            get_cifar10_dataset(train=True, shuffle=True, num_samples=args.n_tr, 
                                random_seed=args.random_seed), train_size=0.8)
        test_dataset = get_cifar10_dataset(train=False, shuffle=False, num_samples=args.n_ts)
    else:
        train_dataset, val_dataset, test_dataset = get_imagenet_dataset(normalize=False)
    # shuffle can be set to True if reference models are evaluated on the fly
    # without exploiting precomputed outputs
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    #############################
    # MODEL PAIR LEVEL
    #############################
    for mid_i, (old_model_id, model_id) in enumerate(zip(args.old_model_ids, args.model_ids)):
          
        logger.info(f"------- MODELS: {old_model_id} -> {model_id} -------") 
        model_pair_dir = f"old-{old_model_id}_new-{model_id}"
        model_pair_path = os.path.join(exp_path, model_pair_dir)
        os.makedirs(model_pair_path, exist_ok=True)

        # Starting test set performances
        logger.debug('Get baseline results')
        
        #############################
        # LOSS TYPE LEVEL
        #############################
        for _loss_name in args.loss_names:
            logger.info(f"------- LOSS: {_loss_name} --------")
            loss_dir_path = os.path.join(model_pair_path, _loss_name)      
            # NB: non mi serve fare un check per creare la cartella perchè lo faccio un livello dopo            
            adv_at_sel = 'AT' in _loss_name
            loss_name = _loss_name.split('-AT')[0]
            #############################
            # HYPERPARAMETERS LEVEL
            #############################
            
            alphas = args.alphas_pct if 'PCT' in loss_name else args.alphas_mix
            betas = args.betas_pct if 'PCT' in loss_name else args.betas_mix
            
            for alpha, beta in zip(alphas, betas):
                try:
                    # if loss_name=='PCT':
                    #     alpha, beta = int(alpha), int(beta)
                    logger.info(f">>> Alpha {alpha}, Beta: {beta}")
                    params_dir = f"a-{alpha}_b-{beta}"
                    params_dir_path = os.path.join(loss_dir_path, params_dir)

                    os.makedirs(params_dir_path, exist_ok=True)

                    try:
                        #####################################
                        # TRAIN POSITIVE CONGRUENT
                        #####################################
                        old_model = load_model(ut.MODEL_NAMES[ds_id][old_model_id], dataset=ds_id, threat_model='Linf').to(device)
                        model = load_model(ut.MODEL_NAMES[ds_id][model_id], dataset=ds_id, threat_model='Linf').to(device)
                        
                        if not args.test_only:
                            logger.debug('Start training...')
                            train_pct_model(model=model, old_model=old_model,
                                            train_loader=train_loader, val_loader=val_loader,
                                            epochs=args.epochs, optim=args.optim,
                                            loss_name=loss_name, lr=args.lr, random_seed=args.random_seed, device=device,
                                            alpha=alpha, beta=beta, ds_id=ds_id,
                                            adv_training=adv_at_sel,#args.adv_tr,
                                            logger=logger, exp_dir=params_dir_path)#, writer=writer)

                    except Exception as e:
                        logger.debug('Training failed.')
                        logger.debug(e)
                    
                    
                    # If trained model has not been saved skip and try next configuration
                    # Otherwise start the evaluation
                    model_fname = os.path.join(params_dir_path, 'checkpoints', "last.pt")
                    if not os.path.exists(model_fname):
                        logger.debug(f"Pretrained model does not exist: {model_fname}")
                        continue

                    checkpoint = torch.load(model_fname, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    
                    # for ds_loader, ds_name in zip([val_loader, test_loader], ['val', 'test']):    
                    # for ds_loader, ds_name in zip([test_loader], ['test']):       
                    for ds_loader, ds_name in zip([test_loader, val_loader], ['test', 'val']):                        
                        results_fname = os.path.join(params_dir_path, f"results_{ds_name}.gz")
                        clean_results_fname = os.path.join(params_dir_path, f"results_clean_{ds_name}.gz")
                        adv_results_fname = os.path.join(params_dir_path, f"results_advx_{ds_name}.gz")
                        
                        def check(res_fname):
                            res = None
                            res_check, cln_check, adv_check = False, False, False
                            if os.path.exists(res_fname):
                                with open(res_fname, 'rb') as f:
                                    res = pickle.load(f)
                                res_check = True
                                cln_check = True if 'clean' in res.keys() else False
                                adv_check = True if 'advx' in res.keys() else False
                            return res, res_check, cln_check, adv_check
                        
                        loaded_results, results_check, clean_check, advx_check = check(results_fname)

                        if args.test_overwrite:
                            clean_check, advx_check = False, False

                                
                        if not clean_check:
                            #####################################
                            # SAVE CLEAN RESULTS
                            #####################################
                            logger.info(f'Evaluating {ds_name} set ...')
                            old_correct_clean = check_baseline(old_model_id, ds_name, logger, 
                                                               args.random_seed, ds_id,
                                                               cuda_id=args.cuda, batch_size=args.test_batch_size, 
                                                               sel_advx=False)
                            new_correct_clean = check_baseline(model_id, ds_name, logger, 
                                                               args.random_seed, ds_id,
                                                               cuda_id=args.cuda, batch_size=args.test_batch_size, 
                                                               sel_advx=False)
                            
                            # Get results of model M wrt M0 and M1
                            ut.set_all_seed(args.random_seed)
                            clean_results = get_pct_results(new_model=model, ds_loader=ds_loader, 
                                                        old_correct=old_correct_clean,
                                                        device=device)
                            # Add baseline results for comparison
                            clean_results['loss'] = checkpoint['loss'].loss_path
                            clean_results['orig_acc'] = new_correct_clean.cpu().numpy().mean()
                            clean_results['orig_nfr'] = compute_nflips(old_correct_clean, new_correct_clean)
                            clean_results['orig_pfr'] = compute_pflips(old_correct_clean, new_correct_clean)
                            with open(clean_results_fname, 'wb') as f:
                                pickle.dump(clean_results, f)
                                
                            clean_check = True
                        else:
                            logger.debug(f"Clean results already computed for {ds_name}")
                            if not os.path.exists(clean_results_fname):
                                with open(clean_results_fname, 'wb') as f:
                                    pickle.dump(loaded_results['clean'], f)                                
                            with open(clean_results_fname, 'rb') as f:
                                clean_results = pickle.load(f)
                        
                        
                        if not advx_check:
                            #####################################
                            # SAVE ADVX RESULTS
                            #####################################
                            try:
                                logger.debug(f'Generating advx on {ds_name} set ...')
                                adv_dir_path = os.path.join(params_dir_path, 'advx', ds_name)

                                ut.set_all_seed(args.random_seed)
                                generate_advx(model=model, ds_loader=ds_loader, n_steps=args.n_steps, 
                                            adv_dir_path=adv_dir_path,
                                            logger=logger, device=device,
                                            n_max_advx_samples=args.n_adv_ts,
                                            eps=ut.EPS[ds_id])                            
                                adv_ds = MyTensorDataset(ds_path=adv_dir_path)
                                adv_ds_loader = DataLoader(adv_ds, batch_size=ds_loader.batch_size)

                                old_correct_adv = check_baseline(old_model_id, ds_name, logger, 
                                                                args.random_seed, ds_id,
                                                                cuda_id=args.cuda, batch_size=args.test_batch_size, 
                                                                sel_advx=True, n_max_advx=args.n_adv_ts, eps=None)
                                new_correct_adv = check_baseline(model_id, ds_name, logger, 
                                                                args.random_seed, ds_id,
                                                                cuda_id=args.cuda, batch_size=args.test_batch_size, 
                                                                sel_advx=True)

                                # todo: dopo aver pigliato i correct veri per test o ts
                                # uso le predizioni già salvate e le confronto con la baseline (da calcolare a parte con generate_baseline_advx)
                                
                                # Get results of model M wrt M0 and M1
                                ut.set_all_seed(args.random_seed)
                                adv_results = get_pct_results(new_model=model, ds_loader=adv_ds_loader, 
                                                            old_correct=old_correct_adv,
                                                            device=device)
                                # Add baseline results for comparison
                                adv_results['orig_acc'] = new_correct_adv.cpu().numpy().mean()
                                adv_results['orig_nfr'] = compute_nflips(old_correct_adv, new_correct_adv)
                                adv_results['orig_pfr'] = compute_pflips(old_correct_adv, new_correct_adv)
                                
                                with open(adv_results_fname, 'wb') as f:
                                    pickle.dump(adv_results, f)
                                
                                # delete_advx_ts(params_dir_path)
                                advx_check = True

                            except Exception as e:
                                logger.debug(f"Advx generation on {ds_name} gone wrong. {e}")
                        else:
                            logger.debug(f"Advx results already computed for {ds_name}")                         
                            if not os.path.exists(adv_results_fname):
                                with open(adv_results_fname, 'wb') as f:
                                    pickle.dump(loaded_results['advx'], f)    
                            with open(adv_results_fname, 'rb') as f:
                                adv_results = pickle.load(f)
                        
                        
                        if clean_check and advx_check:
                            results = {}
                            results['clean'] = clean_results
                            results['advx'] = adv_results
                            with open(os.path.join(params_dir_path, f"results_{ds_name}.gz"), 'wb') as f:
                                pickle.dump(results, f)
                        
                            if ds_name=='test':
                                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                                plot_loss(results['clean']['loss'], ax, window=20)
                                ax.set_title(_loss_name)
                                fig.savefig(os.path.join(params_dir_path, "loss_path.pdf"))
                            # else:
                            #     fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                            #     plot_loss(results['clean']['val_loss'], ax)
                            #     # fig.savefig(os.path.join(params_dir_path, "val_loss_path.pdf")) 
                            
                            logger.info(f">>> Clean Results")
                            logger.info(f"Old Acc: {results['clean']['old_acc']:.4f}")
                            logger.info(f"New Acc: {results['clean']['orig_acc']:.4f}, New Acc(FT): {results['clean']['new_acc']:.4f}")
                            logger.info(f"New NFR: {results['clean']['orig_nfr']:.4f}, New NFR(FT): {results['clean']['nfr']:.4f}")
                            logger.info(f">>> Advx Results")
                            logger.info(f"Old Acc: {results['advx']['old_acc']:.4f}")
                            logger.info(f"New Acc: {results['advx']['orig_acc']:.4f}, New Acc(FT): {results['advx']['new_acc']:.4f}")
                            logger.info(f"New NFR: {results['advx']['orig_nfr']:.4f}, New NFR(FT): {results['advx']['nfr']:.4f}")
                            logger.info("")
                            with open(os.path.join(params_dir_path, f"{ds_name}_perf.txt"), 'w') as f:
                                f.write(f">>> Clean Results\n")
                                f.write(f"Old Acc: {results['clean']['old_acc']:.4f}\n")
                                f.write(f"New Acc: {results['clean']['orig_acc']:.4f}, New Acc(FT): {results['clean']['new_acc']:.4f}\n")
                                f.write(f"New NFR: {results['clean']['orig_nfr']:.4f}, New NFR(FT): {results['clean']['nfr']:.4f}\n")
                                f.write(f">>> Advx Results\n")
                                f.write(f"Old Acc: {results['advx']['old_acc']:.4f}\n")
                                f.write(f"New Acc: {results['advx']['orig_acc']:.4f}, New Acc(FT): {results['advx']['new_acc']:.4f}\n")
                                f.write(f"New NFR: {results['advx']['orig_nfr']:.4f}, New NFR(FT): {results['advx']['nfr']:.4f}\n")
                        else:
                            logger.debug(f"Clean check: {clean_check}, Advx check: {advx_check}, Results cannot be saved jointly.")
                    logger.info("")
                except Exception as e:
                    logger.debug("Something went wrong ...")
                    logger.debug(e)
                    

                
            #     with open(os.path.join(params_dir_path, f"results_test.gz"), 'rb') as f:
            #         results = pickle.load(f)

            #     fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            #     plot_loss(results['clean']['loss'], ax)
            #     fig.savefig(os.path.join(params_dir_path, "loss_path.pdf"))
            #     with open(os.path.join(params_dir_path, f"test_perf.txt"), 'w') as f:
            #         f.write(f">>> Clean Results\n")
            #         f.write(f"Old Acc: {results['clean']['old_acc']}\n")
            #         f.write(f"New Acc: {results['clean']['orig_acc']}, New Acc(FT): {results['clean']['new_acc']}\n")
            #         f.write(f"New NFR: {results['clean']['orig_nfr']}, New NFR(FT): {results['clean']['nfr']}\n")
            #         f.write(f">>> Advx Results\n")
            #         f.write(f"Old Acc: {results['advx']['old_acc']}\n")
            #         f.write(f"New Acc: {results['advx']['orig_acc']}, New Acc(FT): {results['advx']['new_acc']}\n")
            #         f.write(f"New NFR: {results['advx']['orig_nfr']}, New NFR(FT): {results['advx']['nfr']}\n")
            # # show_hps_behaviour(root=loss_dir_path, fig_path=os.path.join('images', 'MixMSE.png'))


    logger.info("Pipeline completed :)")
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    

    parser.add_argument('-exp_name', default='DEBUG', type=str)
    parser.add_argument('-root', default='results', type=str)

    parser.add_argument('-n_tr', default=45000, type=int)   
    parser.add_argument('-n_ts', default=5000, type=int) 

    parser.add_argument('-epochs', default=10, type=int)  
    parser.add_argument('-lr', default=1e-3, type=float)
    parser.add_argument('-batch_size', default=128, type=int)  
    parser.add_argument('-optim', default='sgd', type=str) 
    parser.add_argument('-test_batch_size', default=512, type=int)    
    
    parser.add_argument('-n_steps', default=50, type=int, help='number of attack steps during robusntess evaluation')
    parser.add_argument('-n_adv_ts', default=10000, type=int, help='number of advx used for robustness evaluation')
    
    parser.add_argument('-old_model_ids', default=[0], type=int, nargs='+')
    parser.add_argument('-model_ids', default=[1], type=int, nargs='+')
    parser.add_argument('-loss_names', default=['PCT', 'PCT-AT', 'MixMSE-AT'], type=str, nargs='+')
    parser.add_argument('-alphas_mix', default=[0.5], type=float, nargs='+')
    parser.add_argument('-betas_mix', default=[0.4], type=float, nargs='+')
    parser.add_argument('-alphas_pct', default=[1.], type=float, nargs='+')
    parser.add_argument('-betas_pct', default=[2.], type=float, nargs='+')

    
    parser.add_argument('-date', action='store_true')
    parser.add_argument('-test_only', action='store_true')
    parser.add_argument('-test_overwrite', action='store_true')

    parser.add_argument('-dataset', default=ut.cifar10_id, type=str, 
                        choices=[ut.cifar10_id, ut.imagenet_id])
    
    parser.add_argument('-random_seed', default=0, type=int)
    parser.add_argument('-cuda', default=0, type=int)
    
    
    args = parser.parse_args()
    
    train_pct_pipeline(args)











    # old_model_ids=[1,2,3,4,5,6]
    # model_ids = [old_model_id + 1 for old_model_id in old_model_ids]
    # old_model_ids, model_ids = find_candidate_model_pairs()

    # All 14 combinations of models with both increasing clean and robust accuracy
    # old_model_ids = [1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 5, 5, 6]
    # model_ids = [4, 7, 4, 5, 7, 2, 4, 5, 6, 7, 7, 4, 7, 7]

    # betas = np.round(np.arange(0, 1, 0.1), 2) #[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # alphas = 0.9 - betas
    
    # n_tr = None  
    # n_ts = None
    # epochs=12
    # batch_size=500
    # lr=1e-3
    # loss_names = ['PCT', 'MixMSE']