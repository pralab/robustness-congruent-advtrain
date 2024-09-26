from utils.data import get_cifar10_dataset, split_train_valid
from utils.utils import MODEL_NAMES, set_all_seed, init_logger, save_params
from torch.utils.data import DataLoader
from robustbench.utils import load_model
from utils.trainer import pc_train_epoch, adv_pc_train_epoch, freeze_network
from utils.eval import get_ds_outputs, get_pct_results, compute_nflips, compute_pflips, compute_common_nflips, correct_predictions
from utils.custom_loss import PCTLoss, MixedPCTLoss
import torch
import os
from datetime import datetime
import pickle
import math
import numpy as np
import argparse
from utils.data import MyTensorDataset
from utils.visualization import show_hps_behaviour, plot_loss
import matplotlib.pyplot as plt
from copy import deepcopy


from generate_advx import generate_advx, generate_baseline_advx, check_baseline_advx
from manage_files import delete_advx_ts

# from torch.utils.tensorboard import SummaryWriter

# from confusion_matrix import find_candidate_model_pairs


def train_pct_model(model, old_model,
                    train_loader, val_loader,
                    epochs, loss_name, lr, random_seed, device, 
                    alpha, beta, exp_dir, trainable_layers=None, 
                    adv_training: bool = False, keep_best: str = 'both', writer=None,
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
        loss_fn = PCTLoss(old_output_clean=old_outputs, alpha=alpha, beta=beta)
        mixmse = False
        new_model = None
    elif loss_name == 'MixMSE':
        loss_fn = MixedPCTLoss(old_output=old_outputs, new_output=new_outputs,
                                alpha=alpha, beta=beta,
                                only_nf=False)
        if adv_training:
            new_model = deepcopy(model).to(device)
            new_model.eval()

        mixmse = True
    elif loss_name == 'MixMSE(NF)':
        loss_fn = MixedPCTLoss(old_output=old_outputs, new_output=new_outputs,
                                alpha=alpha, beta=beta,
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

    # # Compute starting performances in validation set
    # results = get_pct_results(new_model=model, ds_loader=val_loader, 
    #                         old_model=old_model, device=device)
    # # Compute acc of old model
    # # initialize the best acc and best nfr of the current model as the starting ones
    # old_acc, best_acc, best_nfr = results['old_acc'], results['new_acc'], results['nfr']

    set_all_seed(random_seed)
    # Start the training loop...
    for e in range(epochs):
        if not adv_training:
            pc_train_epoch(model=model, device=device, train_loader=train_loader, 
                    optimizer=optimizer, epoch=e, loss_fn=loss_fn, logger=logger)
            
        else:
            adv_pc_train_epoch(model=model, old_model=old_model, device=device, train_loader=train_loader, 
                    optimizer=optimizer, epoch=e, loss_fn=loss_fn, new_model=new_model, mixmse=mixmse,
                    eps=0.03, n_steps=50, logger=logger)
            
            
            
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
        model_data = {
            'epoch': e,
            'model_state_dict': model.cpu().state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_fn
            }

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
    torch.save(model_data, os.path.join(checkpoints_dir, f"last.pt"))
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

    # random_seed=args.random_seed

    trainable_layers = None

    exp_name = f"{args.exp_name}"    
    date = datetime.now().strftime("day-%d-%m-%Y_hr-%H-%M-%S")
    exp_path = os.path.join(args.root, exp_name if not args.date else f"{date}_{exp_name}") 

    # exp_name = "prova_DEBUG_ADV"
    # exp_path = os.path.join(root, exp_name)

    if not os.path.isdir(exp_path):
        os.mkdir(exp_path)

    save_params(locals().items(), exp_path, 'info')

    logger = init_logger(exp_path, fname=f'progress_{args.exp_name}')

    # writer = SummaryWriter('logs/')

    #####################################
    # PREPARE DATA
    #####################################
    train_dataset, val_dataset = split_train_valid(
        get_cifar10_dataset(train=True, shuffle=False, num_samples=args.n_tr), train_size=0.8)
    test_dataset = get_cifar10_dataset(train=False, shuffle=False, num_samples=args.n_ts)
    # shuffle can be set to True if reference models are evaluated on the fly
    # without exploiting precomputed outputs
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    #############################
    # MODEL PAIR LEVEL
    #############################
    for mid_i, (old_model_id, model_id) in enumerate(zip(args.old_model_ids, args.model_ids)):
          
        logger.info(f"------- MODELS: {old_model_id} -> {model_id} -------") 
        model_pair_dir = f"old-{old_model_id}_new-{model_id}"
        model_pair_path = os.path.join(exp_path, model_pair_dir)

        if not os.path.isdir(model_pair_path):
            os.mkdir(model_pair_path)

        # The architecture of the old model will be the same, I load it here
        old_model = load_model(MODEL_NAMES[old_model_id], dataset='cifar10', threat_model='Linf') 
        
        model = load_model(MODEL_NAMES[model_id], dataset='cifar10', threat_model='Linf')
        # Starting test set performances
        logger.debug('Get baseline results')
        
        base_results = {}
        for ds_loader, ds_name in zip([val_loader, test_loader], ['val', 'test']):
            base_results[ds_name] = get_pct_results(new_model=model, ds_loader=ds_loader, 
                                            old_model=old_model,
                                            device=device)

            logger.info(print_perf(f"\n>>> Starting {ds_name} perf \n",
                base_results[ds_name]['old_acc'], base_results[ds_name]['new_acc'], 
                base_results[ds_name]['nfr'], base_results[ds_name]['pfr']))     

        #############################
        # LOSS TYPE LEVEL
        #############################
        for loss_name in args.loss_names:
            logger.info(f"------- LOSS: {loss_name} --------")
            loss_dir_path = os.path.join(model_pair_path, loss_name)      
            # NB: non mi serve fare un check per creare la cartella perchè lo faccio un livello dopo            
            adv_at_sel = 'AT' in loss_name
            loss_name = loss_name.split('-AT')[0]
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
                        model = load_model(MODEL_NAMES[model_id], dataset='cifar10', threat_model='Linf')
                        
                        if not args.test_only:
                            logger.debug('Start training...')
                            train_pct_model(model=model, old_model=old_model,
                                            train_loader=train_loader, val_loader=val_loader,
                                            epochs=args.epochs, loss_name=loss_name, lr=args.lr, random_seed=args.random_seed, device=device,
                                            alpha=alpha, beta=beta, trainable_layers=trainable_layers,
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
                    
                    for ds_loader, ds_name in zip([val_loader, test_loader], ['val', 'test']):                          
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

                                
                        if not clean_check:
                            #####################################
                            # SAVE CLEAN RESULTS
                            #####################################
                            logger.debug(f'Evaluating {ds_name} set ...')
                            
                            set_all_seed(args.random_seed)
                            clean_results = get_pct_results(new_model=model, ds_loader=ds_loader, 
                                                        old_correct=base_results[ds_name]['old_correct'],
                                                        device=device)
                            clean_results['loss'] = checkpoint['loss'].loss_path
                            clean_results['orig_acc'] = base_results[ds_name]['new_acc']
                            clean_results['orig_nfr'] = base_results[ds_name]['nfr']
                            clean_results['orig_pfr'] = base_results[ds_name]['pfr']
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
                                adv_dir_path = os.path.join(params_dir_path, 'advx', 'ts')

                                set_all_seed(args.random_seed)
                                generate_advx(model=model, ds_loader=ds_loader, n_steps=args.n_steps, 
                                            adv_dir_path=adv_dir_path,
                                            logger=logger, device=device,
                                            n_max_advx_samples=args.n_adv_ts)                            
                                adv_ds = MyTensorDataset(ds_path=adv_dir_path)
                                adv_ds_loader = DataLoader(adv_ds, batch_size=ds_loader.batch_size)

                                old_correct_adv = check_baseline_advx(old_model_id, ds_name, logger, args.random_seed)
                                new_correct_adv = check_baseline_advx(model_id, ds_name, logger, args.random_seed)
                                # todo: dopo aver pigliato i correct veri per test o ts
                                # uso le predizioni già salvate e le confronto con la baseline (da calcolare a parte con generate_baseline_advx)
                                
                                # Get results of model M wrt M0 and M1
                                set_all_seed(args.random_seed)
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
                                plot_loss(results['clean']['loss'], ax)
                                fig.savefig(os.path.join(params_dir_path, "loss_path.pdf"))
                            
                            logger.info(f">>> Clean Results")
                            logger.info(f"Old Acc: {results['clean']['old_acc']}")
                            logger.info(f"New Acc: {results['clean']['orig_acc']}, New Acc(FT): {results['clean']['new_acc']}")
                            logger.info(f"New NFR: {results['clean']['orig_nfr']}, New NFR(FT): {results['clean']['nfr']}")
                            logger.info(f">>> Advx Results")
                            logger.info(f"Old Acc: {results['advx']['old_acc']}")
                            logger.info(f"New Acc: {results['advx']['orig_acc']}, New Acc(FT): {results['advx']['new_acc']}")
                            logger.info(f"New NFR: {results['advx']['orig_nfr']}, New NFR(FT): {results['advx']['nfr']}")
                        
                            with open(os.path.join(params_dir_path, f"{ds_name}_perf.txt"), 'w') as f:
                                f.write(f">>> Clean Results\n")
                                f.write(f"Old Acc: {results['clean']['old_acc']}\n")
                                f.write(f"New Acc: {results['clean']['orig_acc']}, New Acc(FT): {results['clean']['new_acc']}\n")
                                f.write(f"New NFR: {results['clean']['orig_nfr']}, New NFR(FT): {results['clean']['nfr']}\n")
                                f.write(f">>> Advx Results\n")
                                f.write(f"Old Acc: {results['advx']['old_acc']}\n")
                                f.write(f"New Acc: {results['advx']['orig_acc']}, New Acc(FT): {results['advx']['new_acc']}\n")
                                f.write(f"New NFR: {results['advx']['orig_nfr']}, New NFR(FT): {results['advx']['nfr']}\n")
                        else:
                            logger.debug(f"Clean check: {clean_check}, Advx check: {advx_check}, Results cannot be saved jointly.")
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
    

    parser.add_argument('-exp_name', default='DEBUG_VANILLA', type=str)
    parser.add_argument('-root', default='results', type=str)
    parser.add_argument('-adv_tr', action='store_true')

    parser.add_argument('-n_tr', default=200, type=int, help='number of training samples')
    parser.add_argument('-n_ts', default=200, type=int, help='number of test samples')

    parser.add_argument('-epochs', default=5, type=int)
    parser.add_argument('-lr', default=1e-2, type=float)
    parser.add_argument('-batch_size', default=100, type=int)
    
    parser.add_argument('-n_steps', default=10, type=int, help='number of attack steps during robusntess evaluation')
    parser.add_argument('-n_adv_ts', default=200, type=int, help='number of advx used for robustness evaluation')
    
    parser.add_argument('-old_model_ids', default=[1], type=int, nargs='+')
    parser.add_argument('-model_ids', default=[4], type=int, nargs='+')
    parser.add_argument('-loss_names', default=['MixMSE-AT'], type=str, nargs='+')
    parser.add_argument('-alphas_mix', default=[1,1,1,5,10,5], type=float, nargs='+')
    parser.add_argument('-betas_mix', default=[1,5,10,5,1,1], type=float, nargs='+')
    parser.add_argument('-alphas_pct', default=[1, 1], type=int, nargs='+')
    parser.add_argument('-betas_pct', default=[1, 5], type=int, nargs='+')
    
    parser.add_argument('-date', action='store_true')
    parser.add_argument('-test_only', action='store_true')
    
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