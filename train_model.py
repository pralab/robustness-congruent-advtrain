from utils.data import get_cifar10_dataset, split_train_valid
from utils.utils import MODEL_NAMES, set_all_seed, init_logger, save_params
from torch.utils.data import DataLoader
from robustbench.utils import load_model
from utils.trainer import pc_train_epoch, adv_pc_train_epoch, freeze_network
from utils.eval import get_ds_outputs, get_pct_results
from utils.custom_loss import PCTLoss, MixedPCTLoss
import torch
import os
from datetime import datetime
import pickle


def train_pct_model(model, old_model,
                    train_loader, val_loader,
                    epochs, loss_name, lr, random_seed, device, 
                    alpha, beta, exp_dir, trainable_layers=None, adv_training=False,
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
        train_outputs['old'] = get_ds_outputs(old_model, train_loader, device).cpu()   #needed for PCT finetuning
        train_outputs['new'] = get_ds_outputs(model, train_loader, device).cpu() 
        
        with open(os.path.join(exp_dir, 'baseline.gz'), 'wb') as f:
            pickle.dump(train_outputs, f)

    old_outputs, new_outputs = train_outputs['old'].to(device), train_outputs['new'].to(device)

    set_all_seed(random_seed)
    
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
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    if trainable_layers is not None:
        freeze_network(model, n_layer=trainable_layers)

    checkpoints_dir = os.path.join(exp_dir, 'checkpoints')
    if not os.path.isdir(checkpoints_dir):
        os.mkdir(checkpoints_dir)
    
    results = get_pct_results(new_model=model, ds_loader=val_loader, 
                            old_model=old_model, device=device)

    old_acc = results['old_acc']
    best_acc = results['new_acc']
    best_nfr = results['nfr']
    for e in range(epochs):
        if not adv_training:
            pc_train_epoch(model, device, train_loader, optimizer, e, loss_fn)
        else:
            adv_pc_train_epoch(model, old_model, device, train_loader, optimizer, e, loss_fn, mixmse)
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
            'model_state_dict': model.cpu().state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_fn,
            'perf': {'acc': acc, 'nfr': nfr, 'pfr': pfr}
            }

        if acc > best_acc:
            torch.save(model_data, os.path.join(checkpoints_dir, f"best_acc.pt"))
        
        # il secondo causa errore di scrittura file!!!
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
    device = torch.device("cuda:1" if torch.cuda.is_available()
                else "cpu")

    random_seed=0
    old_model_ids=[0,1,2,3,4,5,6,7,8]
    # old_model_ids=[3]
    # specify number of last layers to train, the others will be freezed, train all if None
    trainable_layers = None 
    adv_training = True
    n_tr = None
    n_ts = None
    epochs=12
    batch_size=500
    lr=1e-3
    betas = [1, 2, 5, 10, 100]
    alphas = [1, 1, 1, 1, 1]
    betas = [1]
    alphas = [1]
    tr_model_sel = 'last'   # last, best_acc, best_nfr
    exp_name = f"epochs-{epochs}_batchsize-{batch_size}_AT"
    # exp_name = f"PROVADEBUG"


    root = 'results'
    date = datetime.now().strftime("day-%d-%m-%Y_hr-%H-%M-%S")
    exp_path = os.path.join(root, f"{date}_{exp_name}")
    # exp_path = os.path.join(root, exp_name)

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
    # shuffle può essere messo a True se si valuta il vecchio modello 
    # on the fly senza usare output precalcolati
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for old_model_id in old_model_ids:       

        model_id = old_model_id + 1
        model_pair_dir = f"old-{old_model_id}_new-{model_id}"
        model_pair_path = os.path.join(exp_path, model_pair_dir)
        if not os.path.isdir(model_pair_path):
            os.mkdir(model_pair_path)

        logger.info(f"------- MODELS {old_model_id}- -------")
        try:
            #####################################
            # GET MODELS
            #####################################
            old_model = load_model(MODEL_NAMES[old_model_id], dataset='cifar10', threat_model='Linf')
            model = load_model(MODEL_NAMES[model_id], dataset='cifar10', threat_model='Linf')

            logger.debug('Get baseline results')
            base_results = get_pct_results(new_model=model, ds_loader=test_loader, 
                                            old_model=old_model,
                                            device=device)
            old_correct = base_results['old_correct']

            logger.info(print_perf("\n>>> Starting test perf \n",
                base_results['old_acc'], base_results['new_acc'], 
                base_results['nfr'], base_results['pfr']))

            for i, loss_name in enumerate(['PCT', 'MixMSE', 'MixMSE(NF)']):
                logger.info(f"------- LOSS: {loss_name} --------")
                loss_dir_path = os.path.join(model_pair_path, loss_name)

                for alpha, beta in list(zip(alphas, betas)):
                    logger.info(f">>> Alpha {alpha}, Beta: {beta}")
                    params_dir_path = os.path.join(loss_dir_path, f"a-{alpha}_b-{beta}")
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
