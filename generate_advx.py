from robustbench.utils import load_model
from secml.utils import fm
import pickle
from utils.utils import MODEL_NAMES, advx_fname, custom_dirname, \
    ADVX_DIRNAME_DEFAULT, FINETUNING_DIRNAME_DEFAULT, \
    set_all_seed, init_logger, used_memory_percentage
from utils.eval import correct_predictions, get_pct_results, compute_nflips, compute_pflips
import math
import torch
import os
from tqdm import tqdm
import numpy as np

from utils.data import get_cifar10_dataset, MyTensorDataset, split_train_valid
from torch.utils.data import DataLoader
from utils.visualization import imshow
from utils.eval import evaluate_acc
import argparse

from manage_files import delete_advx_ts

# import foolbox as fb
from adv_lib.attacks.auto_pgd import apgd
from adv_lib.attacks import fmn

import time

def generate_advx_ds(model, ds_loader, ds_path, logger=None, device=None,
                    eps=0.03, n_steps=250, n_max_advx_samples=2000, attack='apgd'):
    """
    crea una cartella dove per ogni sample salva il singolo tensore in ds_path.
    i sample sono rinominati in ordine crescente come vengono incontrati nel ds originale
    """
    
    assert attack in ('apgd', 'fmn')
    
    model.to(device)
    model.eval()

    if not os.path.isdir(ds_path):
        os.makedirs(ds_path)
    
    k=0 # index for samples
    with tqdm(total=len(ds_loader)) as t:
        for batch_idx, (x,y) in enumerate(ds_loader):
            x, y = x.to(device), y.to(device)
            # x.requires_grad = True
            if attack=='apgd':
                advx = apgd(model, x, y, eps=eps, norm=float('inf'), n_iter=n_steps)
            if attack=='fmn':
                advx = fmn(model, x, y, norm=float('inf'), steps=n_steps)
                
            t.set_postfix(
                compl='[{}/{} ({:.0f}%)]'.format(
                    batch_idx * len(x),
                    len(ds_loader.dataset),
                    100. * batch_idx / len(ds_loader)))
            t.update()

            for i in range(len(advx)):
                if k>=n_max_advx_samples:
                    return
                file_path = os.path.join(ds_path, f"{str(k).zfill(10)}.gz")
                advx = advx.detach().cpu()
                y = y.detach().cpu()
                data = (advx[i], y[i])
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
                k += 1
                
    return 




def generate_advx(ds_loader, model, adv_dir_path, logger, device, attack='apgd',
                  eps=0.03, n_steps=50, model_name=None, n_max_advx_samples=2000):
    """
    ft_models e tr_set specificano solo una cartella diversa rispetto agli advx 
    per i modelli originali sul test set

    cartella advx_folder con dentro gli advx WB per ogni modello selezionato
    """

    ds = ds_loader.dataset
    batch_size_temp = ds_loader.batch_size
    escape_flag = 5
    while(True):
        try:
            ds_loader_temp = DataLoader(ds, batch_size=batch_size_temp)
            generate_advx_ds(model=model, ds_loader=ds_loader_temp, 
                            ds_path=adv_dir_path, device=device, attack=attack,
                            eps=eps, n_steps=n_steps, n_max_advx_samples=n_max_advx_samples)
            logger.debug("Advx generation completed.")

        except Exception as e:
            logger.debug(f"Model {model_name} failed: {e}")
            if escape_flag == 0:
                logger.debug('Escape flag activated.')
                break
            else:
                if 'CUDA out of memory' in str(e):                        
                    batch_size_temp = int(batch_size_temp / 2)
                    logger.debug(f"Trying with batch size {batch_size_temp}")
                    escape_flag = escape_flag - 1
                    continue
                else:
                    break
        else:
            break



def get_models_info_list(root, advx_folder, nopes=None):
    old_models_idx = []
    new_models_idx = []
    models_info_list = []
    for path, dirs, files in os.walk(root):
        if ('checkpoints' in path) and (len(files)!=0):
            # if 'best_nfr.pt' in files:
            #     tr_model_sel = 'best_nfr'                
            # else:
            tr_model_sel = 'last'
            model_path = os.path.join(path, f"{tr_model_sel}.pt")
            robustbench_idx = int(path.split('_new-')[1].split('/')[0])
            robustbench_old_idx = int(path.split('old-')[1].split('_new')[0])
            hparams = path.split('/')[-2]
            loss_name = path.split('/')[-3]
            models_pair = path.split('/')[-4]
            advx_path = os.path.join(advx_folder, models_pair, loss_name, hparams)

            if nopes is not None:
                if robustbench_old_idx in nopes:
                    continue

            old_models_idx.append(robustbench_old_idx)
            new_models_idx.append(robustbench_idx)
            models_info_list.append({
                'model_path': model_path,
                'advx_path': advx_path,
                'models_pair': models_pair,
                'loss_name': loss_name,
                'hparams': hparams,
                'robustbench_idx': robustbench_idx,
                'robustbench_old_idx': robustbench_old_idx,
                'tr_model_sel': tr_model_sel
            })
    return models_info_list, old_models_idx, new_models_idx


def check_baseline_advx(mid, ds_name, logger, random_seed):
    """
    If baseline advx already exists for model<mid> load it in correct_adv
    otherwise compute, save and return
    """
    correct_adv_fname = os.path.join('results', 'advx', MODEL_NAMES[mid], f"correct_preds_{ds_name}.gz")
    try:
        # Load WB advx predictions of Mold
        with open(correct_adv_fname, 'rb') as f:
            correct_adv = pickle.load(f)
    except:
        logger.debug(f"Baseline {ds_name} advx for M{mid} does not exist. Generating...")
        set_all_seed(random_seed)
        generate_baseline_advx(mid, ds_name=ds_name)
        with open(correct_adv_fname, 'rb') as f:
            correct_adv = pickle.load(f)
    return correct_adv


def generate_baseline_advx(model_id, ds_name='test'):
    root = 'results/advx'

    device = f"cuda:0" if torch.cuda.is_available() else 'cpu'
    logger = init_logger(root, fname=f"logger_{ds_name}")
    
    batch_size = 500
    train_dataset, val_dataset = split_train_valid(
        get_cifar10_dataset(train=True, shuffle=False, num_samples=None), train_size=0.8)
    test_dataset = get_cifar10_dataset(train=False, shuffle=False, num_samples=None)
    # shuffle can be set to True if reference models are evaluated on the fly
    # without exploiting precomputed outputs
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # for model_id in [2,3,1]:
    logger.info(f">>> Model {model_id}")
    model_name = MODEL_NAMES[model_id]        
    model = load_model(model_name=model_name, dataset='cifar10', threat_model='Linf')
    adv_dir_path = os.path.join(root, model_name, ds_name)
    if not os.path.isdir(adv_dir_path):
        os.makedirs(adv_dir_path)
    corrects_path = os.path.join(root, model_name, f"correct_preds_{ds_name}.gz")
    
    ds_loader = test_loader if ds_name == 'test' else val_loader
    generate_advx(ds_loader=ds_loader, model=model, 
                    adv_dir_path=adv_dir_path,
                    model_name=model_name, device=device, logger=logger,
                    n_max_advx_samples=2000)
    adv_ds = MyTensorDataset(ds_path=adv_dir_path)
    adv_ds_loader = DataLoader(adv_ds, batch_size=ds_loader.batch_size)
    correct_preds = correct_predictions(model=model, test_loader=adv_ds_loader, device=device)
    with open(corrects_path, 'wb') as f:
        pickle.dump(correct_preds.cpu(), f)


def generate_advx_main(root, logger=None):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-cuda', default=0, type=int, choices=[0, 1])
    parser.add_argument('-exp_id', default=0, type=int, choices=[0, 1])
    args = parser.parse_args()

    batch_size = 500
    n_steps = 50
    num_samples = 2000


    set_all_seed(0)
    
    # roots = ['results/day-25-01-2023_hr-15-38-00_epochs-12_batchsize-500_CLEAN_TR',
    #         'results/day-30-01-2023_hr-10-01-02_epochs-12_batchsize-500_ADV_TR']
    # root = roots[args.exp_id]
    
    
    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else 'cpu'
    split_cuda = False


    advx_folder = os.path.join(root, 
                        custom_dirname(ADVX_DIRNAME_DEFAULT, ft_models=True, tr_set=False))
    if not os.path.isdir(advx_folder):
        os.mkdir(advx_folder)

    pfname = 'progress_advx_mix_add'
    if logger is None:
        logger = init_logger(advx_folder, fname=pfname + f"_{device}" if split_cuda else pfname)
    models_info_list, old_models_idx, new_models_idx = get_models_info_list(root, advx_folder, nopes=None)

    if split_cuda:
        half_exps = len(models_info_list) // 2
        if '1' in device:
            models_info_list = models_info_list[:half_exps]
            old_models_idx = old_models_idx[:half_exps]
            new_models_idx = new_models_idx[:half_exps]
        else:
            models_info_list = models_info_list[half_exps:]
            old_models_idx = old_models_idx[half_exps:]
            new_models_idx = new_models_idx[half_exps:]

    # logger.info('>>> Baseline Robustness')
    # models_id = np.unique(np.array(new_models_idx + old_models_idx))
    # nope_list = []
    # for model_id in models_id:
    #     model_name = MODEL_NAMES[model_id]
    #     logger.info(f"-> {model_id} - {model_name}")   
    #     model = load_model(model_name=model_name, dataset='cifar10', threat_model='Linf')
    #     base_path = os.path.join('results', 'advx', model_name)
    #     ds_path = os.path.join(base_path, 'ts')
    #     if not os.path.isdir(ds_path):
    #         os.makedirs(ds_path)
        
    #     batch_size_temp = batch_size
    #     escape_flag = 4
    #     while(True):
    #         try:
    #             ds = get_cifar10_dataset(train=False, num_samples=num_samples)
    #             ds_loader = DataLoader(ds, batch_size=batch_size_temp, shuffle=False)
    #             if len(os.listdir(ds_path)) == 0:
    #                 generate_advx_ds(model=model, ds_loader=ds_loader, 
    #                                 ds_path=ds_path, device=device, n_steps=n_steps)
    #                 logger.debug("Advx generation completed.")
    #             else:
    #                 logger.debug('Advx already existing')
                

    #             corrects_path = os.path.join(base_path, 'correct_preds.gz')
    #             # if not os.path.exists(corrects_path):
    #             adv_ds = MyTensorDataset(ds_path=ds_path)
    #             adv_ds_loader = DataLoader(adv_ds, batch_size=batch_size)
    #             correct_preds = correct_predictions(model=model, test_loader=adv_ds_loader, device=device)
    #             with open(corrects_path, 'wb') as f:
    #                 pickle.dump(correct_preds.cpu(), f)
    #             logger.debug("Preds completed.")
    #             # else:
    #             #     logger.debug('Preds already existing')


    #         except Exception as e:
    #             logger.debug(f"Model {model_id} failed: {e}")
    #             if escape_flag == 0:
    #                 logger.debug('Escape flag activated.')
    #                 break
    #             else:
    #                 if 'CUDA out of memory' in str(e):                        
    #                     batch_size_temp = int(batch_size_temp / 2)
    #                     logger.debug(f"Trying with batch size {batch_size_temp}")
    #                     escape_flag = escape_flag - 1
    #                     continue
    #                 else:
    #                     break
    #         else:
    #             break


    


    logger.info('>>> PCT Robustness')
    for i, model_info in enumerate(models_info_list):
        logger.info(f"Exp {i+1} / {len(models_info_list)} ->")

        # # todo: REMOVE THIS AFTER RUNNING MIXMSE
        # if model_info['loss_name'] == 'PCT':
        #     continue
        
        # if 'a-1_' in model_info['hparams']:
        #     continue

        # if not os.path.exists(os.path.join(model_info['advx_path'], f"results_{model_info['tr_model_sel']}.gz")):                
        model_name = MODEL_NAMES[model_info['robustbench_idx']]
        
        # Load finetuned model
        model = load_model(model_name=model_name, dataset='cifar10', threat_model='Linf')
        checkpoint = torch.load(model_info['model_path'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        ds_path = os.path.join(model_info['advx_path'], 'ts')
        if not os.path.isdir(ds_path):
            os.makedirs(ds_path)
        
        ds = get_cifar10_dataset(train=False, num_samples=num_samples)
        ds_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        generate_advx(ds_loader=ds_loader, model=model, n_steps=n_steps,
                      adv_dir_path=ds_path, logger=logger, device=device, model_name=model_info['model_path'])


        adv_ds = MyTensorDataset(ds_path=ds_path)
        adv_ds_loader = DataLoader(adv_ds, batch_size=batch_size)

        # Load WB advx predictions of M0 and M1
        old_model_name = MODEL_NAMES[model_info['robustbench_old_idx']]
        with open(os.path.join('results', 'advx', old_model_name, 'correct_preds.gz'), 'rb') as f:
            old_correct = pickle.load(f)
        new_model_name = MODEL_NAMES[model_info['robustbench_idx']]
        with open(os.path.join('results', 'advx', new_model_name, 'correct_preds.gz'), 'rb') as f:
            new_correct = pickle.load(f)
        
        # Get results of model M wrt M0 and M1
        results = get_pct_results(new_model=model, ds_loader=adv_ds_loader, 
                                    old_correct=old_correct,
                                    device=device)
        # Add baseline results for comparison
        results['orig_acc'] = new_correct.cpu().numpy().mean()
        results['orig_nfr'] = compute_nflips(old_correct, new_correct)
        results['orig_pfr'] = compute_pflips(old_correct, new_correct)

        with open(os.path.join(model_info['advx_path'], f"results_{model_info['tr_model_sel']}.gz"), 'wb') as f:
            pickle.dump(results, f)
        logger.debug("Preds completed.")
        
        # batch_size_temp = batch_size
        # escape_flag = 5
        # while(True):
        #     try:
        #         # Reload dataset with updated batch_size
        #         ds_loader = DataLoader(ds, batch_size=batch_size_temp, shuffle=False)

        #         generate_advx_ds(model=model, ds_loader=ds_loader, 
        #                         ds_path=ds_path, device=device, n_steps=n_steps)
        #         logger.debug("Advx generation completed.")


        #         adv_ds = MyTensorDataset(ds_path=ds_path)
        #         adv_ds_loader = DataLoader(adv_ds, batch_size=batch_size)

        #         # Load WB advx predictions of M0 and M1
        #         old_model_name = MODEL_NAMES[model_info['robustbench_old_idx']]
        #         with open(os.path.join('results', 'advx', old_model_name, 'correct_preds.gz'), 'rb') as f:
        #             old_correct = pickle.load(f)
        #         new_model_name = MODEL_NAMES[model_info['robustbench_idx']]
        #         with open(os.path.join('results', 'advx', new_model_name, 'correct_preds.gz'), 'rb') as f:
        #             new_correct = pickle.load(f)
                
        #         # Get results of model M wrt M0 and M1
        #         results = get_pct_results(new_model=model, ds_loader=adv_ds_loader, 
        #                                     old_correct=old_correct,
        #                                     device=device)
        #         # Add baseline results for comparison
        #         results['orig_acc'] = new_correct.cpu().numpy().mean()
        #         results['orig_nfr'] = compute_nflips(old_correct, new_correct)
        #         results['orig_pfr'] = compute_pflips(old_correct, new_correct)

        #         with open(os.path.join(model_info['advx_path'], f"results_{model_info['tr_model_sel']}.gz"), 'wb') as f:
        #             pickle.dump(results, f)
        #         logger.debug("Preds completed.")
        #         # else:
        #         #     logger.debug('Advx already existing.')

        #     except Exception as e:
        #         logger.debug(f"Model {model_name} failed: {e}")
        #         if escape_flag == 0:
        #             logger.debug('Escape flag activated.')
        #             break
        #         else:
        #             if 'CUDA out of memory' in str(e):                        
        #                 batch_size_temp = int(batch_size_temp / 2)
        #                 logger.debug(f"Trying with batch size {batch_size_temp}")
        #                 escape_flag = escape_flag - 1
        #                 continue
        #             else:
        #                 break
        #     else:
        #         break


    logger.info("Pipeline completed :D")






if __name__ == '__main__':
    # root = 'results/day-06-03-2023_hr-17-23-52_epochs-12_batchsize-500_HIGH_AB'
    # generate_advx_main(root)
    
    # generate_baseline_advx(ds_name='val')
    
    for m_id in range(8):
        path = fm.join("results/advx", MODEL_NAMES[m_id], 'correct_preds_val.gz')
        
        try:
            with open(path, 'rb') as f:
                correct = pickle.load(f)
            print(f"M{m_id}: {correct.shape[0]}")
            print(f"M{m_id}: {correct.cpu().numpy().mean()}")
        except:
            pass
    
    print("")