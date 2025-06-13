from robustbench.utils import load_model
# from secml.utils import fm
import pickle
import utils.utils as ut
from utils.eval import correct_predictions, get_pct_results, compute_nflips, compute_pflips
import utils.utils as ut
import math
import torch
import os
from tqdm import tqdm
import numpy as np

from utils.data import get_cifar10_dataset, get_imagenet_dataset, \
    MyTensorDataset, split_train_valid
from torch.utils.data import DataLoader
from utils.visualization import imshow
from utils.eval import evaluate_acc
import argparse
import json
import torchvision

# import foolbox as fb
from adv_lib.attacks.auto_pgd import apgd
from adv_lib.attacks import fmn

import time

def generate_advx_ds(model, ds_loader, ds_path, logger=None, device=None,
                    eps=8/255, n_steps=250, n_max_advx_samples=2000, attack='apgd',
                    random_seed=0):
    """
    crea una cartella dove per ogni sample salva il singolo tensore in ds_path.
    i sample sono rinominati in ordine crescente come vengono incontrati nel ds originale
    """
    
    assert attack in ('apgd', 'fmn')
    
    model.to(device)
    model.eval()

    if not os.path.isdir(ds_path):
        os.makedirs(ds_path)

    k = 0   # index for samples
    fname_to_target = {}
    ut.set_all_seed(random_seed)
    with tqdm(total=len(ds_loader)) as t:
        for batch_idx, (x,y) in enumerate(ds_loader):
            x, y = x.to(device), y.to(device)
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
                fname = f"{str(k).zfill(10)}.png"
                fname_to_target[fname] = y[i].item()

                file_path = os.path.join(ds_path, fname)
                torchvision.utils.save_image(advx[i], file_path)
                # file_path = os.path.join(ds_path, f"{str(k).zfill(10)}.gz")
                # advx = advx.detach().cpu()
                # y = y.detach().cpu()
                # data = (advx[i], y[i])
                # with open(file_path, 'wb') as f:
                #     pickle.dump(data, f)
                k += 1
                if k>=n_max_advx_samples:
                    with open(os.path.join(ds_path, 'fname_to_target.json'), 'w') as f:
                        json.dump(fname_to_target, f)
                    return
    with open(os.path.join(ds_path, 'fname_to_target.json'), 'w') as f:
        json.dump(fname_to_target, f)         
    return 




def generate_advx(ds_loader, model, adv_dir_path, logger, device, attack='apgd',
                  eps=8/255, n_steps=50, model_name=None, n_max_advx_samples=2000):
    """
    ft_models e tr_set specificano solo una cartella diversa rispetto agli advx 
    per i modelli originali sul test set

    cartella advx_folder con dentro gli advx WB per ogni modello selezionato
    """
    logger.debug(f"Using EPS = {eps}")
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

def check_baseline(mid, ds_name, logger, random_seed, ds_id=ut.cifar10_id, 
                   cuda_id=1, batch_size=500, sel_advx=False, n_max_advx=None, eps=None):
    """
    If baseline advx already exists for model<mid> load it in correct_adv
    otherwise compute, save and return
    """
    type = 'clean' if not sel_advx else 'advx'
    root = f'results/{type}' if ds_id==ut.cifar10_id else f'results/{type}-imagenet'
    correct_fname = os.path.join(root, ut.MODEL_NAMES[ds_id][mid], f"correct_preds_{ds_name}.gz")
    try:
        # Load WB advx predictions of Mold
        with open(correct_fname, 'rb') as f:
            correct = pickle.load(f)
    except:
        logger.debug(f"Baseline {ds_name} {type} for M{mid} does not exist. Generating...")
        ut.set_all_seed(random_seed)
        generate_baseline(mid, ds_name=ds_name, cuda_id=cuda_id, 
                          batch_size=batch_size, sel_advx=sel_advx, 
                          n_max_advx=n_max_advx, eps=eps, ds_id=ds_id)
        with open(correct_fname, 'rb') as f:
            correct = pickle.load(f)
    return correct


def generate_baseline(model_id, ds_name='test',
                           cuda_id=1, batch_size=500, sel_advx=False, n_max_advx=None, eps=None, ds_id=ut.cifar10_id):
    type = 'clean' if not sel_advx else 'advx'
    root = f'results/{type}' if ds_id==ut.cifar10_id else f'results/{type}-imagenet'
    os.makedirs(root, exist_ok=True)

    device = f"cuda:{cuda_id}" if torch.cuda.is_available() else 'cpu'
    logger = ut.init_logger(root, fname=f"logger_{ds_name}")
    
    if ds_id == ut.cifar10_id:
        _, val_dataset = split_train_valid(
            get_cifar10_dataset(train=True, shuffle=False, num_samples=None), train_size=0.8)
        test_dataset = get_cifar10_dataset(train=False, shuffle=False, num_samples=None)
    else:
        _, val_dataset, test_dataset = get_imagenet_dataset(normalize=False)
    # shuffle can be set to True if reference models are evaluated on the fly
    # without exploiting precomputed outputs
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # for model_id in [2,3,1]:
    logger.info(f">>> Model {model_id}")
    model_name = ut.MODEL_NAMES[ds_id][model_id]        
    # model = load_model(model_name=model_name, dataset='cifar10', threat_model='Linf')
    model = load_model(model_name=model_name, dataset=ds_id, threat_model='Linf')
    dir_path = os.path.join(root, model_name, ds_name)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    corrects_path = os.path.join(root, model_name, f"correct_preds_{ds_name}.gz")
    
    if eps is None:
        eps = ut.EPS[ds_id]
    if n_max_advx is None:
        n_max_advx = ut.N_MAX_ADVX[ds_id]
    ds_loader = test_loader if ds_name=='test' else val_loader

    if sel_advx:
        generate_advx(ds_loader=ds_loader, model=model, 
                        adv_dir_path=dir_path, eps=eps,
                        model_name=model_name, device=device, logger=logger,
                        n_max_advx_samples=n_max_advx)
        adv_ds = MyTensorDataset(ds_path=dir_path)
        ds_loader = DataLoader(adv_ds, batch_size=ds_loader.batch_size)
    
    correct_preds = correct_predictions(model=model, test_loader=ds_loader, device=device)
    
    with open(corrects_path, 'wb') as f:
        pickle.dump(correct_preds.cpu(), f)
    
    acc = correct_preds.cpu().numpy().mean() * 100
    type_of_acc = "Robust" if sel_advx else "Clean"
    logger.info(f"{type_of_acc} accuracy: {acc}")


def generate_advx_main(root, logger=None):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-cuda', default=0, type=int, choices=[0, 1])
    parser.add_argument('-exp_id', default=0, type=int, choices=[0, 1])
    args = parser.parse_args()

    batch_size = 500
    n_steps = 50
    num_samples = 2000

    ut.set_all_seed(0)
    
    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else 'cpu'
    split_cuda = False


    advx_folder = os.path.join(root, 
                        ut.custom_dirname(ut.ADVX_DIRNAME_DEFAULT, ft_models=True, tr_set=False))
    if not os.path.isdir(advx_folder):
        os.mkdir(advx_folder)

    pfname = 'progress_advx_mix_add'
    if logger is None:
        logger = ut.init_logger(advx_folder, fname=pfname + f"_{device}" if split_cuda else pfname)
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

    logger.info('>>> PCT Robustness')
    for i, model_info in enumerate(models_info_list):
        logger.info(f"Exp {i+1} / {len(models_info_list)} ->")

        # if not os.path.exists(os.path.join(model_info['advx_path'], f"results_{model_info['tr_model_sel']}.gz")):                
        model_name = ut.MODEL_NAMES[model_info['robustbench_idx']]
        
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
        old_model_name = ut.MODEL_NAMES[model_info['robustbench_old_idx']]
        with open(os.path.join('results', 'advx', old_model_name, 'correct_preds.gz'), 'rb') as f:
            old_correct = pickle.load(f)
        new_model_name = ut.MODEL_NAMES[model_info['robustbench_idx']]
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
    logger.info("Pipeline completed :D")