import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from adv_lib.attacks import fmn
from generate_advx import generate_advx
from utils.utils import init_logger, MODEL_NAMES, join, set_all_seed
from utils.data import get_cifar10_dataset, MyTensorDataset
from robustbench.utils import load_model
import os
import pickle
import numpy as np
import argparse

ROOT = 'results/distances_results'

def compute_distances(d_path, ds_loader, model, steps=50, advx_dir=None,
                      n_max_advx_samples=2000, 
                      logger=None, device=None, random_seed=0):
    """
    save in each specific folder a vector of 2k samples containing:
    1. the distances to the boundary wrt old model
    2. the distances to the boundary wrt new model

    use FMN to compute the distance
    
    return torch array or numpy, or better, save it
    """
    if device is None:
        device = torch.device(f"cuda:0" if torch.cuda.is_available()
                else "cpu")
    
    # if logger is not None:
    #     print = logger.debug
    
    # generate_advx(model=model, ds_loader=ds_loader, n_steps=50, attack='fmn',
    #         adv_dir_path=exp_path,
    #         logger=logger, device=device,
    #         n_max_advx_samples=n_max_advx_samples) 
    

    
    set_all_seed(random_seed)
    
    if not os.path.exists(advx_dir):
        generate_advx(ds_loader=ds_loader, model=model, adv_dir_path=advx_dir, 
                    logger=logger, device=device, attack='fmn', eps=8/255,
                    n_steps=steps, n_max_advx_samples=n_max_advx_samples)
    
    if not (os.path.exists(join(d_path, 'distances.gz')) and os.path.exists(join(d_path, 'logits.gz'))):
    # if True:
        adv_ds = MyTensorDataset(ds_path=advx_dir)
        adv_ds_loader = DataLoader(adv_ds, batch_size=ds_loader.batch_size)

        model.to(device)
        
        distances = torch.tensor([])
        logits = torch.tensor([])
        for batch_i, ((x,y), (xadv, yadv)) in enumerate(zip(ds_loader, adv_ds_loader)):
            x, y = x.to(device), y.to(device)
            xadv = xadv.to(device)
            # distances in input space
            distances_i = (x-xadv).flatten(1).norm(p=float('inf'), dim=1)
            distances = torch.cat((distances, distances_i.detach().cpu()), 0)
            
            # save logits
            logit_i = model(x)
            logits = torch.cat((logits, logit_i.detach().cpu()), 0)
            
        distances = distances[:n_max_advx_samples]
        logits = logits[:n_max_advx_samples]
        
        with open(join(d_path, 'distances.gz'), 'wb') as f:
            pickle.dump(distances.detach().cpu(), f)
        
        with open(join(d_path, 'logits.gz'), 'wb') as f:
            pickle.dump(logits.detach().cpu(), f)
    else:
        with open(join(d_path, 'distances.gz'), 'rb') as f:
            distances = pickle.load(f)
        
        with open(join(d_path, 'logits.gz'), 'rb') as f:
            logits = pickle.load(f)
    
    return distances, logits
    


def compute_distances_pipeline(model_id_list, exp_path, ts_loader, logger, 
                               device, random_seed=0, steps=50):
    ###########################
    # COMPUTE DISTANCES
    ###########################
    nope_list = []
    for model_id in model_id_list:
        model_name = MODEL_NAMES[model_id]
        logger.debug(f"Running distances on {model_id} -> {model_name}")
        d_path = join(exp_path, model_name)
        model = load_model(model_name, dataset='cifar10', threat_model='Linf')
        try:
            set_all_seed(random_seed)
            compute_distances(ds_loader=ts_loader, model=model, logger=logger, device=device,
                            d_path=d_path, advx_dir=join(d_path, 'advx'), random_seed=random_seed,
                            steps=steps)
        except Exception as e:
            logger.debug(f"{model_name} not computed. RIP.")
            nope_list.append(model_id)
            logger.debug(e)
    
    data = retrieve_distances_data(model_id_list=model_id_list, exp_path=exp_path, logger=logger)
    
    return data


def retrieve_distances_data(model_id_list, exp_path, logger):
    clean_path = 'results/clean'
    adv_path = 'results/advx'

    
    clean_preds_matrix = []
    adv_preds_matrix = []    
    distances_matrix = []
    logits_matrix = []
    model_ids_ok = []
    model_names_ok = []
    for model_id in model_id_list:
        model_name = MODEL_NAMES[model_id]
        model_path = join(exp_path, model_name)
        distances_path = join(model_path, 'distances.gz')
        logits_path = join(model_path, 'logits.gz')
        
        try:
            with open(distances_path, 'rb') as f:
                distances = pickle.load(f)
            distances_matrix.append(distances.tolist())
            
            with open(logits_path, 'rb') as f:
                logits = pickle.load(f)
            logits_matrix.append(logits.tolist())
        
            with open(join(clean_path, model_name, 'correct_preds.gz'), 'rb') as f:
                clean_preds = pickle.load(f)
            clean_preds_matrix.append(clean_preds.tolist())
            
            with open(join(adv_path, model_name, 'correct_preds_test.gz'), 'rb') as f:
                adv_preds = pickle.load(f)
            adv_preds_matrix.append(adv_preds.tolist())
            
            model_ids_ok.append(model_id)
            model_names_ok.append(model_name)
        except Exception as e:
            logger.debug(f"Something gone wrong with {model_id} -> {model_name}")
            logger.debug(e)

    
    distances_matrix = np.array(distances_matrix)
    logits_matrix = np.array(logits_matrix)
    clean_preds_matrix = np.array(clean_preds_matrix)
    adv_preds_matrix = np.array(adv_preds_matrix)

    n_samples = min(distances_matrix.shape[1], clean_preds_matrix.shape[1], adv_preds_matrix.shape[1])
    distances_matrix = distances_matrix[:, :n_samples]
    logits_matrix = logits_matrix[:, :n_samples, :]
    clean_preds_matrix = clean_preds_matrix[:, :n_samples]
    adv_preds_matrix = adv_preds_matrix[:, :n_samples]
    
    data = {'model_ids': model_ids_ok,
            'model_names': model_names_ok,
            'distances': distances_matrix,
            'logits': logits_matrix,
            'clean': clean_preds_matrix,
            'adv': adv_preds_matrix}
    
    return data

def main(args):
    model_id_list = [i+1 for i in range(7)]
    batch_size = args.batch_size    #500
    n_samples = args.n_samples  #500
    steps = args.steps  #50
    
    random_seed = args.random_seed
    
    device = torch.device(f"cuda:0" if torch.cuda.is_available()
                else "cpu")
    
    # exp_path = join(ROOT, f"base_distances_{n_samples}samples_{steps}steps")
    # if not os.path.isdir(exp_path):
    #     os.makedirs(exp_path)
    exp_path = 'results/distances_results/base_distances_2000samples_50steps'
    
    logger = init_logger(exp_path, fname=f'progress')    
    
    logger.debug(args)
    
    ts = get_cifar10_dataset(train=False, shuffle=False, num_samples=n_samples)
    ts_loader = DataLoader(ts, batch_size=batch_size, shuffle=False)
    

    set_all_seed(random_seed)
    data = compute_distances_pipeline(model_id_list=model_id_list, 
                            exp_path=exp_path, ts_loader=ts_loader,
                            logger=logger, device=device,
                            random_seed=random_seed,
                            steps=steps)
    
    data = retrieve_distances_data(model_id_list=model_id_list, exp_path=exp_path, logger=logger)
    
    with open(join(exp_path, 'base_distances.gz'), 'wb') as f:
        pickle.dump(data, f)
    
    with open(join(exp_path, 'base_distances.gz'), 'rb') as f:
        datax = pickle.load(f)
    
    print("")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('-n_samples', default=30, type=int)
    # parser.add_argument('-batch_size', default=10, type=int) 
    # parser.add_argument('-steps', default=2, type=int)   
    # parser.add_argument('-random_seed', default=0, type=int)
    # args = parser.parse_args()
    
    parser.add_argument('-n_samples', default=2000, type=int)
    parser.add_argument('-batch_size', default=50, type=int) 
    parser.add_argument('-steps', default=50, type=int)   
    parser.add_argument('-random_seed', default=0, type=int)
    args = parser.parse_args()
    
    main(args)

