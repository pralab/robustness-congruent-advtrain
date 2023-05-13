import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from adv_lib.attacks import fmn
from generate_advx import generate_advx
from utils.utils import init_logger, MODEL_NAMES, join, set_all_seed
from utils.data import get_cifar10_dataset
from robustbench.utils import load_model
import os
import pickle
import numpy as np
import argparse

ROOT = 'results/distances_results'

def compute_distances(ds_loader, model, steps=50,
                      d_path=join(ROOT, 'distance.gz'), n_max_advx_samples=2000, 
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
    
    if logger is not None:
        print = logger.debug
    
    # generate_advx(model=model, ds_loader=ds_loader, n_steps=50, attack='fmn',
    #         adv_dir_path=exp_path,
    #         logger=logger, device=device,
    #         n_max_advx_samples=n_max_advx_samples) 
    
    distances = torch.tensor([]).to(device)
    
    set_all_seed(random_seed)
    model.to(device)
    for batch_i, (x,y) in enumerate(ds_loader):
        x, y = x.to(device), y.to(device)
        advx = fmn(model, x, y, norm=float('inf'), steps=steps)
        distances_i = (x-advx).flatten(1).norm(p=2, dim=1)
        
        distances = torch.cat((distances, distances_i), 0)
    
    distances = distances[:n_max_advx_samples]
    
    with open(d_path, 'wb') as f:
        pickle.dump(distances, f)
    
    return distances
    


def compute_distances_pipeline(model_id_list, exp_path, ts_loader, logger, device, random_seed=0):
    ###########################
    # COMPUTE DISTANCES
    ###########################
    nope_list = []
    for model_id in model_id_list:
        model_name = MODEL_NAMES[model_id]
        logger.debug(f"Running distances on {model_id} -> {model_name}")
        d_path = join(exp_path, f"{model_name}.gz")
        model = load_model(model_name, dataset='cifar10', threat_model='Linf')
        try:
            set_all_seed(random_seed)
            compute_distances(ds_loader=ts_loader, model=model, logger=logger, device=device,
                            d_path=d_path, random_seed=random_seed)
        except Exception as e:
            logger.debug(f"{model_name} not computed. RIP.")
            nope_list.append(model_id)
            logger.debug(e)
    
    data = retrieve_distances_data(model_id_list=model_id_list, exp_path=exp_path)
    
    return data


def retrieve_distances_data(model_id_list, exp_path, logger=None):
    clean_path = 'results/clean'
    adv_path = 'results/advx'
    
    if logger is not None:
        print = logger.debug
    
    clean_preds_matrix = []
    adv_preds_matrix = []    
    distances_matrix = []
    model_ids_ok = []
    model_names_ok = []
    for model_id in model_id_list:
        model_name = MODEL_NAMES[model_id]
        d_path = join(exp_path, f"{model_name}.gz")
        
        try:
            with open(d_path, 'rb') as f:
                distances = pickle.load(f)
            distances_matrix.append(distances.tolist())
        
            with open(join(clean_path, model_name, 'correct_preds.gz'), 'rb') as f:
                clean_preds = pickle.load(f)
            clean_preds_matrix.append(clean_preds.tolist())
            
            with open(join(adv_path, model_name, 'correct_preds_test.gz'), 'rb') as f:
                adv_preds = pickle.load(f)
            adv_preds_matrix.append(adv_preds.tolist())
            
            model_ids_ok.append(model_id)
            model_names_ok.append(model_name)
        except Exception as e:
            print(f"Something gone wrong with {model_id} -> {model_name}")
            print(e)

    
    distances_matrix = np.array(distances_matrix)
    clean_preds_matrix = np.array(clean_preds_matrix)
    adv_preds_matrix = np.array(adv_preds_matrix)

    n_samples = min(distances_matrix.shape[1], clean_preds_matrix.shape[1], adv_preds_matrix.shape[1])
    distances_matrix = distances_matrix[:, :n_samples]
    clean_preds_matrix = clean_preds_matrix[:, :n_samples]
    adv_preds_matrix = adv_preds_matrix[:, :n_samples]
    
    data = {'model_ids': model_ids_ok,
            'model_names': model_names_ok,
            'distances': distances_matrix,
            'clean': clean_preds_matrix,
            'adv': adv_preds_matrix}
    
    return data

def main(args):
    logger = init_logger(ROOT, fname=f'progress')    
    
    logger.debug(args)
    
    model_id_list = [i+1 for i in range(7)]
    batch_size = args.batch_size    #500
    n_samples = args.n_samples  #500
    
    random_seed = args.random_seed
    
    device = torch.device(f"cuda:0" if torch.cuda.is_available()
                else "cpu")
    
    exp_path = join(ROOT, 'base_distances')
    
    ts = get_cifar10_dataset(train=False, shuffle=False, num_samples=n_samples)
    ts_loader = DataLoader(ts, batch_size=batch_size, shuffle=False)
    
    if not os.path.isdir(exp_path):
        os.makedirs(exp_path)
    


    set_all_seed(random_seed)
    # data = compute_distances_pipeline(model_id_list=model_id_list, 
    #                         exp_path=exp_path, ts_loader=ts_loader,
    #                         logger=logger, device=device,
    #                         random_seed=random_seed)
    
    data = retrieve_distances_data(model_id_list=model_id_list, exp_path=exp_path)
    
    with open(join(exp_path, 'base_distances.gz'), 'wb') as f:
        pickle.dump(data, f)
    
    with open(join(exp_path, 'base_distances.gz'), 'rb') as f:
        datax = pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-n_samples', default=50, type=int)
    parser.add_argument('-batch_size', default=50, type=int)   
    parser.add_argument('-random_seed', default=0, type=int)
    args = parser.parse_args()
    
    main(args)

