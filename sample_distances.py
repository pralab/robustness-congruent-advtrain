import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from adv_lib.attacks import fmn
from generate_advx import generate_advx
from utils.utils import init_logger, MODEL_NAMES, join, set_all_seed, get_chosen_ftmodels_path, model_pairs_str_to_ids
from utils.data import get_cifar10_dataset, MyTensorDataset
from robustbench.utils import load_model
import os
import pickle
import numpy as np
import argparse
from utils.eval import compute_nflips, compute_common_nflips

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
    
    if logger is not None:
        print = logger.debug
    
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
        for batch_i, ((x, _), (xadv, _)) in enumerate(zip(ds_loader, adv_ds_loader)):
            x = x.to(device)
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
        if isinstance(model_id, int):
            model_name = MODEL_NAMES[model_id]
            model = load_model(model_name, dataset='cifar10', threat_model='Linf')
        elif isinstance(model_id, str):
            model_name = model_id.split('/')[-5]
            if model_name != 'old-3_new-2':
                continue
            _, new_id = model_pairs_str_to_ids(model_name)
            model = load_model(MODEL_NAMES[new_id], dataset='cifar10', threat_model='Linf')
            checkpoint = torch.load(model_id, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise TypeError("You have to choose either integer model ids or strings")
            
        logger.debug(f"Running distances on {model_id} -> {model_name}")
        d_path = join(exp_path, model_name)
        
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

 
    distances_matrix = []
    logits_matrix = []
    clean_preds_matrix = []
    adv_preds_matrix = []
    model_ids_ok = []
    model_names_ok = []
    for model_id in model_id_list:
        if isinstance(model_id, int):
            model_name = MODEL_NAMES[model_id]
            model_path = join(exp_path, model_name)
            clean_path = join('results/clean', model_name, 'correct_preds.gz')
            adv_path = join('results/advx', model_name, 'correct_preds_test.gz')
        elif isinstance(model_id, str):
            model_name = model_id.split('/')[-5]
            model_path = join(exp_path, model_name)
            
            clean_path = join(model_id.split('/checkpoints')[0], 'results_clean_test.gz')
            adv_path = join(model_id.split('/checkpoints')[0], 'results_advx_test.gz')
        distances_path = join(model_path, 'distances.gz')
        logits_path = join(model_path, 'logits.gz')
        
        try:
            with open(distances_path, 'rb') as f:
                distances = pickle.load(f)
            distances_matrix.append(distances.tolist())
            
            with open(logits_path, 'rb') as f:
                logits = pickle.load(f)
            logits_matrix.append(logits.tolist())
        
            with open(clean_path, 'rb') as f:
                clean_preds = pickle.load(f)
            
            with open(adv_path, 'rb') as f:
                adv_preds = pickle.load(f)
                
            if isinstance(model_id, str):
                clean_preds = clean_preds['new_correct']
                adv_preds = adv_preds['new_correct']
            
            adv_preds_matrix.append(adv_preds.tolist())
            clean_preds_matrix.append(clean_preds.tolist())
            
            model_ids_ok.append(model_id)
            model_names_ok.append(model_name)
        except Exception as e:
            logger.debug(f"Something gone wrong with {model_id} -> {model_name}")
            logger.debug(e)

    
    distances_matrix = np.array(distances_matrix)
    logits_matrix = np.array(logits_matrix)
    clean_preds_matrix = np.array(clean_preds_matrix)
    adv_preds_matrix = np.array(adv_preds_matrix)

    # n_samples = min(distances_matrix.shape[1], clean_preds_matrix.shape[1], adv_preds_matrix.shape[1])
    # # n_samples = distances_matrix.shape[1]
    # distances_matrix = distances_matrix[:, :n_samples]
    # logits_matrix = logits_matrix[:, :n_samples, :]
    # clean_preds_matrix = clean_preds_matrix[:, :n_samples]
    # adv_preds_matrix = adv_preds_matrix[:, :n_samples]
    
    data = {'model_ids': model_ids_ok,
            'model_names': model_names_ok,
            'distances': distances_matrix,
            'logits': logits_matrix,
            'clean': clean_preds_matrix,
            'adv': adv_preds_matrix}
    
    return data

def main(args):
    batch_size = args.batch_size    #500
    n_samples = args.n_samples  #500
    steps = args.steps  #50
    loss_names = args.loss_name
    
    random_seed = args.random_seed
    
    device = torch.device(f"cuda:0" if torch.cuda.is_available()
                else "cpu")
    
    base_exp_path = join(ROOT, "Linf_norm", f"base_distances_{n_samples}samples_{steps}steps")

    # exp_path = 'results/distances_results/base_distances_2000samples_50steps'
    
    for loss_name in loss_names:
        fname = f"base_distances.gz"
        if isinstance(loss_name, str):
            fname = fname.replace('.gz', f"_{loss_name}_ft.gz")
            csv_path = 'results/day-30-03-2023_hr-10-01-01_PIPELINE_50k_3models/model_results_test_with_val_criteria-S-NFR.csv'
            exp_path = f"{base_exp_path}_{loss_name}_ft"
            model_id_list = get_chosen_ftmodels_path(csv_path=csv_path, loss_name=loss_name)
            root_exp_path = 'results/day-30-03-2023_hr-10-01-01_PIPELINE_50k_3models'
            model_id_list = [join(root_exp_path, m, 'checkpoints/last.pt') for m in model_id_list]
            # model_id_list = [model_id_list[5]]
        else:
            model_id_list = [i+1 for i in range(7)]
            exp_path = base_exp_path

        if not os.path.isdir(exp_path):
            os.makedirs(exp_path)
        
        # logger = init_logger(exp_path, fname=f'progress')    
        
        # logger.debug(args)
        
        # ts = get_cifar10_dataset(train=False, shuffle=False, num_samples=n_samples)
        # ts_loader = DataLoader(ts, batch_size=batch_size, shuffle=False)
        

        # set_all_seed(random_seed)
        # data = compute_distances_pipeline(model_id_list=model_id_list, 
        #                         exp_path=exp_path, ts_loader=ts_loader,
        #                         logger=logger, device=device,
        #                         random_seed=random_seed,
        #                         steps=steps)
        
        # data = retrieve_distances_data(model_id_list=model_id_list, exp_path=exp_path, logger=logger)
        
        
        # with open(join(exp_path, fname), 'wb') as f:
        #     pickle.dump(data, f)
        
        with open(join(exp_path, fname), 'rb') as f:
            datax = pickle.load(f)
        print("")
    print("")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-n_samples', default=2000, type=int)
    parser.add_argument('-batch_size', default=100, type=int) 
    parser.add_argument('-steps', default=50, type=int)   

    # parser.add_argument('-n_samples', default=2000, type=int)
    # parser.add_argument('-batch_size', default=50, type=int) 
    
    # parser.add_argument('-loss_name', default=['MixMSE-AT'], type=str, nargs='+')    
    parser.add_argument('-loss_name', default=[None, 'PCT', 'PCT-AT', 'MixMSE-AT'], type=str, nargs='+')
    
    parser.add_argument('-random_seed', default=0, type=int)
    args = parser.parse_args()
    
    # todo: finire di fare la pipeline per i 14 modelli finetunati
    
    main(args)

