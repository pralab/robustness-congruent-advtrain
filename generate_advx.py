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

# import foolbox as fb
from adv_lib.attacks.auto_pgd import apgd

import time

def generate_advx_ds(model, ds_loader, ds_path, logger=None, device=None,
                    eps=0.03, n_steps=250):
    """
    crea una cartella dove per ogni sample salva il singolo tensore in ds_path.
    i sample sono rinominati in ordine crescente come vengono incontrati nel ds originale
    """
    
    model.to(device)
    model.eval()

    k=0 # index for samples
    with tqdm(total=len(ds_loader)) as t:
        for batch_idx, (x,y) in enumerate(ds_loader):
            x, y = x.to(device), y.to(device)
            # x.requires_grad = True
            advx = apgd(model, x, y,
                        eps=eps, norm=float('inf'), n_iter=n_steps)
            t.set_postfix(
                compl='[{}/{} ({:.0f}%)]'.format(
                    batch_idx * len(x),
                    len(ds_loader.dataset),
                    100. * batch_idx / len(ds_loader)))
            t.update()

            for i in range(len(advx)):
                file_path = os.path.join(ds_path, f"{str(k).zfill(10)}.gz")
                advx = advx.detach().cpu()
                y = y.detach().cpu()
                data = (advx[i], y[i])
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
                k += 1
    
    print("")




def generate_advx(ds_loader, model_names, eps,
                  n_steps, exp_folder_name, logger, device, ft_models=False, tr_set=False):
    """
    ft_models e tr_set specificano solo una cartella diversa rispetto agli advx 
    per i modelli originali sul test set

    cartella advx_folder con dentro gli advx WB per ogni modello selezionato
    """
    advx_folder = fm.join(exp_folder_name, custom_dirname(ADVX_DIRNAME_DEFAULT, 
                                                        ft_models=ft_models, tr_set=tr_set))
    if not fm.folder_exist(advx_folder):
        fm.make_folder(advx_folder)

    nope_list = []
    for i, model_name in enumerate(model_names):
        try:
            # ------ LOAD MODEL ------ #
            logger.debug(f"Loading model {i}: {model_name}")
            ds_path = fm.join(advx_folder, advx_fname(model_name))

            if model_name in model_names:
                model = load_model(model_name=model_name, dataset='cifar10', threat_model='Linf')
            else:
                finetuned_models_folder = fm.join(exp_folder_name, FINETUNING_DIRNAME_DEFAULT)
                path = fm.join(finetuned_models_folder, f"{model_name}.pt")
                model.load_state_dict(torch.load(path))
            model.to(device)

            # ------ COMPUTE ADVX ------ #
            # todo: fare un po' di debug degli attacchi, logger, verbose ecc
            start = time.time()
            generate_advx_ds(model=model, ds_loader=ds_loader, ds_path=ds_path,
                             logger=logger, device=device, n_steps=n_steps, eps=eps)
            end = time.time()
            # logger.debug('Robust accuracy: {:.1%}'.format(1 - success.float().mean()))
            logger.debug(f"Took {end - start:.2f} s")

        except:            
            logger.debug(f"{model_name} not processed.")
            nope_list.append(model_name)
    print(f"Model not processed:\n{nope_list}")
    print("")



def get_models_info_list(root, advx_folder):
    old_models_idx = []
    new_models_idx = []
    models_info_list = []
    for path, dirs, files in os.walk(root):
        if ('checkpoints' in path) and (len(files)!=0):
            if 'best_nfr.pt' in files:
                tr_model_sel = 'best_nfr'                
            else:
                tr_model_sel = 'last'
            model_path = os.path.join(path, f"{tr_model_sel}.pt")
            robustbench_idx = int(path.split('_new-')[1].split('/')[0])
            robustbench_old_idx = int(path.split('old-')[1].split('_new')[0])
            hparams = path.split('/')[-2]
            loss_name = path.split('/')[-3]
            models_pair = path.split('/')[-4]
            advx_path = os.path.join(advx_folder, models_pair, loss_name, hparams)

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













if __name__ == '__main__':
    from utils.data import get_cifar10_dataset, MyTensorDataset
    from torch.utils.data import DataLoader
    from utils.visualization import imshow
    from utils.eval import evaluate_acc
    

    batch_size = 200
    n_steps = 50
    num_samples = 2000


    set_all_seed(0)
    root = 'results/day-04-11-2022_hr-16-50-24_epochs-12_batchsize-500'
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'


    advx_folder = os.path.join(root, 
                        custom_dirname(ADVX_DIRNAME_DEFAULT, ft_models=True, tr_set=False))
    if not os.path.isdir(advx_folder):
        os.mkdir(advx_folder)


    logger = init_logger(advx_folder, fname='progress_only_preds')
    models_info_list, old_models_idx, new_models_idx = get_models_info_list(root, advx_folder)


    logger.info('>>> Baseline Robustness')
    models_id = np.unique(np.array(new_models_idx + old_models_idx))
    nope_list = []
    for model_id in models_id:
        model_name = MODEL_NAMES[model_id]
        logger.info(f"-> {model_id} - {model_name}")   
        model = load_model(model_name=model_name, dataset='cifar10', threat_model='Linf')
        base_path = os.path.join('results', 'advx', model_name)
        ds_path = os.path.join(base_path, 'ts')
        if not os.path.isdir(ds_path):
            os.makedirs(ds_path)
        
        batch_size_temp = batch_size
        escape_flag = 4
        while(True):
            try:
                # ds = get_cifar10_dataset(train=False, num_samples=num_samples)
                # ds_loader = DataLoader(ds, batch_size=batch_size_temp, shuffle=False)
                # # if not os.path.exists(ds_path):
                # generate_advx_ds(model=model, ds_loader=ds_loader, 
                #                 ds_path=ds_path, device=device, n_steps=n_steps)
                # logger.debug("Advx generation completed.")
                # else:
                #     logger.debug('Advx already existing')
                

                corrects_path = os.path.join(base_path, 'correct_preds.gz')
                # if not os.path.exists(corrects_path):
                adv_ds = MyTensorDataset(ds_path=ds_path)
                adv_ds_loader = DataLoader(adv_ds, batch_size=batch_size)
                correct_preds = correct_predictions(model=model, test_loader=adv_ds_loader, device=device)
                with open(corrects_path, 'wb') as f:
                    pickle.dump(correct_preds.cpu(), f)
                logger.debug("Preds completed.")
                # else:
                #     logger.debug('Preds already existing')


            except Exception as e:
                logger.debug(f"Model {model_id} failed: {e}")
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


    


    
    logger.info('>>> PCT Robustness')
    for i, model_info in enumerate(models_info_list):
        logger.info(f"Exp {i+1} / {len(models_info_list)} ->")

        # if not os.path.exists(os.path.join(model_info['advx_path'], f"results_{model_info['tr_model_sel']}.gz")):                
        model_name = MODEL_NAMES[model_info['robustbench_idx']]
        
        # Load finetuned model
        model = load_model(model_name=model_name, dataset='cifar10', threat_model='Linf')
        checkpoint = torch.load(model_info['model_path'])
        model.load_state_dict(checkpoint['model_state_dict'])

        ds_path = os.path.join(model_info['advx_path'], 'ts')
        if not os.path.isdir(ds_path):
            os.makedirs(ds_path)
        
        batch_size_temp = batch_size
        escape_flag = 5
        while(True):
            try:
                # # Reload dataset with updated batch_size
                # ds = get_cifar10_dataset(train=False, num_samples=num_samples)
                # ds_loader = DataLoader(ds, batch_size=batch_size_temp, shuffle=False)

                # generate_advx_ds(model=model, ds_loader=ds_loader, 
                #                 ds_path=ds_path, device=device, n_steps=n_steps)
                # logger.debug("Advx generation completed.")


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
                # else:
                #     logger.debug('Advx already existing.')

            except Exception as e:
                logger.debug(f"Model {model_id} failed: {e}")
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

    logger.info("Pipeline completed :D")