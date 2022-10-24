from robustbench.utils import load_model
from secml.utils import fm
import pickle
from utils.utils import MODEL_NAMES
from utils.utils import advx_fname, custom_dirname, ADVX_DIRNAME_DEFAULT, FINETUNING_DIRNAME_DEFAULT
import math
import torch

# import foolbox as fb
from adv_lib.attacks.auto_pgd import apgd

import time


def generate_advx(x, y, batch_size, model_names, eps,
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
            model = load_model(model_name=model_name, dataset='cifar10', threat_model='Linf')
            if ft_models:
                finetuned_models_folder = fm.join(exp_folder_name, FINETUNING_DIRNAME_DEFAULT)
                path = fm.join(finetuned_models_folder, f"{model_name}.pt")
                model.load_state_dict(torch.load(path))
            model.to(device)

            # ------ COMPUTE ADVX ------ #
            # todo: fare un po' di debug degli attacchi, logger, verbose ecc
            start = time.time()
  
            advx = torch.Tensor([])
            n_batches = math.ceil(x.shape[0] / batch_size)
            for batch_i in range(n_batches):            
                start_i = batch_i * batch_size
                end_i = start_i + batch_size

                advx_i = apgd(model, x[start_i:end_i].to(device), y[start_i:end_i].to(device),
                            eps=eps, norm=float('inf'), n_iter=n_steps)
                advx = torch.cat((advx, advx_i.cpu()), 0)
                logger.debug(f"Computed advx: {batch_i}/{n_batches}")
            
            
            # advx = apgd(model, x.to(device), y.to(device),
            #             eps=eps, norm=float('inf'), n_iter=n_steps)

            end = time.time()
            # logger.debug('Robust accuracy: {:.1%}'.format(1 - success.float().mean()))
            logger.debug(f"Took {end - start:.2f} s")

            # ------ SAVE ADVX ------ #
            with open(fm.join(advx_folder, advx_fname(model_name)), 'wb') as f:
                # prendo advs[0] perchè sto usando un solo epsilon
                pickle.dump(advx, f)
        except:            
            logger.debug(f"{model_name} not processed.")
            nope_list.append(model_name)
    print(f"Model not processed:\n{nope_list}")
    print("")

if __name__ == '__main__':
    exp_folder_name = 'data/2ksample_250steps_100batchsize_bancoprova'
    advx_folder = fm.join(exp_folder_name, custom_dirname(ADVX_DIRNAME_DEFAULT, 
                                                    ft_models=False, tr_set=True))
    for model_name in MODEL_NAMES:
        with open(fm.join(advx_folder, advx_fname(model_name)), 'rb') as f:
            # prendo advs[0] perchè sto usando un solo epsilon
            advx = pickle.load(f)
        print(f"{model_name} -> {advx.shape[0]}")
    print("")