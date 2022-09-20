from robustbench.utils import load_model
from secml.utils import fm
import pickle
from utils.utils import advx_fname, ADVX_DIRNAME_DEFAULT
import math
import torch

<<<<<<< HEAD
# import foolbox as fb
=======
import foolbox as fb
>>>>>>> 146be9dd04e37ede9115366392e010c7b7a42240
from adv_lib.attacks.auto_pgd import apgd

import time


def generate_advx(x_test, y_test, batch_size, model_names, eps,
                  n_steps, exp_folder_name, logger, device):

    advx_folder = fm.join(exp_folder_name, ADVX_DIRNAME_DEFAULT)
    if not fm.folder_exist(advx_folder):
        fm.make_folder(advx_folder)

    nope_list = []
    for i, model_name in enumerate(model_names):
        try:
            # ------ LOAD MODEL ------ #
            logger.debug(f"Loading model {i}: {model_name}")
            model = load_model(model_name=model_name, dataset='cifar10', threat_model='Linf')
            model.to(device)
            # model is already in eval mode
            # todo: model to device


            # ------ COMPUTE ADVX ------ #

<<<<<<< HEAD

            # todo: fare un po' di debug degli attacchi, logger, verbose ecc
            start = time.time()

            # # todo: questo si può incapsulare meglio per scegliere l'attacco da terminale
            # fmodel = fb.PyTorchModel(model, bounds=(0, 1))
            # attack = fb.attacks.LinfPGD(steps=n_steps)
            # # lista esterna sono gli epsilon, lista interna sono i sample
            # _, advs, success = attack(fmodel, x_test, y_test, epsilons=[eps])
            # advx = advs[0]
            # x_test, y_test = x_test.to(device), y_test.to(device)
=======
        # ------ COMPUTE ADVX ------ #


        # todo: fare un po' di debug degli attacchi, logger, verbose ecc
        start = time.time()

        # # todo: questo si può incapsulare meglio per scegliere l'attacco da terminale
        # fmodel = fb.PyTorchModel(model, bounds=(0, 1))
        # attack = fb.attacks.LinfPGD(steps=n_steps)
        # # lista esterna sono gli epsilon, lista interna sono i sample
        # _, advs, success = attack(fmodel, x_test, y_test, epsilons=[eps])
        # advx = advs[0]

        advx = apgd(model, x_test, y_test,
                    eps=eps, norm=float('inf'), n_iter=n_steps)

        end = time.time()
        # logger.debug('Robust accuracy: {:.1%}'.format(1 - success.float().mean()))
        logger.debug(f"Took {end - start:.2f} s")

        # ------ SAVE ADVX ------ #
        with open(fm.join(advx_folder, advx_fname(model_name)), 'wb') as f:
            # prendo advs[0] perchè sto usando un solo epsilon
            pickle.dump(advx, f)
>>>>>>> 146be9dd04e37ede9115366392e010c7b7a42240

            
            advx = torch.Tensor([])
            n_batches = math.ceil(x_test.shape[0] / batch_size)
            for batch_i in range(n_batches):            
                start_i = batch_i * batch_size
                end_i = start_i + batch_size

                advx_i = apgd(model, x_test[start_i:end_i].to(device), y_test[start_i:end_i].to(device),
                            eps=eps, norm=float('inf'), n_iter=n_steps)
                advx = torch.cat((advx, advx_i.cpu()), 0)
                logger.debug(f"Computed advx: {batch_i}/{n_batches}")
            
            
            # advx = apgd(model, x_test.to(device), y_test.to(device),
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
