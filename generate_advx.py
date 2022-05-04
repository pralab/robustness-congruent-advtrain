from robustbench.utils import load_model
from secml.utils import fm
import pickle
from utils.utils import advx_fname, ADVX_DIRNAME_DEFAULT

import foolbox as fb
# from adv_lib.attacks import auto_pgd

import time


def generate_advx(x_test, y_test, model_names, eps,
                  n_steps, exp_folder_name, logger, device):

    advx_folder = fm.join(exp_folder_name, ADVX_DIRNAME_DEFAULT)
    if not fm.folder_exist(advx_folder):
        fm.make_folder(advx_folder)

    for i, model_name in enumerate(model_names):

        # ------ LOAD MODEL ------ #
        logger.debug(f"Loading model {i}: {model_name}")
        model = load_model(model_name=model_name, dataset='cifar10', threat_model='Linf')
        # model is already in eval mode
        # todo: model to device


        # ------ COMPUTE ADVX ------ #
        # todo: questo si può incapsulare meglio per scegliere l'attacco da terminale
        fmodel = fb.PyTorchModel(model, bounds=(0, 1))
        attack = fb.attacks.LinfPGD(steps=n_steps)

        # todo: fare un po' di debug degli attacchi, logger, verbose ecc
        start = time.time()
        # lista esterna sono gli epsilon, lista interna sono i sample
        _, advs, success = attack(fmodel, x_test, y_test, epsilons=[eps])
        end = time.time()
        logger.debug('Robust accuracy: {:.1%}'.format(1 - success.float().mean()))
        logger.debug(f"Took {end - start:.2f} s")

        # ------ SAVE ADVX ------ #
        with open(fm.join(advx_folder, advx_fname(model_name)), 'wb') as f:
            # prendo advs[0] perchè sto usando un solo epsilon
            pickle.dump(advs[0], f)

    print("")
