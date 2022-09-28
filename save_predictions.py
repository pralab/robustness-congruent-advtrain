from robustbench.utils import load_model
from secml.utils import fm
import os
import pickle
import pandas as pd
import torch
from utils.eval import predict
from utils.utils import MODEL_NAMES, custom_dirname, preds_fname, advx_fname, PREDS_DIRNAME_DEFAULT, \
    ADVX_DIRNAME_DEFAULT, COLUMN_NAMES, FINETUNING_DIRNAME_DEFAULT


def save_predictions(model_names, x_test, y_test, batch_size,
                     exp_folder_name, device, logger, ft_models=False):
    """
    All vs All predictions pipeline: each model predicts all generated datasets, clean and all wb attacks
    """
    advx_folder = fm.join(exp_folder_name, f"{ADVX_DIRNAME_DEFAULT}{'_ft' if ft_models else ''}")
    predictions_folder = fm.join(exp_folder_name, f"{PREDS_DIRNAME_DEFAULT}{'_ft' if ft_models else ''}")

    # ------ ASSERTIONS AND FOLDER MANAGEMENT ------ #
    assert fm.folder_exist(advx_folder), 'You must run generate_advx first'
    assert len(os.listdir(advx_folder)) > 0, 'Advx directory is empty'

    if not fm.folder_exist(predictions_folder):
        fm.make_folder(predictions_folder)

    with open(fm.join(advx_folder, advx_fname(model_names[1])), 'rb') as f:
        # prima era calcolato su model_names[0]!!!
        advx = pickle.load(f)
    assert x_test.shape[0] == advx.shape[0], 'number of clean samples different from number of saved advx'

    nope_list = []
    for i, model_name_i in enumerate(model_names):
        try:
            predictions_df = pd.DataFrame(columns=COLUMN_NAMES)
            predictions_df['True'] = y_test

            # ------ LOAD MODEL ------ #
            logger.debug(f"Loading model {i}: {model_name_i}")
            model = load_model(model_name=model_name_i, dataset='cifar10', threat_model='Linf')
            if ft_models:
                finetuned_models_folder = fm.join(exp_folder_name, FINETUNING_DIRNAME_DEFAULT)
                path = fm.join(finetuned_models_folder, f"{model_name_i}.pt")
                model.load_state_dict(torch.load(path))
            model.to(device)
            model.eval()
            # model is already in eval mode

            # evaluate model_i with every set of clean samples and advx optimized on model_j
            for j, column_name in enumerate(COLUMN_NAMES[1:]):
                # ------ LOAD ADVX ------ #
                try:
                    if j == 0:
                        logger.debug(f"Loading clean samples")
                        x = x_test
                    else:
                        with open(fm.join(advx_folder, advx_fname(column_name)), 'rb') as f:
                            x = pickle.load(f)
                    x = x.to(device)
                    preds, _ = predict(model, x, batch_size, device=device)
                    predictions_df[column_name] = preds
                except:
                    logger.debug(f"advx WB on {column_name} not existing.")
                    preds = [None]*x.shape[0]
                predictions_df[column_name] = preds   
            predictions_df.to_csv(fm.join(predictions_folder, preds_fname(model_name_i)))
        except:
            logger.debug(f"model {i} ({model_name_i}) not processed.")
            nope_list.append(model_name_i)
    print(f"Model not processed:\n{nope_list}")

    # ### MODIFICA TEMPORANEA PER AGGIUNGERE LE TRUE LABEL AL CSV
    # for i, model_name_i in enumerate(model_names):
    #     predictions_df = pd.read_csv(fm.join(predictions_folder, preds_fname(model_name_i)),
    #                                  index_col=0)
    #     if not 'True' in predictions_df.columns:
    #         predictions_df.insert(loc=0, column='True', value=y_test)
    #         predictions_df.to_csv(fm.join(predictions_folder, preds_fname(model_name_i)))
    #         print(predictions_df.head(10))


    print("")

def save_trainset_predictions(model_names, x_train, y_train, batch_size,
                            exp_folder_name, device, logger, pred_clean=True, pred_advx=False):

    predictions_trset_folder = fm.join(exp_folder_name, custom_dirname(PREDS_DIRNAME_DEFAULT, tr_set=True))
    advx_folder = fm.join(exp_folder_name, custom_dirname(ADVX_DIRNAME_DEFAULT, tr_set=True))

    if not fm.folder_exist(predictions_trset_folder):
        fm.make_folder(predictions_trset_folder)

    if not fm.folder_exist(advx_folder):
        fm.make_folder(advx_folder)

    for model_id, model_name in enumerate(model_names):
        logger.debug(f"model {model_id} ({model_name})")

        if pred_clean or pred_advx:
            model = load_model(model_name=model_name, dataset='cifar10', threat_model='Linf')

        if pred_clean:
            preds, outputs = predict(model, x_train, batch_size, device, logger)
            data = {'preds': preds, 'outs': outputs.cpu()}

            #salvo outputs in predictions folder come {model_name} se clean sample e {model_name} se advx
            with open(fm.join(predictions_trset_folder, f"{model_name}.gz"), 'wb') as f:
                pickle.dump(data, f)

        if pred_advx:
            try:
                with open(fm.join(advx_folder, advx_fname(model_name)), 'rb') as f:
                    # prendo advs[0] perchè sto usando un solo epsilon
                    advx = pickle.load(f)
                assert advx.shape[0] == x_train.shape[0]
                preds, outputs = predict(model, advx, batch_size, device, logger)
                data = {'preds': preds, 'outs': outputs.cpu()}
                with open(fm.join(predictions_trset_folder, advx_fname(model_name)), 'wb') as f:
                    pickle.dump(data, f)
            except:
                logger.debug(f"Advx of train samples on not exist")



    # fare una folder solo per prediction del train, con un file per ogni modello