from robustbench.utils import load_model
from secml.utils import fm
import os
import pickle
import pandas as pd
from utils.eval import predict
from utils.utils import MODEL_NAMES, preds_fname, advx_fname, PREDS_DIRNAME_DEFAULT, \
    ADVX_DIRNAME_DEFAULT, COLUMN_NAMES


def save_predictions(model_names, x_test, y_test, batch_size,
                     exp_folder_name, device, logger):

    advx_folder = fm.join(exp_folder_name, ADVX_DIRNAME_DEFAULT)
    predictions_folder = fm.join(exp_folder_name, PREDS_DIRNAME_DEFAULT)

    # ------ ASSERTIONS AND FOLDER MANAGEMENT ------ #
    assert fm.folder_exist(advx_folder), 'You must run generate_advx first'
    assert len(os.listdir(advx_folder)) > 0, 'Advx directory is empty'

    if not fm.folder_exist(predictions_folder):
        fm.make_folder(predictions_folder)

    with open(fm.join(advx_folder, advx_fname(model_names[0])), 'rb') as f:
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
            # model is already in eval mode

            # evaluate model_i with every set of clean samples and advx optimized on model_j
            for j, column_name in enumerate(COLUMN_NAMES[1:]):
                if j == 0:
                    x = x_test
                else:
                    # ------ LOAD ADVX ------ #
                    with open(fm.join(advx_folder, advx_fname(column_name)), 'rb') as f:
                        x = pickle.load(f)

                preds = predict(model, x, batch_size, device=device)
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