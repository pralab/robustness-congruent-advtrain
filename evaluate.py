from secml.utils import fm
import pandas as pd
import os
import numpy as np
from utils.eval import compute_nflips
from utils.utils import preds_fname, PERF_FNAME, NFLIPS_FNAME, OVERALL_RES_FNAME, \
    RESULTS_DIRNAME_DEFAULT, PREDS_DIRNAME_DEFAULT, \
    COLUMN_NAMES

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def evaluate_pipeline(model_names, exp_folder_name, logger, ft_models=False):
    performances_df = pd.DataFrame(columns=COLUMN_NAMES[1:])
    performances_df.index.name = 'model'
    nflips_df = pd.DataFrame(columns=COLUMN_NAMES[1:])
    nflips_df.index.name = 'model'

    predictions_folder = fm.join(exp_folder_name, f"{PREDS_DIRNAME_DEFAULT}{'_ft' if ft_models else ''}")

    # ------ ASSERTIONS AND FOLDER MANAGEMENT ------ #
    assert fm.folder_exist(predictions_folder), 'You must run save_predictions first'
    assert len(os.listdir(predictions_folder)) > 0, 'Predictions directory is empty'

    results_folder = fm.join(exp_folder_name, f"{RESULTS_DIRNAME_DEFAULT}{'_ft' if ft_models else ''}")
    if not fm.folder_exist(results_folder):
        fm.make_folder(results_folder)


    clean_accs = [] # each entry is clean acc of model i
    # Looping on the rows (models)
    for i, model_name_i in enumerate(model_names):
        try:
            df = pd.read_csv(fm.join(predictions_folder, preds_fname(model_name_i)),
                            index_col=0)
        except:
            logger.debug('Loading the original version predictions (w/o finetuning')
            df = pd.read_csv(fm.join(predictions_folder.replace('_ft', ''), preds_fname(model_name_i)),
                            index_col=0)
        correct_preds_df = pd.DataFrame(columns=COLUMN_NAMES[1:])
        true_labels_df = df.pop('True')
        for c in df:
            correct_preds_df[c] = (df[c] == true_labels_df)

        clean_accs.append(correct_preds_df['Clean'].mean())
        # correct_preds_df is specific for each model
        # A row of correct_preds_df refer to a original sample,
        # columns refer to perturbations of this sample:
        # first column is Clean, so no perturbation, and then all advx optimized against each model of the sequence
        # the entry (i,j) is True if the sample is classified correctly by the model

        rob_accs = []
        # compute robust accs
        for j, cols in enumerate(COLUMN_NAMES[1:]):
            rob_acc = correct_preds_df[COLUMN_NAMES[1:2+j]].all(axis=1).mean()
            # rob_acc = atk_succ_df[cols].mean()
            rob_accs.append(rob_acc)
        performances_df.loc[model_name_i] = rob_accs

    performances_df = (performances_df * 100).round(2)
    performances_df.to_csv(fm.join(results_folder, PERF_FNAME))
    logger.debug(performances_df)
    print("")