from secml.utils import fm
import pandas as pd
import os
from utils.eval import compute_nflips
from utils.utils import preds_fname, PERF_FNAME, NFLIPS_FNAME, \
    RESULTS_DIRNAME_DEFAULT, PREDS_DIRNAME_DEFAULT, \
    COLUMN_NAMES

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def evaluate_pipeline(model_names, y, exp_folder_name, logger):
    y = pd.Series(y.tolist())

    performances_df = pd.DataFrame(columns=COLUMN_NAMES)
    performances_df.index.name = 'model'
    nflips_df = pd.DataFrame(columns=COLUMN_NAMES)
    nflips_df.index.name = 'model'

    predictions_folder = fm.join(exp_folder_name, PREDS_DIRNAME_DEFAULT)

    # ------ ASSERTIONS AND FOLDER MANAGEMENT ------ #
    assert fm.folder_exist(predictions_folder), 'You must run save_predictions first'
    assert len(os.listdir(predictions_folder)) > 0, 'Predictions directory is empty'

    results_folder = fm.join(exp_folder_name, RESULTS_DIRNAME_DEFAULT)
    if not fm.folder_exist(results_folder):
        fm.make_folder(results_folder)

    for i, model_name_i in enumerate(model_names):
        df = pd.read_csv(fm.join(predictions_folder, preds_fname(model_name_i)),
                         index_col=0)

        # Compute performance
        data = []
        for j, col in enumerate(df):
            if j > i + 1:
                data.append(None)
            else:
                acc = (df[col] == y).mean()
                data.append(acc)
        performances_df.loc[model_name_i] = data

        # Compute Negative Flips
        nfs = []
        for j, col in enumerate(df):
            if i == 0 or j > i:
                nfs.append(None)
            else:
                old_df = pd.read_csv(fm.join(predictions_folder, preds_fname(model_names[i-1])), index_col=0)
                nf = compute_nflips(old_preds=old_df[df.columns[j]], new_preds=df[df.columns[j]], y=y)
                nfs.append(nf)
        nflips_df.loc[model_name_i] = nfs

    logger.info(f"\n---------------\nAccuracy\n---------------\n{performances_df}")
    performances_df.to_csv(fm.join(results_folder, PERF_FNAME))
    logger.info(f"\n---------------\nNegative Flips\n---------------\n{nflips_df}")
    nflips_df.to_csv(fm.join(results_folder, NFLIPS_FNAME))

    print("")