from secml.utils import fm
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.eval import compute_nflips
from utils.utils import preds_fname, PERF_FNAME, NFLIPS_FNAME, OVERALL_RES_FNAME, \
    RESULTS_DIRNAME_DEFAULT, PREDS_DIRNAME_DEFAULT, \
    COLUMN_NAMES

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def evaluate_pipeline(model_names, exp_folder_name, logger):
    rob_accs_df = pd.DataFrame(columns=COLUMN_NAMES[1:])
    rob_accs_df.index.name = 'model'
    nflips_df = pd.DataFrame(columns=COLUMN_NAMES[1:])
    nflips_df.index.name = 'model'

    predictions_folder = fm.join(exp_folder_name, PREDS_DIRNAME_DEFAULT)

    # ------ ASSERTIONS AND FOLDER MANAGEMENT ------ #
    assert fm.folder_exist(predictions_folder), 'You must run save_predictions first'
    assert len(os.listdir(predictions_folder)) > 0, 'Predictions directory is empty'

    results_folder = fm.join(exp_folder_name, RESULTS_DIRNAME_DEFAULT)
    if not fm.folder_exist(results_folder):
        fm.make_folder(results_folder)


    clean_accs = [] # each entry is clean acc of model i
    old_correct_preds_df = None
    # Looping on the rows (models)
    for i, model_name_i in enumerate(model_names):
        df = pd.read_csv(fm.join(predictions_folder, preds_fname(model_name_i)),
                         index_col=0)
        correct_preds_df = pd.DataFrame(columns=COLUMN_NAMES[1:])
        true_labels_df = df.pop('True')
        for c in df:
            correct_preds_df[c] = (df[c] == true_labels_df)

        clean_accs.append(correct_preds_df['Clean'].mean())
        # correct_preds_df is specific for each model
        # A row of correct_preds_df refer to an original sample,
        # columns refer to perturbations of this sample:
        # first column is Clean, so no perturbation, and then all advx optimized on each model of the sequence
        # the entry (i,j) is True if the sample is classified correctly by the model


        # compute robust accs and nflips
        rob_accs = []
        nflips = []
        for j, cols in enumerate(COLUMN_NAMES[1:]):
            adv_logical_or = correct_preds_df[COLUMN_NAMES[1:2+j]].all(axis=1)
            rob_acc = adv_logical_or.mean()
            # rob_acc = atk_succ_df[cols].mean()
            rob_accs.append(rob_acc)

            if i == 0:
                nflips.append(None)
            else:
                old_adv_logical_or = old_correct_preds_df[COLUMN_NAMES[1:2+j]].all(axis=1)
                nflips.append((old_adv_logical_or & ~adv_logical_or).mean())

        # Keep the predictions of the previous model
        old_correct_preds_df = correct_preds_df
        rob_accs_df.loc[model_name_i] = rob_accs
        nflips_df.loc[model_name_i] = nflips

        #compute churns


    # performances_df = (performances_df * 100).round(2)
    rob_accs_df = rob_accs_df.round(4)
    rob_accs_df.to_csv(fm.join(results_folder, PERF_FNAME))
    # print(rob_accs_df)

    nflips_df = nflips_df.round(4)
    nflips_df.to_csv(fm.join(results_folder, NFLIPS_FNAME))
    # print(nflips_df)


    overall_df = pd.DataFrame(np.nan, index=rob_accs_df.index, columns=rob_accs_df.columns, dtype=str)
    for i in range(overall_df.shape[0]):  # iterate over rows
        for j in range(overall_df.shape[1]): # iterate over columns
            robacc = rob_accs_df.at[overall_df.index[i], COLUMN_NAMES[1:][j]]
            nflips = nflips_df.at[overall_df.index[i], COLUMN_NAMES[1:][j]]
            overall_df.at[overall_df.index[i], COLUMN_NAMES[1:][j]] = \
                f"{(robacc)} ({'-' if np.isnan(nflips) else nflips})"
            # if (j > i) or (j == 0):
            #     robacc = rob_accs_df.at[overall_df.index[i], COLUMN_NAMES[1:][j]]
            #     nflips = nflips_df.at[overall_df.index[i], COLUMN_NAMES[1:][j]]
            #     overall_df.at[overall_df.index[i], COLUMN_NAMES[1:][j]] = \
            #         f"{(robacc)} ({'-' if np.isnan(nflips) else nflips})"
            # else:
            #     overall_df.at[overall_df.index[i], COLUMN_NAMES[1:][j]] = '-'
            #     rob_accs_df.at[overall_df.index[i], COLUMN_NAMES[1:][j]] = None
            #     nflips_df.at[overall_df.index[i], COLUMN_NAMES[1:][j]] = None
    print(overall_df)
    overall_df.to_csv(fm.join(results_folder, OVERALL_RES_FNAME))


    beta = 0.8
    plt.figure(figsize=(15*beta, 10*beta))
    sns.heatmap(rob_accs_df*100, annot=True, fmt='g')
    plt.title("Robust accuracy")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15*beta, 10*beta))
    sns.heatmap(nflips_df, annot=True, fmt='g')
    plt.title("Negative Flips (%)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15*beta, 10*beta))
    sns.heatmap(nflips_df*2000, annot=True, fmt='g')
    plt.title("Negative Flips")
    plt.tight_layout()
    plt.show()

    print("")





# def evaluate_pipeline(model_names, y, exp_folder_name, logger):
#     y = pd.Series(y.tolist())
#
#     performances_df = pd.DataFrame(columns=COLUMN_NAMES)
#     performances_df.index.name = 'model'
#     nflips_df = pd.DataFrame(columns=COLUMN_NAMES)
#     nflips_df.index.name = 'model'
#
#     predictions_folder = fm.join(exp_folder_name, PREDS_DIRNAME_DEFAULT)
#
#     # ------ ASSERTIONS AND FOLDER MANAGEMENT ------ #
#     assert fm.folder_exist(predictions_folder), 'You must run save_predictions first'
#     assert len(os.listdir(predictions_folder)) > 0, 'Predictions directory is empty'
#
#     results_folder = fm.join(exp_folder_name, RESULTS_DIRNAME_DEFAULT)
#     if not fm.folder_exist(results_folder):
#         fm.make_folder(results_folder)
#
#     for i, model_name_i in enumerate(model_names):
#         df = pd.read_csv(fm.join(predictions_folder, preds_fname(model_name_i)),
#                          index_col=0)
#
#         # Compute performance
#         data = []
#         for j, col in enumerate(df):
#             if j > i + 1:
#                 data.append(None)
#             else:
#                 acc = (df[col] == y).mean()
#                 data.append(acc)
#         performances_df.loc[model_name_i] = data
#
#         # Compute Negative Flips
#         nfs = []
#         for j, col in enumerate(df):
#             if i == 0 or j > i:
#                 nfs.append(None)
#             else:
#                 old_df = pd.read_csv(fm.join(predictions_folder, preds_fname(model_names[i-1])), index_col=0)
#                 nf = compute_nflips(old_preds=old_df[df.columns[j]], new_preds=df[df.columns[j]], y=y)
#                 nfs.append(nf)
#         nflips_df.loc[model_name_i] = nfs
#
#     logger.info(f"\n---------------\nAccuracy\n---------------\n{performances_df}")
#     performances_df.to_csv(fm.join(results_folder, PERF_FNAME))
#     logger.info(f"\n---------------\nNegative Flips\n---------------\n{nflips_df}")
#     nflips_df.to_csv(fm.join(results_folder, NFLIPS_FNAME))
#
#     print("")