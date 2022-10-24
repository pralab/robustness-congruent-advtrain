from secml.utils import fm
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl
from utils.eval import compute_nflips
from utils.utils import preds_fname, PERF_FNAME, NFLIPS_FNAME, OVERALL_RES_FNAME, \
    RESULTS_DIRNAME_DEFAULT, PREDS_DIRNAME_DEFAULT, \
    COLUMN_NAMES, init_logger, MODEL_NAMES, model_name_to_M_i

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def evaluate_pipeline(model_names, exp_folder_name, logger,
                      preds_dirname=PREDS_DIRNAME_DEFAULT,
                      results_dirname=RESULTS_DIRNAME_DEFAULT):
    rob_accs_df = pd.DataFrame(columns=COLUMN_NAMES[1:])
    rob_accs_df.index.name = 'model'
    nflips_df = pd.DataFrame(columns=COLUMN_NAMES[1:])
    nflips_df.index.name = 'model'

    predictions_folder = fm.join(exp_folder_name, preds_dirname)

    # ------ ASSERTIONS AND FOLDER MANAGEMENT ------ #
    assert fm.folder_exist(predictions_folder), 'You must run save_predictions first'
    assert len(os.listdir(predictions_folder)) > 0, 'Predictions directory is empty'

    results_folder = fm.join(exp_folder_name, results_dirname)
    if not fm.folder_exist(results_folder):
        fm.make_folder(results_folder)


    # clean_accs = [] # each entry is clean acc of model i
    old_correct_preds_df = None
    # Looping on the rows (models)
    for i, model_name_i in enumerate(model_names):
        print(f"{i}: {model_name_i}")
        df = pd.read_csv(fm.join(predictions_folder, preds_fname(model_name_i)),
                         index_col=0)
        correct_preds_df = pd.DataFrame(columns=COLUMN_NAMES[1:])
        true_labels_df = df.pop('True')
        for c in df:
            correct_preds_df[c] = (df[c] == true_labels_df)

        # clean_accs.append(correct_preds_df['Clean'].mean())
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
            # adv_logical_or = correct_preds_df[COLUMN_NAMES[1 + j]]

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
    rob_accs_df = (rob_accs_df*100).round(4)
    rob_accs_df.to_csv(fm.join(results_folder, PERF_FNAME))
    # print(rob_accs_df)

    nflips_df = (nflips_df*100).round(4)
    nflips_df.to_csv(fm.join(results_folder, NFLIPS_FNAME))
    # print(nflips_df)


    overall_df = pd.DataFrame(np.nan, index=rob_accs_df.index, columns=rob_accs_df.columns, dtype=str)
    for i in range(overall_df.shape[0]):  # iterate over rows
        for j in range(overall_df.shape[1]): # iterate over columns
            # robacc = rob_accs_df.at[overall_df.index[i], COLUMN_NAMES[1:][j]]
            # nflips = nflips_df.at[overall_df.index[i], COLUMN_NAMES[1:][j]]
            # overall_df.at[overall_df.index[i], COLUMN_NAMES[1:][j]] = \
            #     f"{(robacc)} ({'-' if np.isnan(nflips) else nflips})"
            if (j > i) or (j == 0):
                robacc = rob_accs_df.at[overall_df.index[i], COLUMN_NAMES[1:][j]]
                nflips = nflips_df.at[overall_df.index[i], COLUMN_NAMES[1:][j]]
                overall_df.at[overall_df.index[i], COLUMN_NAMES[1:][j]] = \
                    f"{(robacc)} ({'-' if np.isnan(nflips) else nflips})"
            else:
                overall_df.at[overall_df.index[i], COLUMN_NAMES[1:][j]] = '-'
                rob_accs_df.at[overall_df.index[i], COLUMN_NAMES[1:][j]] = None
                nflips_df.at[overall_df.index[i], COLUMN_NAMES[1:][j]] = None
    print(overall_df)
    overall_df.to_csv(fm.join(results_folder, OVERALL_RES_FNAME))


    beta = 0.8
    plt.figure(figsize=(15*beta, 10*beta))
    sns.heatmap(rob_accs_df, annot=True, fmt='g')
    plt.title("Robust accuracy")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15*beta, 10*beta))
    sns.heatmap(nflips_df, annot=True, fmt='g')
    plt.title("Negative Flips (%)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15*beta, 10*beta))
    sns.heatmap(nflips_df*2000/100, annot=True, fmt='g')
    plt.title("Negative Flips")
    plt.tight_layout()
    plt.show()

    print("")

def matrix_churn(model_names, exp_folder_name, logger, forw_retr_sel=True,
                 preds_dirname=PREDS_DIRNAME_DEFAULT,
                 results_dirname=RESULTS_DIRNAME_DEFAULT):
    """

    :param model_names:
    :param exp_folder_name:
    :param logger:
    :param forw_retr_sel: True if forward looking, False if retrospective
    :return:
    """
    # predictions_folder =

    # # ------ ASSERTIONS AND FOLDER MANAGEMENT ------ #
    # assert fm.folder_exist(predictions_folder), 'You must run save_predictions first'
    # assert len(os.listdir(predictions_folder)) > 0, 'Predictions directory is empty'
    #
    # results_folder = fm.join(exp_folder_name, RESULTS_DIRNAME_DEFAULT)
    # if not fm.folder_exist(results_folder):
    #     fm.make_folder(results_folder)

    beta = 0.2
    plt.figure(figsize=(15, 5))
    matrix_churn_dict = {}
    old_preds_df = None
    for i, model_name_i in enumerate(model_names):
        # Faccio tabella per ogni modello in sequenza
        print(f"{i}: {model_name_i}")
        # Carico predizioni di M_i
        df = pd.read_csv(fm.join(exp_folder_name,
                                 preds_dirname,
                                 preds_fname(model_names[i])),
                         index_col=0)
        attack_success_df = pd.DataFrame(columns=COLUMN_NAMES[1:])  # per ogni sample 1 se attacco funge 0 otherwise

        # Togli True cols da df e conservalo
        true_labels_df = df.pop('True')
        for c in df:
            attack_success_df[c] = ~(df[c] == true_labels_df)

        # ab: a=clean sample misclass?, b=stesso sample ma advx misclass?
        cols = ['00', '01', '11']
        # Skippa il primo modello perchè non posso calcolare churn
        if i > 0:
            # Qui cambio a seconda di forward-looking / retrospective
            if forw_retr_sel:
                adv_logical_or = attack_success_df[COLUMN_NAMES[1:3 + i]].any(axis=1)
                old_adv_logical_or = old_attack_success_df[COLUMN_NAMES[1:3 + i]].any(axis=1)
            else: # I have to consider the best evaluation at the last timestamp
                adv_logical_or = attack_success_df.any(axis=1)
                old_adv_logical_or = old_attack_success_df.any(axis=1)

            # build matrix wrt M_i VS M_{i-1}
            matrix_churn_df = pd.DataFrame(columns=cols)
            # matrix_churn_df.index.name = ?

            for c_i in cols:
                row_i = []
                for c_j in cols:
                    acc_lab1 = bool(int(c_i[0]))
                    rob_lab1 = bool(int(c_i[1]))
                    acc_lab2 = bool(int(c_j[0]))
                    rob_lab2 = bool(int(c_j[1]))

                    churn = (old_attack_success_df['Clean'] == acc_lab1) & \
                            (old_adv_logical_or == rob_lab1) & \
                            (attack_success_df['Clean'] == acc_lab2) & \
                            (adv_logical_or == rob_lab2)
                    row_i.append(churn.mean())

                matrix_churn_df.loc[c_i] = row_i

            matrix_churn_df.loc['Sum'] = matrix_churn_df.sum(axis=0)
            matrix_churn_df['Sum'] = matrix_churn_df.sum(axis=1)
            matrix_churn_dict[model_name_i] = matrix_churn_df
            print(matrix_churn_df)
            print(f"Clean MY: {matrix_churn_df.loc[['00', '01']].sum().sum()*100}, "
                  f"Clean MX: {matrix_churn_df[['00', '01']].sum().sum()*100},"
                  f"Sum: {matrix_churn_df.sum().sum()}")


            # plt.figure(figsize=(15 * beta, 15 * beta))
            plt.subplot(2, 5, i)
            sns.heatmap(matrix_churn_df*100, annot=True, fmt='g')
            # plt.title("Churn types")
            plt.xlabel(model_names[i])
            plt.ylabel(model_names[i-1])
            plt.tight_layout()
            # plt.show()

            # beta = 0.2
            # plt.figure(figsize=(15 * beta, 15 * beta))
            # sns.heatmap(matrix_churn_df * 100, annot=True, fmt='g', ax=axes[i % 5, j])
            # plt.title("Churn types")
            # plt.xlabel(model_names[i])
            # plt.ylabel(model_names[i - 1])

        # Keep preds of previos model
        old_attack_success_df = attack_success_df
    plt.suptitle("Churn types")
    plt.tight_layout()
    figures_folder = fm.join(results_dirname, 'figures')

    if not fm.folder_exist(figures_folder):
        fm.make_folder(figures_folder)

    # plt.savefig(fm.join(exp_folder_name, figures_folder,
    #                     f'{"fwl" if forw_retr_sel else "ret"}_churns.png'), dpi=300)
    # plt.show()

    with open(fm.join(exp_folder_name, results_dirname,
                      f'{"fwl" if forw_retr_sel else "ret"}_matrix_churn.pkl'), "wb") as f:
        pkl.dump(matrix_churn_dict, f)

    print("")


def table_churn(model_names, exp_folder_name, logger, forw_retr_sel=True,
                preds_dirname=PREDS_DIRNAME_DEFAULT,
                results_dirname=RESULTS_DIRNAME_DEFAULT):
    predictions_folder = fm.join(exp_folder_name, preds_dirname)
    results_folder = fm.join(exp_folder_name, results_dirname)

    # for i, model_name_i in enumerate(MODEL_NAMES):
    #     print(f"{i}: {model_name_i}")
    #     df = pd.read_csv(fm.join(predictions_folder, preds_fname(model_name_i)),
    #                      index_col=0)
    #     correct_preds_df = pd.DataFrame(columns=COLUMN_NAMES[1:])
    #     true_labels_df = df.pop('True')
    #     for c in df:
    #         correct_preds_df[c] = (df[c] == true_labels_df)

    robacc_df = pd.read_csv(fm.join(results_folder, PERF_FNAME),
                            index_col=0)
    nflips_df = pd.read_csv(fm.join(results_folder, NFLIPS_FNAME),
                            index_col=0)

    # Evaluations on the fly with cumulative advx
    cum_rob_acc = []
    cum_nflips = []
    for i, m in enumerate(model_names):
        for j, c in enumerate(COLUMN_NAMES[1:]):
            if (j == (i + 1)):
                cum_rob_acc.append(robacc_df.loc[model_names[i]][COLUMN_NAMES[1:][j]])
                cum_nflips.append(nflips_df.loc[model_names[i]][COLUMN_NAMES[1:][j]])

    # Best possible evaluation with all advx in the hystory
    robacc_df_best = robacc_df[[robacc_df.columns[0], robacc_df.columns[-1]]]
    robacc_df_best = robacc_df_best.rename(columns={robacc_df.columns[-1]: 'Robust Acc.',
                                                    'Clean': 'Clean Acc.'})
    nflips_df_best = nflips_df[[nflips_df.columns[0], nflips_df.columns[-1]]]
    nflips_df_best = nflips_df_best.rename(columns={nflips_df.columns[-1]: 'Robust Churn',
                                                    'Clean': 'Acc. Churn'})
    best_res = pd.concat([robacc_df_best, nflips_df_best], axis=1)

    cum_res = best_res.copy()
    cum_res['Robust Acc.'] = cum_rob_acc
    cum_res['Robust Churn'] = cum_nflips

    with open(fm.join(exp_folder_name, results_dirname,
                      f'{"fwl" if forw_retr_sel else "ret"}_matrix_churn.pkl'), "rb") as f:
        matrix_churn_dict = pkl.load(f)

    print(best_res)
    print(cum_res)

    common_churn_list = [None]
    for k in matrix_churn_dict:
        common_churn = matrix_churn_dict[k].loc['00']['11']*100
        common_churn_list.append(common_churn)

    _, mi_dict = model_name_to_M_i(model_names)
    df = cum_res if forw_retr_sel else best_res
    df = df.rename(index=mi_dict)

    beta = 0.5
    plt.figure(figsize=(2 * 15 * beta, 10 * beta))
    plt.subplot(1, 2, 1)
    sns.heatmap(cum_res.transpose(), annot=True, fmt='g')
    plt.title("Forward-looking Evaluation")
    plt.yticks(rotation=45)
    # plt.xticks(rotation=45)
    # plt.tight_layout()

    # plt.figure(figsize=(15 * beta, 10 * beta))
    plt.subplot(1, 2, 2)
    sns.heatmap(best_res.transpose(), annot=True, fmt='g')
    plt.title("Retrospective Evaluation")
    plt.yticks(rotation=45)
    # plt.xticks(rotation=45)

    plt.tight_layout()
    # plt.savefig(fm.join(results_folder, 'figures',
    #                     f'{"fwl" if forw_retr_sel else "ret"}_heatmap_perf.png'), dpi=300)
    plt.show()

    df['Common Churn'] = common_churn_list
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title("Accuracy")
    df.plot(y=['Clean Acc.', 'Robust Acc.'], ax=plt.gca(),
            color=['r', 'b'], style=['-o', '-*'],
            ylim=(0, 100))
    # plt.xlabel('models')
    # plt.ylim((0, 1))
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Churn")
    df.plot(y=['Acc. Churn', 'Robust Churn', 'Common Churn'], ax=plt.gca(),
            color=['r', 'b', 'g'], style=['-o', '-*', '--+'],
            ylim=(0, 12))
    plt.legend(loc='upper left')
    plt.tight_layout()
    figures_folder = fm.join(results_dirname, 'figures')
    if not fm.folder_exist(figures_folder):
        fm.make_folder(figures_folder)

    # plt.savefig(fm.join(figures_folder,
    #                     f'{"fwl" if forw_retr_sel else "ret"}_plot.png'), dpi=300)
    plt.show()



    print("")





if __name__ == '__main__':
    EXP_FOLDER_NAME = 'data/2ksample_250steps_100batchsize_day-09-07-2022_hr-19-46-34'
    # EXP_FOLDER_NAME = 'data/new'
    logger = init_logger(EXP_FOLDER_NAME)
    model_names = MODEL_NAMES
    # preds_dirname = 'predictions_ft'
    # results_dirname = 'results_ft'
    # for b in [True, False]:
    #     matrix_churn(model_names=model_names,
    #                  exp_folder_name=EXP_FOLDER_NAME, logger=logger,
    #                  forw_retr_sel=b)

    evaluate_pipeline(model_names, EXP_FOLDER_NAME, logger)
    matrix_churn(model_names, EXP_FOLDER_NAME, logger, True)
    table_churn(model_names, EXP_FOLDER_NAME, logger, True)

    # evaluate_pipeline(model_names, EXP_FOLDER_NAME, logger, preds_dirname, results_dirname)
    # matrix_churn(model_names, EXP_FOLDER_NAME, logger, True, preds_dirname, results_dirname)
    # table_churn(model_names, EXP_FOLDER_NAME, logger, True, preds_dirname, results_dirname)


    # for b in [True, False]:
    #     table_churn(model_names=model_names,
    #                 exp_folder_name=EXP_FOLDER_NAME, logger=logger, forw_retr_sel=b)
    print("")