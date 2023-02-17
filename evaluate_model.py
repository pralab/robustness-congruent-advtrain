import os
from os.path import isdir
import pickle as pkl
import pandas as pd
import numpy as np
from utils.utils import MODEL_NAMES, join
from utils.eval import compute_nflips, compute_common_nflips
from utils.visualization import plot_loss
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from itertools import product
import pickle
import math
from pylatex import LongTable, MultiColumn
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['mathtext.fontset'] = 'stix'


pd.set_option('display.max_columns', None)

def main_plot_results_over_time():

    root = 'results/day-25-11-2022_hr-17-09-48_epochs-12_batchsize-500_TEMPORAL_ADV_TR'
    root_clean = 'results/day-25-11-2022_hr-17-09-48_epochs-12_batchsize-500_TEMPORAL_CLEAN_TR'
    root_advx = f"{root_clean}/advx_ft"
    root_clean_AT = root
    root_advx_AT = f"{root_clean_AT}/advx_ft"

    b = None
    # root = 'results/day-04-11-2022_hr-16-50-24_epochs-12_batchsize-500/advx_AT'
    # root = 'results/day-04-11-2022_hr-16-50-24_epochs-12_batchsize-500'

    # for root_i in [root_clean, root_advx, root_clean_AT, root_advx_AT]:
    #     performance_csv(root_i)

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    plot_results_over_time(root_clean, ax=ax, row=0, adv_tr=False, b=b)
    plot_results_over_time(root_clean_AT, ax=ax, row=0, adv_tr=True, b=b)
    plot_results_over_time(root_advx, ax=ax, row=1, adv_tr=False, b=b)
    plot_results_over_time(root_advx_AT, ax=ax, row=1, adv_tr=True, b=b)

    ax[0, 0].set_ylabel('Clean data')
    ax[1, 0].set_ylabel('Adversarial data')
    ax[-1, -1].legend()
    fig.tight_layout()
    fig.savefig(join(root, 'perf_AT.pdf'))
    fig.show()

    # df = pd.read_csv(join(root, 'all_models_results.csv'))
    #
    # df_list = []
    #
    # for d_model in df.groupby(by='Models ID'):
    #     d_model = d_model[1].reindex(np.hstack(
    #         (d_model[1].index.values[-5:],
    #          d_model[1].index.values[:-5])))
    #     # d_model = d_model.reset_index(drop=True)
    #     for loss in ['PCT', 'MixMSE', 'MixMSE(NF)']:
    #         d_loss = d_model[d_model['Loss'] == loss]
    #         idxs = d_loss['Hparams']
    #         bs = []
    #         for i, idx in enumerate(idxs):
    #             b = int(idx.split('b=')[1])
    #             bs.append(b)
    #         bs = np.array(bs)
    #         d_loss = d_loss.reset_index().drop('index', axis=1)
    #         d_loss = d_loss.reindex(bs.argsort())
    #         d_loss = d_loss.reset_index().drop('index', axis=1)
    #         df_list.append(d_loss)
    #
    #
    # df = pd.concat(df_list)
    # df = df.reset_index().drop('index', axis=1)
    # df.to_csv(join(root, 'all_models_results2.csv'))
    print("")


def main_perf_csv():
    root_clean = 'results/day-04-11-2022_hr-16-50-24_epochs-12_batchsize-500'
    root_advx = f"{root_clean}/advx_ft"
    root_clean_AT = 'results/day-16-11-2022_hr-15-14-52_epochs-12_batchsize-500_AT'
    root_advx_AT = f"{root_clean_AT}/advx_ft"

    for root in (root_clean, root_advx, root_clean_AT, root_advx_AT):
        performance_csv(root)
    print("")


def performance_csv(root, fname="all_models_results"):
    column_names = ['Acc0', 'Acc1', 'NFR1', 'PFR1',
                    'Acc(FT)', 'NFR(FT)', 'PFR(FT)']
                    
                    # 'val-Acc(FT)', 'val-NFR(FT)', 'val-PFR(FT)']
    index_name = 'Hparams'

    nf_idxs = {}

    models_dict = {}
    for model_pair_dir in os.listdir(root):
        if model_pair_dir.startswith('old'):
            # for model_pair_dir in ['old-3_new-4', 'old-4_new-5', 'old-5_new-6', 'old-6_new-7']:
            model_pair_path = join(root, model_pair_dir)            
            nf_idxs[model_pair_dir] = {}
            loss_dict = {}
            if isdir(model_pair_path):
                for loss_exp_dir in ['PCT', 'MixMSE', 'MixMSE(NF)']:
                    loss_exp_path = join(model_pair_path, loss_exp_dir)
                    nf_idxs[model_pair_dir][loss_exp_dir] = {}
                    if isdir(loss_exp_path):
                        i = 0

                        params_df = pd.DataFrame(columns=column_names)
                        params_df.index.name = index_name
                        for params_dir in os.listdir(loss_exp_path):
                            params_path = join(loss_exp_path, params_dir)
                            if isdir(params_path):
                                params_name = params_dir.replace('-', '=').replace('_', ',')                                
                                if loss_exp_dir.startswith('Mix'):
                                    params_name = params_name.split(',')[1]

                                if os.path.exists(join(params_path, 'results_best_nfr.gz')):
                                    with open(join(params_path, 'results_best_nfr.gz'), 'rb') as f:
                                        results = pkl.load(f)
                                        # ckpt = torch.load(join(params_path, 'checkpoints/best_nfr.pt'))
                                        # val_res = ckpt['perf']

                                elif os.path.exists(join(params_path, 'results_best_acc.gz')):
                                    with open(join(params_path, 'results_best_acc.gz'), 'rb') as f:
                                        results = pkl.load(f)
                                        # ckpt = torch.load(join(params_path, 'checkpoints/best_acc.pt'))
                                        # val_res = ckpt['perf']
                                else:
                                    with open(join(params_path, 'results_last.gz'), 'rb') as f:
                                        results = pkl.load(f)
                                        # ckpt = torch.load(join(params_path, 'checkpoints/last.pt'))
                                        # val_res = ckpt['perf']
                                nf_idxs[model_pair_dir][loss_exp_dir][params_name] = results['nf_idxs']

                                acc0 = results['old_acc']
                                acc1 = results['orig_acc']
                                nfr1 = results['orig_nfr']
                                pfr1 = results['orig_pfr']
                                acc = results['new_acc']
                                nfr = results['nfr']
                                pfr = results['pfr']

                                params_df.loc[params_name] = [acc0, acc1, nfr1, pfr1, 
                                                                acc, nfr, pfr]#,
                                                                # val_res['acc'], val_res['nfr'], val_res['pfr']]

                                i += 1

                        idxs = params_df.index
                        bs = []
                        for i, idx in enumerate(idxs):
                            b = int(idx.split('b=')[1])
                            bs.append(b)
                        bs = np.array(bs)
                        params_df = params_df.reindex(list(idxs[bs.argsort()]))
                        loss_dict[loss_exp_dir] = params_df

                model_df = pd.concat([loss_dict[k] for k in loss_dict.keys()], keys=loss_dict)
                model_df = (model_df * 100).round(3)
                models_dict[model_pair_dir] = model_df

    with open(join(root, 'all_nf_idxs.pkl'), 'wb') as f:
        pickle.dump(nf_idxs, f)

    all_models_df = pd.concat([models_dict[k] for k in models_dict.keys()], keys=models_dict)
    all_models_df.index.names = ['Models ID', 'Loss', 'Hparams']
    # all_models_df.sort_index()
    all_models_df.sort_index(inplace=True)
    all_models_df.to_csv(join(root, f"{fname}.csv"))


def plot_all_loss(root):
    model_dirs = ['old-3_new-4', 'old-4_new-5', 'old-5_new-6', 'old-6_new-7']
    loss_dirs = ['PCT', 'MixMSE', 'MixMSE(NF)']
    params_dirs = ['a-1_b-1', 'a-1_b-2', 'a-1_b-5', 'a-1_b-10', 'a-1_b-100']

    n_plot_x = len(loss_dirs)
    n_plot_y = len(params_dirs)

    for i, model_pair_dir in enumerate(model_dirs):
        model_pair_path = join(root, model_pair_dir)

        fig, ax = plt.subplots(n_plot_x, n_plot_y, figsize=(5 * n_plot_y, 5 * n_plot_x), squeeze=False)
        for j, loss_exp_dir in enumerate(loss_dirs):
            ax[j, 0].set_ylabel(loss_exp_dir)

            loss_exp_path = join(model_pair_path, loss_exp_dir)
            for k, params_dir in enumerate(params_dirs):
                params_path = join(loss_exp_path, params_dir)
                params_name = params_dir.replace('-', '=').replace('_', ',')
                if j == 0:
                    ax[0, k].set_title(params_name)
                with open(join(params_path, 'results_last.gz'), 'rb') as f:
                    results = pkl.load(f)

                loss = results['loss']
                plot_loss(loss, ax[j, k], window=None)
        fig.tight_layout()
        fig.savefig(join(root, f"{model_pair_dir}.pdf"))
        print("")


def reorder_csv(root, b=None):
    try:
        acc_df = pd.read_csv(join(root, 'acc.csv'), index_col='Models')
        nfr_df = pd.read_csv(join(root, 'nfr.csv'), index_col='Models')
        pfr_df = pd.read_csv(join(root, 'pfr.csv'), index_col='Models')
    except:
        csv_path = join(root, 'all_models_results.csv')

        if os.name == 'nt':
            # check if windows...
            csv_path = csv_path.replace('\\', '/')
        df = pd.read_csv(csv_path)

        loss_list = df['Loss'].unique()
        models_pair_list = df['Models ID'].sort_values().unique()
        # sort_values -> NFR dal più piccolo al più grande
        # drop_duplicates -> prende la prima occorrenza dei duplicati, NFR più piccolo quindi
        # sort_index -> restore indexes, così ho i modelli in ordine

        if b is None:
            df = df.sort_values(by='Acc(FT)', ascending=False).drop_duplicates(['Models ID', 'Loss', 'NFR(FT)'])
            df = df.sort_values(by='NFR(FT)').drop_duplicates(['Models ID', 'Loss']).sort_index()
        else:
            df = df.loc[df['Hparams'].str.endswith(f"b={b}")]

        df.drop(['Hparams'], axis=1, inplace=True)
        new_cols = ['old', 'new'] + list(loss_list)
        acc_df = pd.DataFrame(columns=new_cols)
        nfr_df = pd.DataFrame(columns=new_cols)
        pfr_df = pd.DataFrame(columns=new_cols)

        acc_df.index.name = 'Models'
        nfr_df.index.name = 'Models'
        pfr_df.index.name = 'Models'

        for model in models_pair_list:
            acc_list = [df[df['Models ID'] == model]['Acc0'].iloc[0],
                        df[df['Models ID'] == model]['Acc1'].iloc[0]]
            nfr_list = [None,
                        df[df['Models ID'] == model]['NFR1'].iloc[0]]
            pfr_list = [None,
                        df[df['Models ID'] == model]['PFR1'].iloc[0]]
            for loss in loss_list:
                acc_list.append(df[df['Models ID'] == model][df['Loss'] == loss]['Acc(FT)'].item())
                nfr_list.append(df[df['Models ID'] == model][df['Loss'] == loss]['NFR(FT)'].item())
                pfr_list.append(df[df['Models ID'] == model][df['Loss'] == loss]['PFR(FT)'].item())

            acc_df.loc[model] = acc_list
            nfr_df.loc[model] = nfr_list
            pfr_df.loc[model] = pfr_list

        acc_df.to_csv(join(root, 'acc.csv'))
        nfr_df.to_csv(join(root, 'nfr.csv'))
        pfr_df.to_csv(join(root, 'pfr.csv'))

    return acc_df, nfr_df, pfr_df

def sort_df(df, b = None):
    if b is None:
        # df = df.sort_values(by='Acc(FT)', ascending=False).drop_duplicates(['Models ID', 'Loss', 'NFR(FT)']+other_cols)
        # df = df.sort_values(by='NFR(FT)').drop_duplicates(['Models ID', 'Loss']+other_cols).sort_index()
        df = df.sort_values(by='NFR(Sum)', ascending=True).drop_duplicates(
            ['Loss', 'AT'])

    else:
        df = df.loc[df['Hparams'].str.endswith(f"b={b}")]

    return df

def plot_results_over_time(root, ax, row, adv_tr=False, b=None):
    df = pd.read_csv(join(root, 'all_models_results.csv'))

    loss_list = df['Loss'].unique()
    models_pair_list = df['Models ID'].sort_values().unique()
    # sort_values -> NFR dal più piccolo al più grande
    # drop_duplicates -> prende la prima occorrenza dei duplicati, NFR più piccolo quindi
    # sort_index -> restore indexes, così ho i modelli in ordine

    df = sort_df(df, b=b)

    df.drop(['Hparams'], axis=1, inplace=True)
    new_cols = ['old', 'new'] + list(loss_list)
    acc_df = pd.DataFrame(columns=new_cols)
    nfr_df = pd.DataFrame(columns=new_cols)
    pfr_df = pd.DataFrame(columns=new_cols)

    acc_df.index.name = 'Models'
    nfr_df.index.name = 'Models'
    pfr_df.index.name = 'Models'

    for model in models_pair_list:
        acc_list = [df[df['Models ID'] == model]['Acc0'].iloc[0],
                    df[df['Models ID'] == model]['Acc1'].iloc[0]]
        nfr_list = [None,
                    df[df['Models ID'] == model]['NFR1'].iloc[0]]
        pfr_list = [None,
                    df[df['Models ID'] == model]['PFR1'].iloc[0]]
        for loss in loss_list:
            acc_list.append(df[df['Models ID'] == model][df['Loss'] == loss]['Acc(FT)'].item())
            nfr_list.append(df[df['Models ID'] == model][df['Loss'] == loss]['NFR(FT)'].item())
            pfr_list.append(df[df['Models ID'] == model][df['Loss'] == loss]['PFR(FT)'].item())

        acc_df.loc[model] = acc_list
        nfr_df.loc[model] = nfr_list
        pfr_df.loc[model] = pfr_list

    acc_df.to_csv(join(root, 'acc.csv'))
    nfr_df.to_csv(join(root, 'nfr.csv'))
    pfr_df.to_csv(join(root, 'pfr.csv'))

    for i, df_i in enumerate([acc_df, nfr_df, pfr_df]):
        # if i == 0:
        #     df_i[['old', 'new']].plot(ax=ax[i], style='o--')
        # else:

        # df_i[['new']].plot(ax=ax[row, i], style='o--')
        # df_i[['PCT', 'MixMSE', 'MixMSE(NF)']]\
        #     .plot(ax=ax[row, i], style='o-', rot=45)

        models_ids = df_i.index.to_numpy()
        new = df_i['new'].to_numpy()
        pct = df_i['PCT'].to_numpy()
        # mixmse = df_i['MixMSE'].to_numpy()[:3]
        # mixmsenf = df_i['MixMSE(NF)'].to_numpy()[:3]

        line = 'solid' if adv_tr else 'dashed'
        alpha = 0.6
        markersize = 6
        linewidth = 1

        if not adv_tr:
            ax[row, i].plot(new, color='gray', marker='o', linestyle='dotted',
                            label='baseline', alpha=alpha,
                            markersize=markersize, linewidth=linewidth)
        ax[row, i].plot(pct, color='green', marker='*', linestyle=line,
                        label='AT-PCT' if adv_tr else 'PCT', alpha=alpha,
                        markersize=markersize, linewidth=linewidth)
        # ax[row, i].plot(mixmse, color='blue', marker='^', linestyle=line,
        #                 label='AT-MixMSE' if adv_tr else 'MixMSE', alpha=alpha,
        #                 markersize=markersize, linewidth=linewidth)
        # ax[row, i].plot(mixmsenf, color='red', marker='v', linestyle=line,
        #                 label='AT-MixMSE(NF)' if adv_tr else 'MixMSE(NF)', alpha=alpha,
        #                 markersize=markersize, linewidth=linewidth)

        ax[row, i].set_xticks(range(len(models_ids)), models_ids, rotation=45)

    titles = ['Accuracy', 'NFR', 'PFR']
    titles = [f"{t} (%)" for t in titles]
    for i in range(3):
        ax[row, i].set_title(titles[i])
        # ax[i].get_xaxis().set_visible(False)
        # ax[i].set_xticks(list(np.arange(acc_df.shape[0])),
        #                  rotation=45)

        # if i == 0:
        #     ax[row, i].set_ylim([0, 95])
        # elif i == 1:
        #     ax[row, i].set_ylim([0, 30])
        # else:
        #     ax[row, i].set_ylim([0, 90])

    # fig.tight_layout()
    # # fig.savefig(join(root, 'perf.pdf'))
    # fig.show()

    print("")

#
# def table_model_results():
#     root_clean = 'results/day-04-11-2022_hr-16-50-24_epochs-12_batchsize-500'
#     root_advx = f"{root_clean}/advx_ft"
#     root_clean_AT = 'results/day-16-11-2022_hr-15-14-52_epochs-12_batchsize-500_AT'
#     root_advx_AT = f"{root_clean_AT}/advx_ft"
#
#     single_model_res_path = join('results/single_models_res')
#     if not os.path.isdir(single_model_res_path):
#         os.mkdir(single_model_res_path)
#
#     df = None
#     df_clean = pd.read_csv(join(root_clean, 'all_models_results.csv'))
#     df_advx = pd.read_csv(join(root_advx, 'all_models_results.csv'))
#     df_clean_at = pd.read_csv(join(root_clean_AT, 'all_models_results.csv'))
#     df_advx_at = pd.read_csv(join(root_advx_AT, 'all_models_results.csv'))
#
#     for i, df_i in enumerate([df_clean, df_advx, df_clean_at, df_advx_at]):
#         df_i['AT'] = True if i > 1 else False
#         df_i['ts_data'] = 'clean' if (i % 2) == 0 else 'advx'
#         df = df_i if df is None else pd.concat([df, df_i])
#         df.reset_index(inplace=True, drop=True)
#
#     df['NFR (Clean+Robust)'] = df[]
#
#     for models_id in df['Models ID'].unique():
#         print(f'{"-"*50}\n{models_id} - new -> {MODEL_NAMES[int(models_id.split("new-")[-1])]}')
#         # models_id = df['Models ID'].unique()[3]
#
#         df_model = df.loc[df['Models ID'] == models_id]
#         df_model.reset_index(inplace=True, drop=True)
#
#         df_model = sort_df(df_model, b=None, other_cols=['AT', 'ts_data'])
#         df_model.drop(['Models ID'], axis=1, inplace=True)
#
#         model_results_df = pd.DataFrame(columns=['Acc', 'Rob Acc', 'NFR', 'Rob NFR'])#, 'Hparams'])
#         model_results_df.index.name = 'model'
#
#         non_ft_df = df_model.drop_duplicates(['ts_data'])
#         model_results_df.loc['old'] = [non_ft_df.loc[non_ft_df['ts_data'] == 'clean']['Acc0'].item(),
#                                        non_ft_df.loc[non_ft_df['ts_data'] == 'advx']['Acc0'].item(),
#                                        None,
#                                        None]
#         model_results_df.loc['new'] = [non_ft_df.loc[non_ft_df['ts_data'] == 'clean']['Acc1'].item(),
#                                        non_ft_df.loc[non_ft_df['ts_data'] == 'advx']['Acc1'].item(),
#                                        non_ft_df.loc[non_ft_df['ts_data'] == 'clean']['NFR1'].item(),
#                                        non_ft_df.loc[non_ft_df['ts_data'] == 'advx']['NFR1'].item()]
#
#         for at in [False, True]:
#             for loss in ['PCT', 'MixMSE', 'MixMSE(NF)']:
#                 loss_df = df_model.loc[(df_model['Loss'] == loss) & (df_model['AT'] == at)]
#                 idx_name = loss if not at else f"{loss}-AT"
#                 model_results_df.loc[idx_name] = [loss_df.loc[(loss_df['ts_data'] == 'clean')]['Acc(FT)'].item(),
#                                                loss_df.loc[(loss_df['ts_data'] == 'advx')]['Acc(FT)'].item(),
#                                                loss_df.loc[(loss_df['ts_data'] == 'clean')]['NFR(FT)'].item(),
#                                                loss_df.loc[(loss_df['ts_data'] == 'advx')]['NFR(FT)'].item()
#                                                ]
#         print(model_results_df)
#         model_results_df.to_csv(join(single_model_res_path, f"{models_id}.csv"))
#
#
#     print("")


def table_model_results(model_sel=(1,3,5,6),
                        losses=('PCT', 'MixMSE', 'MixMSE(NF)'),
                        diff=False, perc=False):
    # 4 Folders, clean/advx and standard/AT
    root_clean = 'results/day-25-01-2023_hr-15-38-00_epochs-12_batchsize-500_CLEAN_TR'
    root_advx = f"{root_clean}/advx_ft"
    root_clean_AT = 'results/day-30-01-2023_hr-10-01-02_epochs-12_batchsize-500_ADV_TR'
    root_advx_AT = f"{root_clean_AT}/advx_ft"

    single_model_res_path = 'results/single_models_res'
    if not os.path.isdir(single_model_res_path):
        os.mkdir(single_model_res_path)

    # Table for standard training
    df_clean = pd.read_csv(join(root_clean, 'all_models_results.csv'))
    df_advx = pd.read_csv(join(root_advx, 'all_models_results.csv'))
    df_clean.drop(['PFR1', 'PFR(FT)'], axis=1, inplace=True)
    df_advx = df_advx[['Acc0', 'Acc1', 'Acc(FT)', 'NFR1', 'NFR(FT)']].rename(
        columns={'Acc0': 'Rob Acc0', 'Acc1': 'Rob Acc1', 'NFR1': 'Rob NFR1',
                 'Acc(FT)': 'Rob Acc(FT)', 'NFR(FT)': 'Rob NFR(FT)'})
    df = pd.concat([df_clean, df_advx], axis=1)

    # Table for AT
    df_clean_at = pd.read_csv(join(root_clean_AT, 'all_models_results.csv'))
    df_advx_at = pd.read_csv(join(root_advx_AT, 'all_models_results.csv'))
    df_clean_at.drop(['PFR1', 'PFR(FT)'], axis=1, inplace=True)
    df_advx_at = df_advx_at[['Acc0', 'Acc1', 'Acc(FT)', 'NFR1', 'NFR(FT)']].rename(
        columns={'Acc0': 'Rob Acc0', 'Acc1': 'Rob Acc1', 'NFR1': 'Rob NFR1',
                 'Acc(FT)': 'Rob Acc(FT)', 'NFR(FT)': 'Rob NFR(FT)'})
    df_at = pd.concat([df_clean_at, df_advx_at], axis=1)

    # Merge the 2 tables
    df['AT'] = False
    df_at['AT'] = True
    df = pd.concat([df, df_at])
    df.reset_index(inplace=True, drop=True)

    # Load and compute common churn between clean and advx data
    nf_idxs = {}
    for root, name in zip((root_clean, root_advx, root_clean_AT, root_advx_AT),
                          ("root_clean", "root_advx", "root_clean_AT", "root_advx_AT")):
        with open(join(root, 'all_nf_idxs.pkl'), 'rb') as f:
            nf_idxs[name] = pickle.load(f)

    common_nfs = []
    for i, r in df.iterrows():
        nf_idxs_clean = nf_idxs['root_clean' if not r['AT'] else 'root_clean_AT']
        nf_idxs_clean = nf_idxs_clean[r['Models ID']][r['Loss']][r['Hparams']]
        nf_idxs_advx = nf_idxs['root_advx' if not r['AT'] else 'root_advx_AT']
        nf_idxs_advx = nf_idxs_advx[r['Models ID']][r['Loss']][r['Hparams']]
        nf_idxs_clean = nf_idxs_clean[:nf_idxs_advx.shape[0]]
        _, _, common_nfr = compute_common_nflips(nf_idxs_clean, nf_idxs_advx)
        # nfr_row = nf_idxs_clean.mean()*100
        common_nfs.append(common_nfr*100)

    df['NFR (Both)'] = common_nfs
    df['NFR(Sum)'] = df['NFR(FT)'] + df['Rob NFR(FT)']

    model_results_df_list = []
    diff_model_res_list = []
    keys = []
    for models_id in df['Models ID'].unique():
        # Check single models as single table
        print(f'{"-"*50}\n{models_id} - new -> {MODEL_NAMES[int(models_id.split("new-")[-1])]}')
        # models_id = df['Models ID'].unique()[3]
        keys.append(models_id)

        df_model = df.loc[df['Models ID'] == models_id]
        df_model.drop(['Models ID'], axis=1, inplace=True)

        df_model = sort_df(df_model, b=None)
        df_model.reset_index(inplace=True, drop=True)

        model_results_df = pd.DataFrame(columns=['Acc', 'Rob Acc',
                                                 'NFR', 'Rob NFR', 'NFR (Both)',
                                                 'NFR (Sum)'])#, 'Hparams'])
        model_results_df.index.name = 'model'

        model_results_df.loc['old'] = [df_model['Acc0'][0],
                                       df_model['Rob Acc0'][0],
                                       None, None, None, None]
        model_results_df.loc['new'] = [df_model['Acc1'][0],
                                       df_model['Rob Acc1'][0],
                                       df_model['NFR1'][0],
                                       df_model['Rob NFR1'][0],
                                       df_model['NFR (Both)'][0],
                                       df_model['NFR1'][0]
                                       + df_model['Rob NFR1'][0]
                                       - df_model['NFR (Both)'][0]]
        for loss in losses:
            for at in [False, True]:
                loss_df = df_model.loc[(df_model['Loss'] == loss) & (df_model['AT'] == at)]
                idx_name = loss if not at else f"{loss}-AT"
                model_results_df.loc[idx_name] = [loss_df['Acc(FT)'].item(),
                                                  loss_df['Rob Acc(FT)'].item(),
                                                  loss_df['NFR(FT)'].item(),
                                                  loss_df['Rob NFR(FT)'].item(),
                                                  loss_df['NFR (Both)'].item(),
                                                  loss_df['NFR(FT)'].item()
                                                  + loss_df['Rob NFR(FT)'].item()
                                                  - loss_df['NFR (Both)'].item()
                                                  ]
                if loss_df['NFR (Both)'].item() > 0:
                    print("")
        print(model_results_df)
        model_results_df_list.append(model_results_df)
        model_results_df.to_csv(join(single_model_res_path, f"{models_id}.csv"),
                                float_format='%.2f')

    # model_results_df_list = pd.concat([model_results_df_list[i] for i in model_sel],
    #                                   keys=[keys[i] for i in model_sel])
    keys = [key.replace('old-', 'M').replace('_new-', ' - M') for key in keys]

    model_results_df_list = pd.concat(model_results_df_list,
                                      keys=keys)
    model_results_df_list.to_csv('results/all_results_table.csv')
    latex_table(model_results_df_list, diff=diff, perc=perc)


def latex_table(df, diff=False, perc=False, fout='latex_files/models_results.tex'):
    model_pairs = np.unique(np.array(list(zip(*df.index))[0])).tolist()

    idxs_best_list = []
    idxs_list = []
    idxs_at_list = []
    for model_pair in model_pairs:
        df_m = df.loc[model_pair]
        idxs = df_m.iloc[1:].idxmax()
        idxs.iloc[2:] = df_m.iloc[1:, 2:].idxmin()
        idxs_best_list.append(idxs)
        for i in range(2):
            sel_loss = [2+i, 4+i, 6+i][:(df_m.index.shape[0] - 2)//2]
            if diff:
                df_m.iloc[sel_loss, :] = df_m.iloc[sel_loss, :] - df_m.iloc[1, :]
            if perc:
                df_m.iloc[sel_loss, :] = (df_m.iloc[sel_loss, :] / df_m.iloc[1, :]).fillna(0)
            idxs = df_m.iloc[sel_loss].idxmax()
            idxs.iloc[2:] = df_m.iloc[sel_loss, 2:].idxmin()
            if i == 0:
                idxs_list.append(idxs)
            else:
                idxs_at_list.append(idxs)

    df = df.applymap(lambda x: f"{x:.2f}" if not math.isnan(x) else "-")

    for model_pair, idxs, idxs_at, idxs_best in zip(model_pairs,
                                                    idxs_list,
                                                    idxs_at_list,
                                                    idxs_best_list):

        df_m = df.loc[model_pair]
        for col in df_m.columns:
            try:
                # value = df_m.loc[idxs.loc[col]][col]
                # df_m.loc[idxs.loc[col]][col] = r"\textcolor{blue}{" + value + r"}"
                #
                # value = df_m.loc[idxs_at.loc[col]][col]
                # df_m.loc[idxs_at.loc[col]][col] = r"\textcolor{red}{" + value + r"}"

                value = df_m.loc[idxs_best.loc[col]][col]
                df_m.loc[idxs_best.loc[col]][col] = r"\textbf{" + value + r"}"
            except:
                print("")


        new_index_name =r"\hline \multirow{6}{*}{\rotatebox[origin=c]{90}{" + model_pair.replace('_', r'\_') + r"}}"
        df = df.rename(index={model_pair: new_index_name})

    # df.rename(index={k:v in })

    df_str = df.to_latex(
            caption="Models results", label="tab:ft_results",
            column_format="l|l|c c|c c c|c|", escape=False
        )
    df_str = df_str.replace(r'\begin{tabular}', r'\resizebox{0.99\linewidth}{!}{\begin{tabular}')
    df_str = df_str.replace(r'\end{tabular}', r'\hline \end{tabular}}')
    eof = ""
    if diff:
        eof = "_diff"
    if perc:
        eof = "_perc"
    with open(f'latex_files/models_results{eof}.tex', 'w') as f:
        f.write(df_str)

    print("")


def plot_histogram(path='results/single_models_res'):
    sel = ['Acc', 'Rob Acc', 'NFR', 'Rob NFR', 'NFR (Sum)']
    # sel = ['NFR', 'Rob NFR']

    model_list = []
    keys = []
    for file in os.listdir(path):
        model_results_df = pd.read_csv(path + f"/{file}", index_col='model')
        # model_results_df.iloc[2:] = model_results_df.iloc[2:] - model_results_df.iloc[1]
        model_results_df = model_results_df.drop('old')#[sel]
        # model_results_df['model'] = model_results_df.index
        model_list.append(model_results_df)
        keys.append(file)

    model_list, keys = model_list[1:], keys[1:]
    all_models_df = pd.concat(model_list, keys=keys)
    max = all_models_df.max().max() + 5
    min = all_models_df.min().min() - 5

    n_rows, n_cols = 2, len(model_list)
    size = 4
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(size*n_cols, size*n_rows), squeeze=False)

    for i in range(n_rows):
        for j in range(n_cols):
            # idx = n_cols * i + j
            model_df = model_list[j][sel[:2] if i == 0 else sel[2:]]
            model_df.rename(
                columns={k : v for k, v in zip(model_df.columns, ['Clean', 'Robust', 'Overall'])},
                inplace=True)
            model_df.transpose().plot.bar(ax=ax[i, j], rot=0,
                                          legend=False)


            # ax[i, j].set_ylim([min, max])
            # ax[i, j].axhline(y=0, color='k', linestyle='--')
    for j in range(n_cols):
        title = keys[j]
        title = title.replace('.csv', '').replace(
            'old-', 'old: M').replace(
            '_new-', ', new: M')
        ax[0, j].set_title(title)
    ax[0, 0].set_ylabel('Accuracy (%)')
    ax[1, 0].set_ylabel('NFR (%)')

    fig.tight_layout()
    fig.show()

    # create legend
    h, l = ax[0, 0].get_legend_handles_labels()
    legend_dict = dict(zip(l, h))
    legend_fig = plt.figure(figsize=(10, 0.5))

    legend_fig.legend(legend_dict.values(), legend_dict.keys(), loc='upper left',
                      ncol=len(legend_dict.values()), frameon=False)
    legend_fig.tight_layout()
    legend_fig.show()

    fig.savefig('images/ftuning_plots/ftuning_results.pdf')
    legend_fig.savefig('images/ftuning_plots/legend_ftuning_results.pdf')

    print("")


def evaluate_ensemble():
    result_path = 'results'
    old_ensemble_id = [1,2,3]
    new_ensemble_id = [4,5,6]

    def get_ensemble_preds(ensemble_ids):
        ensemble_preds = []
        for model_id in ensemble_ids:
            model_name = MODEL_NAMES[model_id]
            path = join(result_path, 'advx', model_name, 'correct_preds.gz')
            with open(path, 'rb') as f:
                preds = pkl.load(f)
            ensemble_preds.append(preds.tolist())

        ensemble_preds = np.array(ensemble_preds)
        ensemble_preds = (ensemble_preds.mean(axis=0)>0.5)
        return ensemble_preds
    
    old_ensemble_preds = get_ensemble_preds(old_ensemble_id)
    new_ensemble_preds = get_ensemble_preds(new_ensemble_id)

    old_ensemble_acc = old_ensemble_preds.mean()
    new_ensemble_acc = new_ensemble_preds.mean()

    ensemble_nfr = compute_nflips(old_ensemble_preds, new_ensemble_preds)

    print(f"Old Acc: {old_ensemble_acc}")
    print(f"New Acc: {new_ensemble_acc}")
    print(f"NFR: {ensemble_nfr}")



    print("")





if __name__ == '__main__':

    # # model_sel = tuple(range(1, 7))
    model_sel = (1, 3, 5, 6)
    
    losses = ('PCT',)#, 'MixMSE(NF)')
    table_model_results(model_sel=model_sel, losses=losses, diff=False)
    # # table_model_results(model_sel=model_sel, losses=losses, diff=True)
    # # table_model_results(model_sel=model_sel, losses=losses, diff=False, perc=True)
    # # plot_histogram()














