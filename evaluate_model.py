import os
from os.path import join, isdir
import pickle as pkl
import pandas as pd
import numpy as np
from utils.utils import MODEL_NAMES
from utils.visualization import plot_loss
import matplotlib.pyplot as plt
import torch


def performance_csv(root, fname="all_models_results"):
    column_names = ['Acc0', 'Acc1', 'NFR1', 'PFR1',
                    'Acc(FT)', 'NFR(FT)', 'PFR(FT)']
                    
                    # 'val-Acc(FT)', 'val-NFR(FT)', 'val-PFR(FT)']
    index_name = 'Hparams'

    models_dict = {}
    for model_pair_dir in os.listdir(root):
        if model_pair_dir.startswith('old'):
            # for model_pair_dir in ['old-3_new-4', 'old-4_new-5', 'old-5_new-6', 'old-6_new-7']:
            model_pair_path = join(root, model_pair_dir)

            loss_dict = {}
            if isdir(model_pair_path):
                for loss_exp_dir in ['PCT', 'MixMSE', 'MixMSE(NF)']:
                    loss_exp_path = join(model_pair_path, loss_exp_dir)
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


def plot_results_over_time(root, ax, row, adv_tr=False, b=None):
    df = pd.read_csv(join(root, 'all_models_results.csv'))

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


if __name__ == '__main__':

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














