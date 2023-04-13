import os
import pandas as pd
import numpy as np
from utils.utils import MODEL_NAMES, join
from utils.eval import compute_common_nflips
from utils.data import sort_df
import pickle
import math


dict_loss_names = {'PCT-AT': '\pctat',
                   'PCT': '\pct',
                   'MixMSE-AT': '\mixmseat',
                   'MixMSE': '\mixmse',}

pd.set_option('display.max_columns', None)


def table_new(model_ids=None, old_model_ids=None, loss_names=None):
    path = 'results/day-30-03-2023_hr-10-01-01_PIPELINE_50k_3models'
    
    # ds_name = 'val'

    for ds_name in ['test', 'val']:
        print(f"Dataset: {ds_name}")
        rows = []
        
        if (model_ids is not None) and (old_model_ids is not None):
            model_pair_dirs = [f"old-{old_id}_new-{new_id}" for (old_id, new_id) in zip(old_model_ids, model_ids)]
        else:
            model_pair_dirs = list(os.walk(path))[0][1]
        

        columns = ['Models ID', 'Loss', 'Hparams', 'Acc', 'RobAcc', 'NFR', 'R-NFR', 'B-NFR', 'S-NFR']
        
        """
        Per il validation posso mettere altre colonne delle performance corrispondenti al validation
        e poi usare la stessa funz sort_df però indicando colonne del validation, così da fare poi lo slicing necessario
        sul test set
        """
        
        model_results_df_list = []
        keys = []
        for model_pair_dir in model_pair_dirs:
            rows_ft = []
            model_pair_path = os.path.join(path, model_pair_dir)
            
            if loss_names is None:
                loss_names = list(os.walk(model_pair_path))[0][1]
                
            for loss_dir in loss_names:
                loss_path = os.path.join(model_pair_path, loss_dir)
                
                for hparams_dir in list(os.walk(loss_path))[0][1]:
                    hparams_path = os.path.join(loss_path, hparams_dir)

                    # with open(os.path.join(hparams_path, 'results_val.gz'), 'rb') as f:
                    #     results_val = pickle.load(f)
                    
                    try:
                        with open(os.path.join(hparams_path, f"results_{ds_name}.gz"), 'rb') as f:
                            results_test = pickle.load(f)
                            

                        acc0, acc1, acc = results_test['clean']['old_acc'], results_test['clean']['orig_acc'], results_test['clean']['new_acc']
                        nfr1, nfr = results_test['clean']['orig_nfr'], results_test['clean']['nfr']
                        rob_acc0, rob_acc1, rob_acc = results_test['advx']['old_acc'], results_test['advx']['orig_acc'], results_test['advx']['new_acc']
                        rob_nfr1, rob_nfr = results_test['advx']['orig_nfr'], results_test['advx']['nfr']
                        
                        # common_nfr1 = 
                        # sum_nfr1 = 
                        # todo: calcolare common_nfr1!!! accrocchio da risolvere
                        _, _, common_nfr = compute_common_nflips(results_test['clean']['nf_idxs'], results_test['advx']['nf_idxs'])
                        sum_nfr = nfr + rob_nfr - common_nfr
                        sum_nfr1 = nfr1 + rob_nfr1

                        # rows_new.append([f"M{Mnew} + {loss_dir}", hparams_dir, acc, rob_acc, nfr, rob_nfr, common_nfr, sum_nfr])
                        rows_ft.append([loss_dir, hparams_dir, 
                                        acc, rob_acc, nfr, 
                                        rob_nfr, common_nfr, sum_nfr])
                    except Exception as e:
                        print(f"{model_pair_dir}/{loss_dir}/{hparams_dir}")
                        rows_ft.append([loss_dir, hparams_dir, 
                                        math.nan, math.nan, math.nan, 
                                        math.nan, math.nan, math.nan])
                        continue
            
            # rows_model_i = [[f"M{Mold}", math.nan, acc0, rob_acc0, math.nan, math.nan, math.nan, math.nan],
            #                 [f"M{Mnew}", math.nan, acc1, rob_acc1, nfr1, rob_nfr1, math.nan, sum_nfr1]]
            rows_old_new = [['old', math.nan, acc0, rob_acc0, math.nan, math.nan, math.nan, math.nan],
                            ['new', math.nan, acc1, rob_acc1, nfr1, rob_nfr1, math.nan, sum_nfr1]]
            model_base_results_df = pd.DataFrame(data=rows_old_new, columns=columns[1:])
            model_ft_results_df = pd.DataFrame(data=rows_ft, columns=columns[1:])
            model_ft_results_df = model_ft_results_df.sort_values(by='S-NFR', ascending=True).drop_duplicates(['Loss']).sort_index()
            model_results_df = pd.concat([model_base_results_df, model_ft_results_df])
            # model_results_df.reset_index(inplace=True, drop=True)
            model_results_df.set_index('Loss', inplace=True) 
            model_results_df[columns[3:]] *= 100
            model_results_df.drop(['Hparams'], axis=1, inplace=True)
            Mold = model_pair_dir.split('old-')[-1].split('_new')[0]
            Mnew = model_pair_dir.split('new-')[-1]
            
            # keys.append(f"M{Mold}-{Mnew}")
            keys.append(model_pair_dir)
            model_results_df_list.append(model_results_df)
            
            
            # rows.extend(rows_old_new + rows_ft)
            
        model_results_df_list = pd.concat(model_results_df_list,
                                        keys=keys)
        # model_results_df_list.to_csv('results/all_results_table.csv')
        latex_table(model_results_df_list, dir_out=path, fname=f'model_results_{ds_name}')
    

        print("")
            
        
    

def latex_table(df, diff=False, perc=False, dir_out='latex_files', fname='models_results'):
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

    keys = np.unique(list(zip(*df.index.to_list()))[0])
    keys = [key.replace('old-', 'M').replace('_new-', '-M') for key in keys]

    for model_pair, idxs, idxs_at, idxs_best, key in zip(model_pairs,
                                                         idxs_list,
                                                         idxs_at_list,
                                                         idxs_best_list,
                                                         keys):

        df_m = df.loc[model_pair]
        for col in df_m.columns:
            try:
                # value = df_m.loc[idxs.loc[col]][col]
                # df_m.loc[idxs.loc[col]][col] = r"\textcolor{blue}{" + value + r"}"
                #
                # value = df_m.loc[idxs_at.loc[col]][col]
                # df_m.loc[idxs_at.loc[col]][col] = r"\textcolor{red}{" + value + r"}"

                value = df.loc[model_pair].loc[idxs_best.loc[col]][col]
                df.loc[model_pair].loc[idxs_best.loc[col]][col] = r"\textbf{" + value + r"}"
            except:
                print("")

        new_index_name = key
        # new_index_name =r"\rotatebox[origin=t]{90}{" + new_index_name + r"}"
        new_index_name = r"\hline \multirow{6}{*}{" + new_index_name + "}"
        # new_index_name = r"\hline \multirow{6}{*}{" + dict_loss_names[new_index_name] + "}"
        # new_index_name = model_pair
        df = df.rename(index={model_pair: new_index_name})
        print("")

    # df.rename(index={k:v in })
    # df = df.transpose()

    df_str = df.to_latex(
            caption="Models results", label="tab:ft_results",
            column_format="l|l|c c|c c c|c|", escape=False
        )
    df_str = df_str.replace(r'\begin{tabular}', r'\resizebox{0.99\linewidth}{!}{\begin{tabular}')
    df_str = df_str.replace(r'\end{tabular}', r'\hline \end{tabular}}')
    
    for k,v in dict_loss_names.items():
        df_str = df_str.replace(k, v)
    
    eof = ""
    if diff:
        eof = "_diff"
    if perc:
        eof = "_perc"
    with open(join(dir_out, f'{fname}{eof}.tex'), 'w') as f:
        f.write(df_str)

    print("")


if __name__ == '__main__':
    old_model_ids=[1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 5, 5, 6]
    model_ids=    [4, 7, 4, 5, 7, 2, 4, 5, 6, 7, 7, 4, 7, 7]
    loss_names = ['PCT', 'PCT-AT', 'MixMSE-AT']
    
    table_new(old_model_ids=old_model_ids, model_ids=model_ids, loss_names=loss_names)
    
    # path="results/day-25-01-2023_hr-15-38-00_CLEAN_TR_backup"
    
    # with open(os.path.join(path, "all_nf_idxs.pkl"), 'rb') as f:
    #     nf_idxs = pickle.load(f)
    
    # print("")
    
    # table_model_results()
