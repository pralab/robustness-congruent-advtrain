import os
from os.path import join, isdir
import pickle as pkl
import pandas as pd
import numpy as np

if __name__ == '__main__':

    # root = 'results/day-04-11-2022_hr-12-32-47_prova_3-6-9_5ktr_2kts'
    root = 'results2'

    column_names = ['Acc0', 'Acc1', 'NFR1', 'PFR1', 
                    'Acc(FT)', 'NFR(FT)', 'PFR(FT)']
    index_name = 'Hparams'

    models_dict = {}
    for model_pair_dir in os.listdir(root):
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
                            with open(join(params_path, 'results.gz'), 'rb') as f:
                                results = pkl.load(f)
                            print("")
                            acc0 = results['old_acc']
                            acc1 = results['orig_acc']
                            nfr1 = results['orig_nfr']
                            pfr1 = results['orig_pfr']
                            acc = results['new_acc']
                            nfr = results['nfr']
                            pfr = results['pfr']

                            params_df.loc[params_name] = [acc0, acc1, nfr1, pfr1, acc, nfr, pfr]
                            
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
            model_df = (model_df*100).round(3)
            models_dict[model_pair_dir] = model_df

    all_models_df = pd.concat([models_dict[k] for k in models_dict.keys()], keys=models_dict)
    all_models_df.index.names = ['Models ID', 'Loss', 'Hparams']
    all_models_df.to_csv(join(root, f"all_models_results.csv"))

    single_model_df = None


                            
                            
        

