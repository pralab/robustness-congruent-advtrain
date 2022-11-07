import os
from os.path import join, isdir
import pickle as pkl
import pandas as pd
import numpy as np
from utils.utils import MODEL_NAMES
from utils.visualization import plot_loss
import matplotlib.pyplot as plt

def performance_csv(root):
    column_names = ['Acc0', 'Acc1', 'NFR1', 'PFR1', 
                    'Acc(FT)', 'NFR(FT)', 'PFR(FT)']
    index_name = 'Hparams'

    models_dict = {}
    # for model_pair_dir in os.listdir(root):
    for model_pair_dir in ['old-3_new-4', 'old-4_new-5', 'old-5_new-6', 'old-6_new-7']:
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
                            try:                           
                                with open(join(params_path, 'results_best_nfr.gz'), 'rb') as f:
                                    results = pkl.load(f)
                            except:
                                with open(join(params_path, 'results_last.gz'), 'rb') as f:
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


def plot_all_loss(root):

    model_dirs = ['old-3_new-4', 'old-4_new-5', 'old-5_new-6', 'old-6_new-7']
    loss_dirs = ['PCT', 'MixMSE', 'MixMSE(NF)']
    params_dirs = ['a-1_b-1', 'a-1_b-2', 'a-1_b-5', 'a-1_b-10', 'a-1_b-100']

    n_plot_x = len(loss_dirs)
    n_plot_y = len(params_dirs)

    
    for i, model_pair_dir in enumerate(model_dirs):
        model_pair_path = join(root, model_pair_dir)

        fig, ax = plt.subplots(n_plot_x, n_plot_y, figsize=(5*n_plot_y, 5*n_plot_x), squeeze=False)      
        for j, loss_exp_dir in enumerate(loss_dirs):
            ax[j, 0].set_ylabel(loss_exp_dir)

            loss_exp_path = join(model_pair_path, loss_exp_dir)
            for k, params_dir in enumerate(params_dirs):
                params_path = join(loss_exp_path, params_dir)                
                params_name = params_dir.replace('-', '=').replace('_', ',')
                if j==0:
                    ax[0, k].set_title(params_name)                
                with open(join(params_path, 'results_last.gz'), 'rb') as f:
                    results = pkl.load(f)

                loss = results['loss']
                plot_loss(loss, ax[j, k], window=None)
        fig.tight_layout()
        fig.savefig(join(root, f"{model_pair_dir}.pdf"))
        print("")



if __name__ == '__main__':

    root = 'results/day-04-11-2022_hr-16-50-24_epochs-12_batchsize-500'
    performance_csv(root)
    plot_all_loss(root)







                            
                            
        

