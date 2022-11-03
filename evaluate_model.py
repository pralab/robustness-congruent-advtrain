import os
from os.path import join, isdir
import pickle as pkl
import pandas as pd

if __name__ == '__main__':

    root = 'results2'

    

    for model_pair_dir in os.listdir(root):
        model_pair_path = join(root, model_pair_dir)

        model_df = pd.DataFrame(columns=
            ['Old Acc', 
            'Acc0', 'NFR0', 'PFR0',
            'Acc1', 'NFR1', 'PFR1',
            'Acc2', 'NFR2', 'PFR2',
            'Acc3', 'NFR3', 'PFR3'])
        model_df.index.name = 'loss_type'
        if isdir(model_pair_path):            
            for loss_exp_dir in os.listdir(model_pair_path):
                loss_exp_path = join(model_pair_path, loss_exp_dir)
                if isdir(loss_exp_path):
                    i = 0

                    perf = []
                    for params_dir in os.listdir(loss_exp_path):
                        params_path = join(loss_exp_path, params_dir)
                        if isdir(params_path):
                            with open(join(params_path, 'results.gz'), 'rb') as f:
                                results = pkl.load(f)
                            print("")
                            acc = results['new_acc']
                            nfr = results['nfr']
                            pfr = results['pfr']

                            perf.extend([acc, nfr, pfr])
                            
                            i += 1
                    with open(join(params_path, 'results.gz'), 'rb') as f:
                        baseline = pkl.load(f)
                    old_acc = baseline['old_acc']
                    acc0 = baseline['new_acc']
                    nfr0 = baseline['nfr']
                    pfr0 = baseline['pfr']

                    perf = [old_acc, acc0, nfr0, pfr0] + perf

                    model_df.loc[loss_exp_dir] = perf
            model_df = model_df*100
            model_df.to_csv(join(model_pair_path, f"{model_pair_dir}.csv"))

                            
                            
        

