import os
from os.path import join, exists

import pickle

def refactor_results(old_model_ids, model_ids):

    ds_name = 'test'

    model_pair_dirs = [f"old-{old_id}_new-{new_id}" for (old_id, new_id) in zip(old_model_ids, model_ids)]
    
    
    for model_pair_dir in model_pair_dirs:
        adv_loss_path = f"results/day-25-01-2023_hr-15-38-00_CLEAN_TR_backup/advx_ft/{model_pair_dir}/PCT"
        
        loss_path = f"results/day-30-03-2023_hr-10-01-01_PIPELINE_50k_3models/{model_pair_dir}/PCT"
        
        params_dir_list = list(os.walk(loss_path))[0][1]
        for params_dir in params_dir_list:
            params_dir_path = os.path.join(loss_path, params_dir)
            adv_params_dir_path = os.path.join(adv_loss_path, params_dir)
            print(params_dir_path)
            results = {}
            with open(os.path.join(params_dir_path, 'results_last.gz'), 'rb') as f:
                results['clean'] = pickle.load(f)
            
        
            with open(os.path.join(adv_params_dir_path, 'results_last.gz'), 'rb') as f:
                results['advx'] = pickle.load(f)

            with open(os.path.join(params_dir_path, f"results_{ds_name}.gz"), 'wb') as f:
                pickle.dump(results, f)

            with open(os.path.join(params_dir_path, f"results_{ds_name}.gz"), 'rb') as f:
                results_test_check = pickle.load(f)
                
            with open(os.path.join(params_dir_path, f"{ds_name}_perf.txt"), 'w') as f:
                f.write(f">>> Clean Results\n")
                f.write(f"Old Acc: {results['clean']['old_acc']}\n")
                f.write(f"New Acc: {results['clean']['orig_acc']}, New Acc(FT): {results['clean']['new_acc']}\n")
                f.write(f"New NFR: {results['clean']['orig_nfr']}, New NFR(FT): {results['clean']['nfr']}\n")
                f.write(f">>> Advx Results\n")
                f.write(f"Old Acc: {results['advx']['old_acc']}\n")
                f.write(f"New Acc: {results['advx']['orig_acc']}, New Acc(FT): {results['advx']['new_acc']}\n")
                f.write(f"New NFR: {results['advx']['orig_nfr']}, New NFR(FT): {results['advx']['nfr']}\n")



def delete_non_last_checkpoints(root):
    # root = 'results/day-25-01-2023_hr-15-38-00_epochs-12_batchsize-500_CLEAN_TR'

    paths = []
    for root_i, dirs, files in os.walk(root):
        if 'checkpoints' in root_i:
            paths.append(root_i)

    for i, path in enumerate(paths):
        print(f"{i}/{len(paths)}")
        nfr_path = join(path, 'best_nfr.pt')
        acc_path = join(path, 'best_acc.pt')
        last_path = join(path, 'last.pt')

        if exists(acc_path):
            os.remove(acc_path)

        if exists(nfr_path):
            os.remove(nfr_path)

def delete_advx_ts(root):

    paths = []
    for root_i, dirs, files in os.walk(root):
        if 'ts' in dirs:
            paths.append(os.path.join(root_i, 'ts'))

    for i, path in enumerate(paths):
        print(f"{i}/{len(paths)}")
        # print(path)

        for root_i, dirs, files in os.walk(path):
            if len(files) > 0:
                for file in files:
                    os.remove(os.path.join(root_i, file))


if __name__ == '__main__':
    # # root = 'results/day-25-01-2023_hr-15-38-00_epochs-12_batchsize-500_CLEAN_TR'
    # root = 'results/day-06-03-2023_hr-17-23-52_epochs-12_batchsize-500_HIGH_AB/advx_ft'
    # # delete_non_last_checkpoints(root)
    # delete_advx_ts(root)
    
    old_model_ids = [1,1,2,2]
    model_ids = [4,7,4,5]
    
    # old_model_ids = [1]
    # model_ids = [4]
    
    refactor_results(old_model_ids, model_ids)