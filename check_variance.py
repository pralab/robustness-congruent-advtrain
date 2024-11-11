from utils.utils import my_load, join
from utils.eval import compute_nflips
import os
import random
import numpy as np

path = 'results/day-26-09-2024_hr-17-34-46_IMAGENET_FIRST_TRIAL'
clean_fname = f'results_clean_test.gz'
advx_fname = f'results_advx_test.gz'

loss_name_list = ['PCT', 'PCT-AT', 'MixMSE-AT']
hparams = {'PCT': 'a-1.0_b-2.0',
           'PCT-AT': 'a-1.0_b-2.0',
           'MixMSE-AT': 'a-0.5_b-0.4'}

num_reps = 100
num_samples = 5000


loss_name = loss_name_list[1]

for loss_name in loss_name_list:
    print(f"##### {loss_name} ####")

    clean_res_list = []
    advx_res_list = []
    for root_i, dirs, files in os.walk(path):
        if root_i.split('/')[-2] != loss_name:
            continue
        # if root_i.split('/')[-1] != hparams[loss_name]:
        #     continue

        if clean_fname in files:
            clean_res_list.append(join(root_i, clean_fname))
        if advx_fname in files:
            advx_res_list.append(join(root_i, advx_fname))
    
    
    for res_list, rob_level in zip((clean_res_list, advx_res_list), ('clean', 'advx')):
        accs_list, nfs_list = [], []
        print(f">>> {rob_level}")
        for res_fname in res_list:
            model_pairs = res_fname.split('/')[-4]
            hparams = res_fname.split('/')[-2]
            old_correct = my_load(res_fname)['old_correct'].cpu().numpy()
            new_correct = my_load(res_fname)['new_correct'].cpu().numpy()
            
            accs, nfs = [], []
            for i in range(num_reps):
                bootstrap_idx = np.array(random.choices(np.arange(num_samples), k=num_samples))
                bs_old_correct = old_correct[bootstrap_idx]
                bs_new_correct = new_correct[bootstrap_idx]

                acc = bs_new_correct.mean()*100
                nf = compute_nflips(bs_old_correct, bs_new_correct)*100

                accs.append(acc)
                nfs.append(nf)
            
            accs, nfs = np.array(accs), np.array(nfs)

            accs_list.append(accs.std())
            nfs_list.append(nfs.std())

            print(f"({model_pairs})({hparams}) Acc: {accs.mean():.4f} ({accs.std():.4f}) / NFs: {nfs.mean():.4f} ({nfs.std():.4f})")
    
        accs_list = np.array(accs_list)
        nfs_list = np.array(nfs_list)
        print(f"> MEAN OF STD -> Acc: {accs_list.mean():.4f} / NFs: {nfs_list.mean():.4f}")
    print("")


print("")
