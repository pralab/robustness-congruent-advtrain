import os
from os.path import join, exists

root = 'results/day-25-11-2022_hr-17-09-48_epochs-12_batchsize-500_TEMPORAL_CLEAN_TR'

paths = []
for root, dirs, files in os.walk(root):
    if 'checkpoints' in root:
        paths.append(root)

for i, path in enumerate(paths):
    print(f"{i}/{len(paths)}")
    nfr_path = join(path, 'best_nfr.pt')
    acc_path = join(path, 'best_acc.pt')
    last_path = join(path, 'last.pt')

    if exists(acc_path):
        if exists(last_path):
            os.remove(last_path)

    if exists(nfr_path):
        if exists(acc_path):
            os.remove(acc_path)