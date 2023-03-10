import os
from os.path import join, exists

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
    # root = 'results/day-25-01-2023_hr-15-38-00_epochs-12_batchsize-500_CLEAN_TR'
    root = 'results/day-06-03-2023_hr-17-23-52_epochs-12_batchsize-500_HIGH_AB/advx_ft'
    # delete_non_last_checkpoints(root)
    delete_advx_ts(root)