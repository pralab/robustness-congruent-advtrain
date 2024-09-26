
from torch.utils.data import DataLoader
import torch
import numpy as np
import utils.utils as ut
from utils.data import get_imagenet_dataset
from utils.eval import correct_predictions, compute_nflips

from robustbench import load_model

MODEL_NAMES = {}
MODEL_NAMES['imagenet'] = [
'Salman2020Do_R18',                 # 52.95 / 25.32 (R18)
'Engstrom2019Robustness',           # 62.56 / 29.22 (R50)
'Chen2024Data_WRN_50_2',            # 68.76 / 40.60 (WR50-2)
'Liu2023Comprehensive_ConvNeXt-B',  # 76.02 / 55.82 (ConvNeXt-B)
'Liu2023Comprehensive_Swin-L',      # 78.92 / 59.56 (Swin-L)
]

CLEAN_ACCS = [52.95, 62.56, 68.76, 76.02, 78.92]

# MODEL_NAMES['imagenet'] = [
# 'Liu2023Comprehensive_Swin-L',      # 78.92 / 59.56
# # 'Liu2023Comprehensive_ConvNeXt-L',  # 78.02 / 58.48
# # 'Liu2023Comprehensive_Swin-B',      # 76.16 / 56.16
# 'Liu2023Comprehensive_ConvNeXt-B',  # 76.02 / 55.82
# 'Chen2024Data_WRN_50_2',            # 68.76 / 40.60 (WR50-2)
# # 'Salman2020Do_R50',                 # 64.02 / 34.96 (R50)
# 'Engstrom2019Robustness',           # 62.56 / 29.22 (R50)
# # 'Wong2020Fast',                     # 55.62 / 26.24 (R50)
# 'Salman2020Do_R18',                 # 52.95 / 25.32 (R18)
# ]



if __name__ == '__main__':
    import os

    batch_size = 1024

    device = 'cuda:1'

    n_train_samples = 45000
    train_size = 0.8 # train/validation split, starting from <n_train_samples>
    n_test_samples = 5000

    train_dataset, val_dataset, test_dataset = get_imagenet_dataset(normalize=False)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    model_id = 0

    ds_name = 'val'

    ds_loader = test_loader if ds_name == 'test' else val_loader

    for model_id in range(len(MODEL_NAMES['imagenet'])):
        model_name = ut.MODEL_NAMES[ut.imagenet_id][model_id]
        print(f"> Model: {model_name}")

        clean_root = f'results/clean-imagenet'
        clean_model_path = os.path.join(clean_root, model_name)
        os.makedirs(clean_model_path, exist_ok=True)
        clean_corrects_path = os.path.join(clean_root, model_name, f"correct_preds_{ds_name}.gz")
        
        if os.path.exists(clean_corrects_path):
            clean_preds = ut.my_load(clean_corrects_path)
        else:
            model = load_model(model_name,
                            dataset='imagenet', 
                            threat_model='Linf').to(device)
            clean_preds = correct_predictions(model, ds_loader, device=device)
            ut.my_save(clean_preds, clean_corrects_path)

        clean_acc = clean_preds.cpu().numpy().mean() * 100
        print(f"Clean Accuracy (on {ds_name}): {clean_acc:.2f}")

        if model_id > 0:
            old_model_name = ut.MODEL_NAMES[ut.imagenet_id][model_id - 1]
            old_clean_corrects_path = os.path.join(clean_root, old_model_name, f"correct_preds_{ds_name}.gz")
            old_clean_preds = ut.my_load(old_clean_corrects_path)
            nfs = compute_nflips(old_clean_preds, clean_preds) * 100
            print(f"NFs (on {ds_name}): {nfs:.2f}")


        advx_root = f'results/advx-imagenet'
        advx_model_path = os.path.join(advx_root, model_name)
        advx_corrects_path = os.path.join(advx_root, model_name, f"correct_preds_{ds_name}.gz")
        
        if os.path.exists(advx_corrects_path):
            advx_preds = ut.my_load(advx_corrects_path)
        else:
            raise FileNotFoundError(f"{advx_corrects_path}\n you should first run generate_advx.py to obtain the adversarial examples.")
        rob_acc = advx_preds.cpu().numpy().mean() * 100
        print(f"Robust Accuracy (on {ds_name}): {rob_acc:.2f}")

        if model_id > 0:
            old_model_name = ut.MODEL_NAMES[ut.imagenet_id][model_id - 1]
            old_advx_corrects_path = os.path.join(advx_root, old_model_name, f"correct_preds_{ds_name}.gz")
            old_advx_preds = ut.my_load(old_advx_corrects_path)
            rnfs = compute_nflips(old_advx_preds, advx_preds) * 100
            print(f"RNFs (on {ds_name}): {rnfs:.2f}")

        print("")
    # u, c = np.unique(train_dataset.targets.numpy(), return_counts=True)
    # print(f"# train samples: {u.shape[0]}")
    # print(f"# train samples per class: {c.min()}")

    # u, c = np.unique(validation_dataset.targets.numpy(), return_counts=True)
    # print(f"Num. val classes: {u.shape[0]}")
    # print(f"Val samples per class: {c.min()}")

    # u, c = np.unique(test_dataset.targets.numpy(), return_counts=True)
    # print(f"Num. test classes: {u.shape[0]}")
    # print(f"Test samples per class: {c.min()}")


    print("")
