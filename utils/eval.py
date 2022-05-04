import torch
import math
import pandas as pd

def predict(model, x, batch_size, device):
    preds = []
    n_examples = x.shape[0]
    x.to(device)
    model.to(device)
    with torch.no_grad():
        for batch_i in range(math.ceil(n_examples / batch_size)):
            start_i = batch_i * batch_size
            end_i = start_i + batch_size

            out = model(x[start_i:end_i])
            pred = torch.argmax(out, axis=1)
            preds.extend(pred.tolist())
    return preds

def compute_nflips(old_preds, new_preds, y):
    df = pd.concat({'y': y,
                    'old': old_preds,
                    'new': new_preds}, axis=1)
    nf_idxs = (old_preds != new_preds) & (old_preds == y)

    return nf_idxs.mean()


if __name__ == "__main__":
    import pandas as pd
    from secml.utils import fm

    MODEL_NAMES = ['Kang2021Stable',
                   'Rebuffi2021Fixing_70_16_cutmix_extra',
                   'Gowal2021Improving_70_16_ddpm_100m']

    ROOT = 'data'
    exp_folder_name = fm.join(ROOT, 'exp_prova2')
    advx_folder = fm.join(exp_folder_name, 'advx')
    predictions_folder = fm.join(exp_folder_name, 'predictions')

    df_old = pd.read_csv(fm.join(predictions_folder, f"{MODEL_NAMES[0]}_predictions.csv"), index_col=0)
    df_new = pd.read_csv(fm.join(predictions_folder, f"{MODEL_NAMES[1]}_predictions.csv"), index_col=0)




    print("")