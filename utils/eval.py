import torch
import math
import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle
# from utils import model_pairs_str_to_ids, MODEL_NAMES
import utils.utils as ut
import os



# def check_baseline_clean(mid, ds_name, logger, random_seed, ds_id=ut.cifar10_id, sel_advx=False):
#     """
#     If baseline advx already exists for model<mid> load it in correct_adv
#     otherwise compute, save and return
#     """
#     type = 'clean' if not sel_advx else 'advx'
#     root = f'results/{type}' if ds_id==ut.cifar10_id else f'results/{type}-imagenet'
#     correct_adv_fname = os.path.join(root, ut.MODEL_NAMES[ds_id][mid], f"correct_preds_{ds_name}.gz")
#     try:
#         # Load WB advx predictions of Mold
#         with open(correct_adv_fname, 'rb') as f:
#             correct = pickle.load(f)
#     except:
#         logger.debug(f"Baseline {ds_name} advx for M{mid} does not exist. Generating...")
#         ut.set_all_seed(random_seed)
#         generate_baseline_advx(mid, ds_name=ds_name)
#         with open(correct_adv_fname, 'rb') as f:
#             correct_adv = pickle.load(f)
#     return correct

def get_pct_results(new_model, ds_loader, old_correct=None, old_model=None, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available()
                else "cpu")
    
    if old_correct is None:
        old_correct = correct_predictions(old_model, ds_loader, device)
    new_correct = correct_predictions(new_model, ds_loader, device)

    n_min = min(old_correct.shape[0], new_correct.shape[0])
    old_correct = old_correct[:n_min]
    new_correct = new_correct[:n_min]

    old_acc = old_correct.cpu().numpy().mean()
    nf_idxs = compute_nflips(old_correct, new_correct, indexes=True)
    pf_idxs = compute_pflips(old_correct, new_correct, indexes=True)
    new_acc = new_correct.cpu().numpy().mean()
    diff_acc = new_acc - old_acc
    pfr = pf_idxs.mean()
    nfr = nf_idxs.mean()

    results = {'old_correct': old_correct.cpu(),
                'new_correct': new_correct.cpu(),
                'old_acc': old_acc,
                'new_acc': new_acc,
                'diff_acc': diff_acc,
                'pfr': pfr,
                'nfr': nfr,
                'nf_idxs': nf_idxs,
                'pf_idxs': pf_idxs
                }

    return results


def correct_predictions(model, test_loader, device):
    model = model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as t:
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = torch.argmax(output, dim=1)  # get the index of the max log-probability
                preds.append(pred==target)
                t.set_postfix(
                    completed='[{}/{} ({:.0f}%)]'.format(
                        batch_idx * len(data),
                        len(test_loader.dataset),
                        100. * batch_idx / len(test_loader)))
                t.update()
    preds = torch.cat(preds)
    return preds


def get_ds_outputs(model, ds_loader, device):
    model = model.to(device)
    model.eval()
    outputs = []
    with torch.no_grad():
        with tqdm(total=len(ds_loader)) as t:
            for batch_idx, (data, target) in enumerate(ds_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                outputs.append(output)
                t.set_postfix(
                    completed='[{}/{} ({:.0f}%)]'.format(
                        batch_idx * len(data),
                        len(ds_loader.dataset),
                        100. * batch_idx / len(ds_loader)))
                t.update()
    outputs = torch.cat(outputs)
    return outputs


# def compute_nflips(old_preds, new_preds, y):
#     nf_idxs = (old_preds != new_preds) & (old_preds == y)
#     return nf_idxs.mean()


def compute_nflips(old_preds, new_preds, indexes=False):
    if not isinstance(old_preds, np.ndarray):
        old_preds = old_preds.cpu().tolist()
        new_preds = new_preds.cpu().tolist()
    old_preds = pd.Series(old_preds)
    new_preds = pd.Series(new_preds)
    nf_idxs = (old_preds & (~new_preds))
    return nf_idxs if indexes else nf_idxs.mean()


def compute_pflips(old_preds, new_preds, indexes=False):
    if not isinstance(old_preds, np.ndarray):
        old_preds = old_preds.cpu().tolist()
        new_preds = new_preds.cpu().tolist()
    old_preds = pd.Series(old_preds)
    new_preds = pd.Series(new_preds)
    pf_idxs = ((~old_preds) & new_preds)
    return pf_idxs if indexes else pf_idxs.mean()

def compute_common_nflips(clean_nf_idxs, advx_nf_idxs, indexes=False):
    """

    :param clean_nf_idxs:
    :param advx_nf_idxs:
    :param indexes:
    :return: only_acc_nf, only_rob_nf, common_nf
    """
    n_sample = min(clean_nf_idxs.size, advx_nf_idxs.size)
    clean_nf_idxs, advx_nf_idxs = clean_nf_idxs[:n_sample], advx_nf_idxs[:n_sample]

    only_acc_nf = ((clean_nf_idxs) & ~advx_nf_idxs)
    only_rob_nf = ((~clean_nf_idxs) & advx_nf_idxs)
    common_nf = ((clean_nf_idxs) & advx_nf_idxs)

    if not indexes:
        only_acc_nf = only_acc_nf.mean()
        only_rob_nf = only_rob_nf.mean()
        common_nf = common_nf.mean()

    return only_acc_nf, only_rob_nf, common_nf


def retrieve_baseline_bnf(model_pair_str, ds_id=ut.cifar10_id):
    old_id, new_id = ut.model_pairs_str_to_ids(model_pair_str)
    old_correct_clean_path = f"results/clean/{ut.MODEL_NAMES[ds_id][old_id]}/correct_preds.gz"
    new_correct_clean_path = f"results/clean/{ut.MODEL_NAMES[ds_id][new_id]}/correct_preds.gz"
    old_correct_adv_path = f"results/advx/{ut.MODEL_NAMES[ds_id][old_id]}/correct_preds_test.gz"
    new_correct_adv_path = f"results/advx/{ut.MODEL_NAMES[ds_id][new_id]}/correct_preds_test.gz"

    with open(old_correct_clean_path, 'rb') as f:
        old_correct_clean = pickle.load(f)
    with open(new_correct_clean_path, 'rb') as f:
        new_correct_clean = pickle.load(f)
    with open(old_correct_adv_path, 'rb') as f:
        old_correct_adv = pickle.load(f)
    with open(new_correct_adv_path, 'rb') as f:
        new_correct_adv = pickle.load(f)

    anf = compute_nflips(old_correct_clean, new_correct_clean, indexes=True)
    rnf = compute_nflips(old_correct_adv, new_correct_adv, indexes=True)
    _, _, bnfr = compute_common_nflips(anf, rnf)

    return bnfr


def evaluate_acc(model, device, test_loader, epoch=None, loss_fn=None):
    model = model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as t:
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                if loss_fn is not None:
                    loss = loss_fn(output, target)
                    test_loss += loss.item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

                t.set_postfix(
                    epoch='{}'.format(epoch),
                    completed='[{}/{} ({:.0f}%)]'.format(
                        batch_idx * len(data),
                        len(test_loader.dataset),
                        100. * batch_idx / len(test_loader)))
                    # loss='{:.4f}'.format(loss.item()))
                t.update()

        test_loss /= len(test_loader.dataset)
    return correct / len(test_loader.dataset)


if __name__ == "__main__":
    import pandas as pd
    # from secml.utils import fm

    # MODEL_NAMES = ['Kang2021Stable',
    #                'Rebuffi2021Fixing_70_16_cutmix_extra',
    #                'Gowal2021Improving_70_16_ddpm_100m']

    # ROOT = 'data'
    # exp_folder_name = fm.join(ROOT, 'exp_prova2')
    # advx_folder = fm.join(exp_folder_name, 'advx')
    # predictions_folder = fm.join(exp_folder_name, 'predictions')

    # df_old = pd.read_csv(fm.join(predictions_folder, f"{MODEL_NAMES[0]}_predictions.csv"), index_col=0)
    # df_new = pd.read_csv(fm.join(predictions_folder, f"{MODEL_NAMES[1]}_predictions.csv"), index_col=0)




    print("")