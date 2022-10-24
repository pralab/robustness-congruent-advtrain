import torch
import math
import pandas as pd
from tqdm import tqdm


# def predict(model, x, batch_size, device, logger=None):
#     preds = []
#     outputs = []
#     n_examples = x.shape[0]
#     x = x.to(device)
#     model.to(device)

#     num_batches = math.ceil(n_examples / batch_size)
#     with torch.no_grad():
#         for batch_i in range(num_batches):
#             if logger is not None:
#                 logger.debug(f"{batch_i + 1}/{num_batches}")
#             start_i = batch_i * batch_size
#             end_i = start_i + batch_size

#             out = model(x[start_i:end_i])
#             outputs.append(out)
#             pred = torch.argmax(out, axis=1)
#             preds.extend(pred.tolist())
#     outputs = torch.cat(outputs)

#     return preds, outputs

def correct_predictions(model, test_loader, device,):
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

def compute_nflips(old_preds, new_preds):
    old_preds = pd.Series(old_preds.cpu().tolist())
    new_preds = pd.Series(new_preds.cpu().tolist())
    nf_idxs = (old_preds & (~new_preds))
    return nf_idxs.mean()


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