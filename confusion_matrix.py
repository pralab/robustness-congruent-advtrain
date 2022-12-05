import os
import pickle
from utils.utils import MODEL_NAMES
import numpy as np
import seaborn as sns
from itertools import product
from utils.eval import compute_nflips
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.size'] = 15
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['mathtext.fontset'] = 'stix'

def compute_churn_matrix(model_ids = (1,2,3,4,5,6),
                        path='results/day-04-11-2022_hr-16-50-24_epochs-12_batchsize-500',
                         advx=False):

    # for model_id in model_ids:
    #     model_name = MODEL_NAMES[model_id]
    #     model = load_model(model_name, dataset='cifar10', threat_model='Linf')

    if advx:
        path = 'results/advx'

    c = 0
    correct_preds_matrix = None #np.empty(shape=())
    new_correct = None
    for i, model_id in enumerate(model_ids):
        for root, dirs, files in os.walk(path):
            if (f"old-{model_id}" in root) \
                and (any('results_' in file_name for file_name in files) and ('advx' not in root)):

                # res_list = [file_name for file_name in files if 'results_' in file_name]
                # for res in res_list:
                results_fname = next((file_name for file_name in files if 'results_' in file_name))
                with open(os.path.join(root, results_fname), 'rb') as f:
                    results = pickle.load(f)
                old_correct = results['old_correct'].numpy()
                print(f"{model_id} -> Old acc: {old_correct.mean()}")
                if correct_preds_matrix is None:
                    correct_preds_matrix = np.empty(shape=(len(model_ids), old_correct.shape[0]), dtype=bool)
                correct_preds_matrix[i, :] = old_correct
                c += 1
                break

    if c == 0:
        for i, model_id in enumerate(model_ids):
            for root, dirs, files in os.walk(path):
                if ((MODEL_NAMES[model_id] in root) and ('advx' in root) and ('correct_preds.gz' in files)):
                    with open(os.path.join(root, 'correct_preds.gz'), 'rb') as f:
                        old_correct = pickle.load(f).numpy()
                    print(f"{model_id} -> Old acc: {old_correct.mean()}")
                    if correct_preds_matrix is None:
                        correct_preds_matrix = np.empty(shape=(len(model_ids), old_correct.shape[0]), dtype=bool)
                    correct_preds_matrix[i, :] = old_correct
                    c += 1
                    break
    assert c == len(model_ids)

    idxs = np.arange(len(model_ids))

    models_acc_gain_matrix = np.empty(shape=(len(model_ids), len(model_ids)))
    models_nfr_matrix = models_acc_gain_matrix.copy()
    for i, j in product(idxs, idxs):
        models_acc_gain_matrix[i, j] = (correct_preds_matrix[j, :].mean() - correct_preds_matrix[i, :].mean())*100
        models_nfr_matrix[i, j] = compute_nflips(correct_preds_matrix[i, :], correct_preds_matrix[j, :])*100

    return models_acc_gain_matrix, models_nfr_matrix

def reorder_churn_matrix(churn_matrix_dict, order_by='rob_accs'):
    from collections.abc import Iterable
    new_idxs = np.argsort(churn_matrix_dict[order_by])

    for key, value in churn_matrix_dict.items():
        if isinstance(value, np.ndarray):
            if len(value.shape) > 1:
                churn_matrix_dict[key][:] = churn_matrix_dict[key][:, new_idxs]
                churn_matrix_dict[key][:] = churn_matrix_dict[key][new_idxs, :]

            else:
                churn_matrix_dict[key] = churn_matrix_dict[key][new_idxs]
        elif isinstance(value, Iterable):
            churn_matrix_dict[key] = tuple([value[idx] for idx in new_idxs])

    return churn_matrix_dict

def plot_churn_matrix(ax, model_names, accs, nfr,
                      bar_color='seagreen', cmap='summer'):

    sns.heatmap(nfr, annot=True, fmt='g', ax=ax[1],
                cmap=cmap, cbar=False, vmin=0, vmax=10,
                xticklabels=[], yticklabels=[])

    model_names = list(model_names)
    model_names.reverse()
    accs = np.flip(accs)
    p = ax[0].barh(model_names, accs*100, align='center', color=bar_color)
    ax[0].bar_label(p, label_type='edge')


def plot_all_churn_matrix():
    with open('results/perf_matrix.gz', 'rb') as f:
        data = pickle.load(f)

    for adv in (False, True):
        fig, ax = plt.subplots(1, 2, figsize=(11, 7), squeeze=True)

        order = 'rob_accs' if adv else 'accs'
        accs_key = order
        nfr_key = 'rob_nfr' if adv else 'nfr'
        color = 'seagreen'#'tomato' if adv else 'seagreen'
        data = reorder_churn_matrix(data, order_by=order)

        plot_churn_matrix(ax, model_names=data['model_names'],
                          accs=data[accs_key], nfr=data[nfr_key],
                          bar_color=color)
        ax[1].set_xticks(np.arange(len(data['model_names'])) + 0.5,
                         data['model_names'],
                         rotation=90)
        titles = ['Accuracy', 'NFR']
        for j, title_j in enumerate(titles):
            ax[j].set_title(f'{"Robust " if adv else ""}{titles[j]} (%)')
            if j % 2 == 0:
                ax[j].set_xlim([0, 100])
        fig.tight_layout()
        fig.show()
        fig.savefig(f'images/churn_matrix_{"robs" if adv else "accs"}.pdf')

    print("")


if __name__ == '__main__':
    # acc_gain_matrix, nfr_matrix = compute_churn_matrix()
    # rob_acc_gain_matrix, rob_nfr_matrix = compute_churn_matrix(advx=True)

    # data = {'acc': acc_gain_matrix, 'nfr': nfr_matrix,
    #         'rob_acc': acc_gain_matrix, 'rob_nfr': nfr_matrix,
    #         'model_ids': (1,2,3,4,5,6),
    #         'info': 'questa roba contiene le matrici tutti contro tutti dei modelli baseline'}
    # data['model_names'] = tuple(MODEL_NAMES[i] for i in data['model_ids'])

    # with open('results/perf_matrix.gz', 'wb') as f:
    #     pickle.dump(data, f)

    # for ordered_by in (None, 'accs', 'rob_accs', 'both'):
    #     plot_all_churn_matrix(order_by=ordered_by)
    #     print("")

    plot_all_churn_matrix()

    print("")
