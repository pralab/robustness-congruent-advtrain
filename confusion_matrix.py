import os
import pickle
from utils.utils import MODEL_NAMES, MODEL_NAMES_LONG_SHORT_DICT, MODEL_NAMES_LONG_SYMBOLS_DICT
import numpy as np
import seaborn as sns
from itertools import product
from utils.eval import compute_nflips
import matplotlib.pyplot as plt
import matplotlib

# mpl.rcParams.update(mpl.rcParamsDefault)
# import scienceplots
# plt.style.use('science')
#
# mpl.rcParams['font.size'] = 15

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['mathtext.fontset'] = 'stix'

def compute_churn_matrix(model_ids = (1,2,3),
                        root='results',
                         advx=False):

    if advx:
        root = os.path.join(root, 'advx')
    else:
        root = os.path.join(root, 'clean')

    correct_preds_matrix = None
    for i, model_id in enumerate(model_ids):
        with open(os.path.join(root, MODEL_NAMES[model_id], 'correct_preds.gz'), 'rb') as f:
            correct_preds = pickle.load(f).numpy()
        print(f"{model_id} -> Acc: {correct_preds.mean()}")
        if correct_preds_matrix is None:
            correct_preds_matrix = np.empty(shape=(len(model_ids), correct_preds.shape[0]), dtype=bool)
        correct_preds_matrix[i, :] = correct_preds

    idxs = np.arange(len(model_ids))

    models_accs = np.empty(shape=len(model_ids))
    models_nfr_matrix = np.empty(shape=(len(model_ids), len(model_ids)))
    for i, j in product(idxs, idxs):
        models_accs[i] = correct_preds_matrix[i].mean()
        models_nfr_matrix[i, j] = compute_nflips(correct_preds_matrix[i, :], correct_preds_matrix[j, :])


    return models_accs, models_nfr_matrix


def reorder_churn_matrix(churn_matrix_dict, order_by='rob_acc', ascending=True):
    from collections.abc import Iterable
    new_idxs = np.argsort(churn_matrix_dict[order_by])
    if not ascending:
        new_idxs = new_idxs[::-1]

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

    nfr *= 100
    # import pandas as pd
    # nfr = pd.DataFrame(nfr)
    # nfr = nfr.applymap(lambda x: f"{x:.2f}")
    sns.heatmap(nfr, annot=True, fmt='.2f', ax=ax[1],
                cmap=cmap, cbar=False, vmin=0, vmax=10,
                xticklabels=[], yticklabels=[])

    model_names = list(model_names)
    # Change order
    model_names.reverse()
    accs = np.flip(accs)
    # model_names_short = [MODEL_NAMES_LONG_SHORT_DICT[m] for m in model_names]
    # todo: incubo
    model_names_short = [f"{MODEL_NAMES_LONG_SHORT_DICT[m]} ({MODEL_NAMES_LONG_SYMBOLS_DICT[m]})"
                         for m in model_names]

    bars = ax[0].barh(model_names_short, accs*100, align='center', color=bar_color)
    ax[0].bar_label(bars, labels=[f'{x:,.2f}' for x in bars.datavalues],
                    label_type='edge', padding=-50, color='white', fmt='{:,.2f}')


def plot_all_churn_matrix():
    with open('results/perf_matrix.gz', 'rb') as f:
        data = pickle.load (f)

    # correzioni a mano
    data['rob_acc'] = np.array([0.5585, 0.5845, 0.5595, 0.5935, 0.5910, 0.6055, 0.6330])

    data['nfr'][1, 3] = 0.0334
    data['nfr'][2, 3] = 0.0327
    data['nfr'][3, 6] = 0.0239
    data['nfr'][4, 3] = 0.0358

    data['rob_nfr'][0, 3] = 0.0540
    data['rob_nfr'][0, 6] = 0.0325
    data['rob_nfr'][1, 4] = 0.0535
    data['rob_nfr'][1, 6] = 0.0415
    data['rob_nfr'][2, 1] = 0.0515
    data['rob_nfr'][2, 3] = 0.0525
    data['rob_nfr'][2, 4] = 0.0280
    data['rob_nfr'][2, 6] = 0.0250
    data['rob_nfr'][3, 6] = 0.0495
    data['rob_nfr'][4, 3] = 0.0545
    data['rob_nfr'][4, 6] = 0.0295
    data['rob_nfr'][5, 6] = 0.0400

    data['acc'] = 1 - data['acc']
    data['rob_acc'] = 1 - data['rob_acc']

    for adv in (False, True):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6.7), squeeze=True)

        order = 'rob_acc' if adv else 'acc'
        accs_key = order
        nfr_key = 'rob_nfr' if adv else 'nfr'
        color = 'seagreen'#'tomato' if adv else 'seagreen'


        data = reorder_churn_matrix(data, order_by=order, ascending=False)


        model_names_short = [f"{MODEL_NAMES_LONG_SHORT_DICT[m]} ({MODEL_NAMES_LONG_SYMBOLS_DICT[m]})"
                             for m in data['model_names']]

        plot_churn_matrix(ax, model_names=data['model_names'],
                          accs=data[accs_key], nfr=data[nfr_key],
                          bar_color=color)

        ax[1].set_xticks(np.arange(len(data['model_names'])) + 0.5,
                         model_names_short,
                         rotation=45)

        if adv:
            titles = ['Robust Error', 'RNFs']
        else:
            titles = ['Test Error', 'NFs']

        for i, t in enumerate(titles):
            ax[i].set_title(f"{t} (%)")

        ax[0].set_xlim([0, 50 if adv else 20])
        ax[0].minorticks_off()
        ax[1].minorticks_off()
        # ax[0].set_aspect('equal')
        # ax[1].set_aspect('equal')


        fig.tight_layout()
        fig.show()
        fig.savefig(f'images/churn_matrix_{"robs" if adv else "accs"}.pdf')

    # for i in range(2):
    #     ax_all[i] = ax[i]
    #     ax_all[i+2] = ax[i]

    # fig_all.show()

    print("")

def find_candidate_model_pairs():
    with open('results/perf_matrix.gz', 'rb') as f:
        data = pickle.load(f)
    accs = data['acc']
    robs = data['rob_acc']
    model_ids = data['model_ids']
    model_names = data['model_names']

    n_models = len(model_ids)


    old_model_ids = []
    new_model_ids = []
    for old_id, new_id in product(range(n_models), range(n_models)):
        if (accs[new_id] > accs[old_id]) and (robs[new_id] > robs[old_id]):
            old_model_ids.append(model_ids[old_id])
            new_model_ids.append(model_ids[new_id])

    # old_accs = [accs[i-1] for i in old_model_ids]
    # new_accs = [accs[i-1] for i in new_model_ids]
    # old_robs = [robs[i-1] for i in old_model_ids]
    # new_robs = [robs[i-1] for i in new_model_ids]
    #
    # plt.figure()
    # plt.plot(old_accs, 'b')
    # plt.plot(new_accs, 'b--')
    # plt.plot(old_robs, 'r')
    # plt.plot(new_robs, 'r--')
    # plt.show()


    return old_model_ids, new_model_ids


if __name__ == '__main__':
    # model_ids = (1,2,3,4,5,6,7)
    # acc, nfr_matrix = compute_churn_matrix(model_ids=model_ids)
    # rob_acc, rob_nfr_matrix = compute_churn_matrix(model_ids=model_ids, advx=True)
    #
    # data = {'acc': acc, 'nfr': nfr_matrix,
    #         'rob_acc': rob_acc, 'rob_nfr': rob_nfr_matrix,
    #         'model_ids': (1,2,3,4,5,6,7),
    #         'info': 'questa roba contiene le matrici tutti contro tutti dei modelli baseline'}
    # data['model_names'] = tuple(MODEL_NAMES[i] for i in data['model_ids'])

    # with open('results/perf_matrix.gz', 'wb') as f:
    #     pickle.dump(data, f)

    plot_all_churn_matrix()

    # find_candidate_model_pairs()

    print("")