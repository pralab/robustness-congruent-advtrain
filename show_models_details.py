from utils.utils import MODEL_NAMES
import json
from os.path import join
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

path = 'models\model_info\cifar10\Linf'

keys = ['clean_acc', 'reported', 'autoattack_acc', 'external',
        'additional_data', 'architecture', 'unreliable']

# Some info
models_info_df = pd.DataFrame(columns=['Clean Acc.',
                                       'Reported',
                                       'AutoAttack Acc.',
                                       'Best Known Acc.'],
                              dtype=float)
models_info_df.index.name = 'Model'

# All info
models_all_info_df = pd.DataFrame(columns=keys,
                              dtype=float)
models_info_df.index.name = 'Model'

# Minumum Rob acc
models_info_minimum_df = pd.DataFrame(columns=['Clean Acc.',
                                       'Robust Acc.'],
                              dtype=float)
models_info_minimum_df.index.name = 'Model'
# print(f"Name: Clean Acc. (%) / Robust Acc. (%)")
metric = []
for model in MODEL_NAMES:
    metric_sel = np.array([100., 100., 100.])
    with open(join(path, f"{model}.json")) as f:
        model_info = json.load(f)

    try:
        metric_sel[0] = float(model_info['external'])
        models_info_df.loc[model] = [float(model_info['clean_acc']),
                                     float(model_info['reported']),
                                     float(model_info['autoattack_acc']),
                                     float(model_info['external'])]
    except:
        models_info_df.loc[model] = [float(model_info['clean_acc']),
                                     float(model_info['reported']),
                                     float(model_info['autoattack_acc']),
                                     None]
    metric_sel[1] = float(model_info['reported'])
    metric_sel[2] = float(model_info['autoattack_acc'])
    metric.append(metric_sel.min())

    models_info_minimum_df.loc[model] = [float(model_info['clean_acc']),
                                         metric_sel.min()]

    cumul_info = []
    for k in keys:
        try:
            cumul_info.append(model_info[k])
        except:
            cumul_info.append(None)
    models_all_info_df.loc[model] = cumul_info


print(models_all_info_df)

metric = np.array(metric)
idxs = np.argsort(metric)
model_names_sorted = np.array(MODEL_NAMES)[idxs]
print(model_names_sorted)

beta = 0.5
plt.figure(figsize=(15*beta, 10*beta))
sns.heatmap(models_info_minimum_df, annot=True, fmt='g')
plt.tight_layout()
plt.show()


# models_info_df.dtypes = float
print(models_info_df)

