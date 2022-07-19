from utils.utils import MODEL_NAMES
import json
from os.path import join
import pandas as pd
import numpy as np

path = 'models\model_info\cifar10\Linf'

models_info_df = pd.DataFrame(columns=['Clean Acc.', 'Robust Acc.'])
models_info_df.index.name = 'Model'

# print(f"Name: Clean Acc. (%) / Robust Acc. (%)")
robaccs = []
for model in MODEL_NAMES:
    with open(join(path, f"{model}.json")) as f:
        model_info = json.load(f)
    models_info_df.loc[model] = [model_info['clean_acc'], model_info['autoattack_acc']]
    robaccs.append(float(model_info['autoattack_acc']))
    # print(f"{model}: {model_info['clean_acc']} / {model_info['autoattack_acc']}")
# robaccs = np.array(robaccs)
# idxs = np.argsort(robaccs)
# model_names_sorted = np.array(MODEL_NAMES)[idxs]
print(models_info_df)

