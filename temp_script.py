from secml.utils import fm
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.eval import compute_nflips
from utils.utils import preds_fname, PERF_FNAME, NFLIPS_FNAME, OVERALL_RES_FNAME, \
    RESULTS_DIRNAME_DEFAULT, PREDS_DIRNAME_DEFAULT, MODEL_NAMES, \
    COLUMN_NAMES

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

EXP_FOLDER_NAME = 'data/2ksample_250steps_100batchsize_day-09-07-2022_hr-19-46-34'
predictions_folder = fm.join(EXP_FOLDER_NAME, PREDS_DIRNAME_DEFAULT)
results_folder = fm.join(EXP_FOLDER_NAME, RESULTS_DIRNAME_DEFAULT)

# for i, model_name_i in enumerate(MODEL_NAMES):
#     print(f"{i}: {model_name_i}")
#     df = pd.read_csv(fm.join(predictions_folder, preds_fname(model_name_i)),
#                      index_col=0)
#     correct_preds_df = pd.DataFrame(columns=COLUMN_NAMES[1:])
#     true_labels_df = df.pop('True')
#     for c in df:
#         correct_preds_df[c] = (df[c] == true_labels_df)

robacc_df = pd.read_csv(fm.join(results_folder, PERF_FNAME),
                        index_col=0)
nflips_df = pd.read_csv(fm.join(results_folder, NFLIPS_FNAME),
                        index_col=0)

# Evaluations on the fly with cumulative advx
cum_rob_acc = []
cum_nflips = []
for i, m in enumerate(MODEL_NAMES):
    for j, c in enumerate(COLUMN_NAMES[1:]):
        if (j == (i+1)):
            cum_rob_acc.append(robacc_df.loc[MODEL_NAMES[i]][COLUMN_NAMES[1:][j]])
            cum_nflips.append(nflips_df.loc[MODEL_NAMES[i]][COLUMN_NAMES[1:][j]])



# Best possible evaluation with all advx in the hystory
robacc_df_best = robacc_df[[robacc_df.columns[0], robacc_df.columns[-1]]]
robacc_df_best = robacc_df_best.rename(columns={robacc_df.columns[-1]: 'Robust Acc.',
                                                'Clean': 'Clean Acc.'})
nflips_df_best = nflips_df[[nflips_df.columns[0], nflips_df.columns[-1]]]
nflips_df_best = nflips_df_best.rename(columns={nflips_df.columns[-1]: 'Robust Churn',
                                                'Clean': 'Acc. Churn'})
best_res = pd.concat([robacc_df_best, nflips_df_best], axis=1)

cum_res = best_res.copy()
cum_res['Robust Acc.'] = cum_rob_acc
cum_res['Robust Churn'] = cum_nflips

print(best_res)
print(cum_res)

beta = 0.5
plt.figure(figsize=(15 * beta, 10 * beta))
sns.heatmap(cum_res, annot=True, fmt='g')
plt.title("Cumulative evaluation")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(15 * beta, 10 * beta))
sns.heatmap(best_res, annot=True, fmt='g')
plt.title("Best evaluation")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


beta = 0.5
plt.figure(figsize=(15 * beta, 10 * beta))
sns.heatmap(cum_res.transpose(), annot=True, fmt='g')
plt.title("Cumulative evaluation")
plt.yticks(rotation=45)
# plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(15 * beta, 10 * beta))
sns.heatmap(best_res.transpose(), annot=True, fmt='g')
plt.title("Best evaluation")
plt.yticks(rotation=45)
# plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print("")


