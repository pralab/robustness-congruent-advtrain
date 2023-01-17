from sklearn.metrics import precision_recall_fscore_support, precision_score, \
    recall_score, f1_score, confusion_matrix, roc_curve
import numpy as np
from utils.eval import compute_nflips, compute_pflips

def compute_all_metrics(preds, y_true):
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds,
                                                       pos_label=1,
                                                       average='binary')
    metrics = {'tpr': tpr, 'fpr': fpr, 'prec': prec, 'rec': rec, 'f1': f1}

    return metrics

def compute_churn_metrics(preds, old_preds, y_true):
    if old_preds is not None:
        nf = compute_nflips(old_preds=old_preds, new_preds=preds, indexes=True)
        pf = compute_pflips(old_preds=old_preds, new_preds=preds, indexes=True)
        nfr_pos = nf[y_true == 1].mean()
        nfr_neg = nf[y_true == 0].mean()
        pfr_pos = pf[y_true == 1].mean()
        pfr_neg = pf[y_true == 0].mean()
        nfr_tot = nf.mean()
        pfr_tot = pf.mean()
    else:
        nfr_pos, nfr_neg, pfr_pos, pfr_neg, nfr_tot, pfr_tot = (None,) * 6

    metrics = {'nfr_pos': nfr_pos, 'nfr_neg': nfr_neg, 'nfr_tot': nfr_tot,
               'pfr_pos': pfr_pos, 'pfr_neg': pfr_neg, 'pfr_tot': pfr_tot}

    return metrics



def adjust_threshold(clf, X_val, y_val, fpr=0.01):
    val_scores_i = clf.decision_function(X_val)
    fpr, tpr, thresholds = roc_curve(y_val, val_scores_i, pos_label=1)
    # Find the index nearest the selected maximum FPR
    idx = (np.abs(fpr - 0.01)).argmin()
    threshold = thresholds[idx]

    # import matplotlib.pyplot as plt
    # plt.plot(fpr, tpr)
    # plt.axvline(x=fpr[idx])
    # plt.show()

    return threshold