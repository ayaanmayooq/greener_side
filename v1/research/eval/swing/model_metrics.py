import numpy as np

def auc_brier(y_true, p):
    from sklearn.metrics import roc_auc_score, brier_score_loss
    return dict(auc=float(roc_auc_score(y_true, p)),
                brier=float(brier_score_loss(y_true, p)))