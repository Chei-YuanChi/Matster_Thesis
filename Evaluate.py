# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import numpy as np
from sklearn import metrics

def get_auc(P, GT, mask_eval):
    """
    Parameters:
        P (list)          -- Drug response concentration prediction
        GT (list)         -- Drug response concentration ground truth
        mask_eval (list)  -- Mask of Nan data in ground truth
    """
    drug_response_gt, drug_response_p = [], []
    
    for i in range(len(np.array(P))):
        temp_gt, temp_p = [], []
        for j in range(len(np.array(mask_eval)[i])):
            if np.array(mask_eval)[i][j] == 1:
                temp_gt.append(np.array(GT)[i][j])
                temp_p.append(np.array(P)[i][j])
        drug_response_gt.append(temp_gt)
        drug_response_p.append(temp_p)
    
    auc_list = []
    for i in range(len(drug_response_gt)):
        try:
            AUC = metrics.roc_auc_score(drug_response_gt[i], drug_response_p[i])
            auc_list.append(AUC)
        except:
            continue
    
    return np.array(auc_list).mean()

def metric(TP, TN, FP, FN):
    """
    Parameters:
        TP (list)    -- True positive
        TN (list)    -- True negative
        FP (list)    -- False positive
        FN (list)    -- False negative
    """
    acc = (TP + TN) / (TP + TN + FP + FN)
    try: precision = TP  / (TP + FP)
    except: precision = 0
    try: recall = TP  / (TP + FN)
    except: recall = 0
    try: f1 = (2 * precision * recall) / (recall + precision)
    except: f1 = 0
    return acc, precision, recall, f1

def evaluate(pred, gt, mask):
    """
    Parameters:
        P (list)     -- Drug response concentration prediction
        GT (list)    -- Drug response concentration ground truth
        mask (list)  -- Mask of Nan data in ground truth
    """
    TP = [0 for i in range(len(gt))]
    TN, FP, FN = TP.copy(), TP.copy(), TP.copy()
    acc = [None for i in range(len(TP))]
    precision, recall, f1 = acc.copy(), acc.copy(), acc.copy()
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j] != 0:
                if gt[i][j] == pred[i][j]:
                    if gt[i][j] == 1: TP[i] += 1
                    else : TN[i] += 1
                else:
                    if gt[i][j] == 1: FN[i] += 1
                    else : FP[i] += 1
        acc[i], precision[i], recall[i], f1[i] = metric(TP[i], TN[i], FP[i], FN[i])
    return np.array(acc), np.array(precision), np.array(recall), np.array(f1)
