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
from model import LAMOI, Regression
from Evaluate import evaluate, get_auc
from Preprocessing import inverse_normalize
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import math

import argparse
import wandb
from tqdm import tqdm
import os

def output(loss, acc, precision, recall, f1, auc):
    print('\nloss')
    print('mean : {}'.format(np.array(loss).mean()))
    print('error : {}'.format(np.array(loss).std()))
    print('\n')

    print('acc')
    print('mean : {}'.format(np.array(acc).mean()))
    print('error : {}'.format(np.array(acc).std()))
    print('\n')

    print('precision')
    print('mean : {}'.format(np.array(precision).mean()))
    print('error : {}'.format(np.array(precision).std()))
    print('\n')

    print('recall')
    print('mean : {}'.format(np.array(recall).mean()))
    print('error : {}'.format(np.array(recall).std()))
    print('\n')

    print('f1')
    print('mean : {}'.format(np.array(f1).mean()))
    print('error : {}'.format(np.array(f1).std()))
    print('\n')

    print('auc')
    print('mean : {}'.format(np.array(auc).mean()))
    print('error : {}'.format(np.array(auc).std()))
    print('\n')

def test_model(x, y, paras):
    loss, acc, precision, recall, f1, auc, mask_eval = [], [], [], [], [], [], []
    for k in range(paras['k_fold']):
        loss.append(0)
        acc.append(0)
        precision.append(0)
        recall.append(0)
        f1.append(0)
        mask_eval.append(0)
        auc.append(0)
        
        indim = [x[i].shape[1] for i in range(len(x))]
        
        if paras['drug_std']:
            dir_name = './model/{}/{}/drug_std/{}fold/fold:{}'.format(paras['name'], paras['method'], paras['k_fold'], k)
        else:
            dir_name = './model/{}/{}/{}fold/fold:{}'.format(paras['name'], paras['method'], paras['k_fold'], k)
            
        model = torch.load(dir_name + '/LAMOI/hlayer({})_thres({})_epoch({})_bs({})_lr({})_dropout({})_tem({})_weight({})_zdim({})_k({})_ch({})_attn({}).pth'.format(
                        paras['h_layer'], 
                        paras['thres'], 
                        paras['epochs'], 
                        paras['batch_size'], 
                        paras['lr'], 
                        paras['dropout'], 
                        paras['temperature'],
                        paras['Weight'],  
                        paras['z_dim'], 
                        paras['k'],
                        paras['channel'],
                        paras['attention']
                    ))
        reg = torch.load(dir_name + '/reg/hlayer({})_thres({})_epoch({})_bs({})_lr({})_dropout({})_tem({})_weight({})_zdim({})_k({})_ch({})_attn({}).pth'.format(
                        paras['h_layer'], 
                        paras['thres'], 
                        paras['epochs'], 
                        paras['batch_size'], 
                        paras['lr'], 
                        paras['dropout'], 
                        paras['temperature'],
                        paras['Weight'],  
                        paras['z_dim'], 
                        paras['k'],
                        paras['channel'],
                        paras['attention']
                    ))
        
        tensor_test_x = [torch.FloatTensor(x[i].to_numpy()).to(paras['device']) for i in range(len(x))]
        tensor_test_y_ori = torch.FloatTensor(y['ori'].to_numpy())
        tensor_test_y_pro = torch.FloatTensor(y['pro'].to_numpy())

        loss_BCE = nn.BCELoss()
        loss_MSE = nn.MSELoss()

        with torch.no_grad():
            model.to(paras['device'])
            model.eval()
            reg.to(paras['device'])
            reg.eval()

            mask_eval, label_reg = [], []
            for i in range(len(tensor_test_y_ori)):
                mask_eval.append([0 if np.isnan(tensor_test_y_ori[i][j]) else 1 for j in range(len(tensor_test_y_ori[i]))])
                label_reg.append([0 if np.isnan(tensor_test_y_ori[i][j]) else tensor_test_y_ori[i][j] for j in range(len(tensor_test_y_ori[i]))])

            if paras['method'] != None:
                mask_train = None

            label_reg = torch.FloatTensor(label_reg)

            encode, concat_data = model(tensor_test_x)

            Pred_reg = reg(concat_data)

            P_reg = Pred_reg.detach().cpu().numpy().copy()
            pre_y = label_reg
            pre_p = P_reg
            if paras['drug_std']:
                P_reg = inverse_normalize(P_reg, paras['scaler'], paras['N_type'])

            P = np.array([[1 if P_reg[i][j] < paras['median'][j] else 0 for j in range(len(P_reg[i]))] for i in range(len(P_reg))]) * mask_eval
            GT = [[0 if np.isnan(tensor_test_y_ori[i][j]) else 1 if tensor_test_y_pro[i][j] < paras['median'][j] else 0 for j in range(len(tensor_test_y_pro[i]))] for i in range(len(tensor_test_y_pro))]
            P_reg = P_reg * mask_eval

            loss[k] = loss_MSE(torch.FloatTensor(P_reg), label_reg)
            acc[k], precision[k], recall[k], f1[k] = evaluate(P.T, np.array(GT).T, np.array(mask_eval).T)
            auc[k] = get_auc(P.T, np.array(GT).T, np.array(mask_eval).T)
            
    mean_acc = [np.array(acc[k]).mean() for k in range(paras['k_fold'])]
    mean_precision = [np.array(precision[k]).mean() for k in range(paras['k_fold'])]
    mean_recall = [np.array(recall[k]).mean() for k in range(paras['k_fold'])]
    mean_f1 = [np.array(f1[k]).mean() for k in range(paras['k_fold'])]
    mean_auc = [np.array(auc[k]).mean() for k in range(paras['k_fold'])]
    
    output(loss, mean_acc, mean_precision, mean_recall, mean_f1, mean_auc)
