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
import matplotlib.pyplot as plt
import seaborn as sns

import argparse
import wandb
from tqdm import tqdm
import os

# Test individual fold
def test(x, y_ori, y_pro, model, reg, paras):
    if type(x[0]) == type(y_ori):
        tensor_test_x = [torch.FloatTensor(x[i].to_numpy()).to(paras['device']) for i in range(len(x))]
    else:
        tensor_test_x = [torch.FloatTensor(x[i]).to(paras['device']) for i in range(len(x))]
    tensor_test_y_ori = torch.FloatTensor(y_ori.to_numpy())
    tensor_test_y_pro = torch.FloatTensor(y_pro.to_numpy())
    
    loss_MSE = nn.MSELoss()
    with torch.no_grad():
        model.to(paras['device'])
        model.eval()
        reg.to(paras['device'])
        reg.eval()

        mask_eval, label_reg = [], []
        if len(tensor_test_y_ori.size()) == 1:
            tensor_test_y_ori = torch.unsqueeze(tensor_test_y_ori, 0)
            tensor_test_y_pro = torch.unsqueeze(tensor_test_y_pro, 0)
        for i in range(len(tensor_test_y_ori)):
            mask_eval.append([0 if np.isnan(tensor_test_y_ori[i][j]) else 1 for j in range(len(tensor_test_y_ori[i]))])
            label_reg.append([0 if np.isnan(tensor_test_y_ori[i][j]) else tensor_test_y_ori[i][j] for j in range(len(tensor_test_y_ori[i]))])

        label_reg = torch.FloatTensor(label_reg)

        encode, concat_data = model(tensor_test_x)

        Pred_reg = reg(concat_data)

        P_reg = Pred_reg.detach().cpu().numpy().copy()
        pre_y = label_reg.detach().cpu().numpy().copy()
        pre_p = P_reg
        if paras['drug_std']:
            P_reg = inverse_normalize(P_reg, paras['scaler'], paras['N_type'])
        P_reg = P_reg * mask_eval
        
        loss = []
        for i in range(len(np.array(P_reg).T)):
            loss.append(loss_MSE(torch.FloatTensor(np.array(P_reg).T[i]), label_reg.T[i]))
#         loss = loss_MSE(torch.FloatTensor(P_reg), label_reg)
        if tensor_test_y_ori.size()[0] == 1:
            return np.array(loss), None, None, None, None, None, pre_p, pre_y, mask_eval
        else:
            P = np.array([[1 if P_reg[i][j] < paras['median'][j] else 0 for j in range(len(P_reg[i]))] for i in range(len(P_reg))]) * mask_eval
            GT = [[0 if np.isnan(tensor_test_y_ori[i][j]) else 1 if tensor_test_y_pro[i][j] < paras['median'][j] else 0 for j in range(len(tensor_test_y_pro[i]))] for i in range(len(tensor_test_y_pro))]
            acc, precision, recall, f1 = evaluate(P.T, np.array(GT).T, np.array(mask_eval).T)
            auc = get_auc(P.T, np.array(GT).T, np.array(mask_eval).T)
        
    return np.array(loss), acc, precision, recall, f1, auc, pre_p, pre_y, mask_eval

# Set the all genes of a certain omics to 0 to judge its importance for predicting drug response
def get_omics_importance(data_x, data_y, index, dataset, base, model, reg, paras):
    loss, acc, precision, recall, f1, auc = {}, {}, {}, {}, {}, {}
    
    x = [data_x[i].copy() for i in dataset]
    x[index].iloc[:, :] = 0
    loss_temp, acc_temp, precision_temp, recall_temp, f1_temp, auc_temp, _, _, _ = test(x, data_y['ori'], data_y['pro'], model, reg, paras)
    loss = loss_temp - base['loss']
    acc = acc_temp - base['acc']
    precision = precision_temp - base['precision']
    recall = recall_temp - base['recall']
    f1 = f1_temp - base['f1']
    auc = auc_temp - base['auc']
    return {'loss' : loss, 'acc' : acc, 'precision' : precision, 'recall' : recall, 'f1' : f1, 'auc' : auc}

# Set the gene of a certain omics to 0 to judge its importance for predicting drug response
def get_gene_importance(data_x, data_y, index, dataset, base, model, reg, paras):
    loss, acc, precision, recall, f1, auc = {}, {}, {}, {}, {}, {}
    
    if type(data_x) == 'dict':
        for i in range(len(data_x[index].columns)):
            x = [data_x[j].copy() for j in dataset]
            x[index].iloc[:, i] = 0
            loss_temp, acc_temp, precision_temp, recall_temp, f1_temp, auc_temp, _, _, _ = test(x, data_y['ori'], data_y['pro'], model, reg, paras)
            loss[i] = loss_temp - base['loss']
            if 'acc' in base:
                acc[i] = acc_temp - base['acc']
                precision[i] = precision_temp - base['precision']
                recall[i] = recall_temp - base['recall']
                f1[i] = f1_temp - base['f1']
                auc[i] = auc_temp - base['auc']
        return {'loss' : loss, 'acc' : acc, 'precision' : precision, 'recall' : recall, 'f1' : f1, 'auc' : auc}
    else :
        for i in range(np.shape(data_x[index])[1]):
            x = [data_x[j].copy() for j in dataset]
            x[index][:, i] = 0
            loss_temp, acc_temp, precision_temp, recall_temp, f1_temp, auc_temp, _, _, _ = test(x, data_y['ori'], data_y['pro'], model, reg, paras)
            loss[i] = loss_temp - base['loss']
            if 'acc' in base:
                acc[i] = acc_temp - base['acc']
                precision[i] = precision_temp - base['precision']
                recall[i] = recall_temp - base['recall']
                f1[i] = f1_temp - base['f1']
                auc[i] = auc_temp - base['auc']
        return {'loss' : loss}

# Save the omics importance heatmap
def get_omics_heatmap(data, index):
    temp_list = []
    for i in range(len(data[0])):
        temp = np.array([data[j][i] for j in range(len(data))])
        xmax = temp.max()
        xmin = temp.min()
        temp = [2 * (temp[j] - xmin) / (xmax - xmin) - 1 for j in range(len(temp))]
        temp_list.append(temp)
    if index == None:
        temp_pd = pd.DataFrame(np.array(temp_list).T, columns = [i for i in range(len(temp_list))])
        plt.yticks([])
    else:
        temp_pd = pd.DataFrame(np.array(temp_list).T, index = index, columns = [i + 1 for i in range(len(temp_list))])
    
    plt.figure(figsize = (20, 9))
    plt.rc('font', size = 18)
    sns.heatmap(temp_pd, cmap = 'Blues', center = 0)
    plt.xlabel('Index of drugs', size = 18)
    plt.savefig('result/datasets_importance.png', transparent = True)

# Obtain the importance of each omics gene for predicting drug response
def get_feature_importance_for_all(data, drug_index):
    features_importance = [data[0][j][drug_index] for j in range(len(data[0]))]
    for i in range(1, 4):
        features_importance += [data[i][j][drug_index] for j in range(len(data[i]))] 
    xmax = np.array(features_importance).max()
    xmin = np.array(features_importance).min()
    temp = [(features_importance[j] - xmin) / (xmax - xmin) for j in range(len(features_importance)) if (xmax - xmin) !=0]
    if len(temp) == 0:
        temp = [0 for j in range(len(features_importance))]
    omics_features_importance = temp
        
    return omics_features_importance

# get the weight of individual omics
def get_omics_weight(x, y, paras):
    if paras['drug_std']:
        dir_name = './model/{}/{}/drug_std/{}fold/fold_{}'.format(paras['name'], paras['method'], paras['k_fold'], 0)
    else:
        dir_name = './model/{}/{}/{}fold/fold_{}'.format(paras['name'], paras['method'], paras['k_fold'], 0)

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
    loss_base, acc_base, precision_base, recall_base, f1_base, auc_base, P, GT, mask_eval = test(x, y['ori'].copy(), y['pro'].copy(), model, reg, paras)

    base = {'loss' : loss_base, 'acc' : acc_base,  'precision' : precision_base, 'recall' : recall_base, 'f1' : f1_base, 'auc' : auc_base}
    
    Me_importance = get_omics_importance(x, y.copy(), 0, paras['dataset'], base, model, reg, paras)
    Ex_importance = get_omics_importance(x, y.copy(), 1, paras['dataset'], base, model, reg, paras)
    CNV_importance = get_omics_importance(x, y.copy(), 2, paras['dataset'], base, model, reg, paras)
    Mu_importance = get_omics_importance(x, y.copy(), 3, paras['dataset'], base, model, reg, paras)
    
    if not os.path.exists('result/'):
        os.makedirs('result/')
        
    get_omics_heatmap([Mu_importance['loss'], CNV_importance['loss'], Me_importance['loss'], Ex_importance['loss']], ['Mutation', 'CNV', 'Methylation', 'Expression'])

# get the weight of individual genes
def get_gene_weight(x, y, paras):
    if paras['drug_std']:
        dir_name = './model/{}/{}/drug_std/{}fold/fold_{}'.format(paras['name'], paras['method'], paras['k_fold'], 0)
    else:
        dir_name = './model/{}/{}/{}fold/fold_{}'.format(paras['name'], paras['method'], paras['k_fold'], 0)

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
    
    x_temp = x.copy()
    for i in range(len(x_temp)):
        x_temp[i] = np.array(x_temp[i].iloc[paras['cell_line'], :]).reshape((1, -1))
    y_temp = {'ori' : y['ori'].iloc[paras['cell_line'],:], 'pro' : y['pro'].iloc[paras['cell_line'],:]}
    cell_line_loss_base, _, _, _, _, _, cell_line_P, cell_line_GT, cell_line_mask_eval = test(x_temp, y_temp['ori'].copy(), y_temp['pro'].copy(), model, reg, paras)
    loss_base, acc_base, precision_base, recall_base, f1_base, auc_base, P, GT, mask_eval = test(x, y['ori'].copy(), y['pro'].copy(), model, reg, paras)
    cell_line_base = {'loss' : loss_base}
    
    cell_line_Me_feature_importance = get_gene_importance(x_temp, y_temp, 0, paras['dataset'], cell_line_base, model, reg, paras)
    cell_line_Ex_feature_importance = get_gene_importance(x_temp, y_temp, 1, paras['dataset'], cell_line_base, model, reg, paras)
    cell_line_CNV_feature_importance = get_gene_importance(x_temp, y_temp, 2, paras['dataset'], cell_line_base, model, reg, paras)
    cell_line_Mu_feature_importance = get_gene_importance(x_temp, y_temp, 3, paras['dataset'], cell_line_base, model, reg, paras)
    
    cell_data = [cell_line_Me_feature_importance['loss'], cell_line_Ex_feature_importance['loss'], cell_line_CNV_feature_importance['loss'], cell_line_Mu_feature_importance['loss']]
    
    drug_related = pd.DataFrame(get_feature_importance_for_all(cell_data, paras['drug_index']), index = list(x[0].columns)+list(x[1].columns)+list(x[2].columns)+list(x[3].columns))
    drug_related.index = [drug_related.index[i].split('_')[0] for i in range(len(drug_related.index))]
    
    if not os.path.exists('result/'):
        os.makedirs('result/')
        
    if not os.path.exists('result/reaction/cell({})'.format(x[0].index[paras['cell_line']])):
        os.makedirs('result/reaction/cell({})'.format(x[0].index[paras['cell_line']]))
    drug_related.sort_values(by = [0], ascending=False).to_csv('result/reaction/cell({})/drug({})_all.csv'.format(x[0].index[paras['cell_line']], y['pro'].columns[paras['drug_index']]))