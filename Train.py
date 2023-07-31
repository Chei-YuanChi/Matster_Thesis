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

def train_model(X, Y, paras):
    all_drugs_index = [i for i in range(Y['ori'].shape[0])]
    
    for k in range(paras['k_fold']):
        if k == 0:
            train_idx = all_drugs_index[: int(len(all_drugs_index) / paras['k_fold'] * (paras['k_fold'] - 1))]
            val_idx = all_drugs_index[int(len(all_drugs_index) / paras['k_fold'] * (paras['k_fold'] - 1)) :]
        elif k == paras['k_fold'] - 1:
            train_idx = all_drugs_index[int(len(all_drugs_index) / paras['k_fold']) + 1:]
            val_idx = all_drugs_index[: int(len(all_drugs_index) / paras['k_fold']) + 1]
        else:
            train_idx = all_drugs_index[: int(len(all_drugs_index) / paras['k_fold'] * k)] + all_drugs_index[int(len(all_drugs_index) / paras['k_fold']  * (k + 1)):]
            val_idx = all_drugs_index[int(len(all_drugs_index) / paras['k_fold'] * k): int(len(all_drugs_index) / paras['k_fold'] * (k + 1))]
        
        if paras['drug_std']:
            dir_name = './model/{}/{}/drug_std/{}fold/fold:{}'.format(paras['name'], paras['method'], paras['k_fold'], k)
        else:
            dir_name = './model/{}/{}/{}fold/fold:{}'.format(paras['name'], paras['method'], paras['k_fold'], k)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name + '/LAMOI/')
            os.makedirs(dir_name + '/reg/')
        
        indim = [X[i].shape[1] for i in range(len(X))]
            
        tensor_train_X = [torch.FloatTensor(X[i].iloc[train_idx, :].to_numpy()) for i in range(len(X))]
        tensor_val_X = [torch.FloatTensor(X[i].iloc[val_idx, :].to_numpy()) for i in range(len(X))]
        
        tensor_train_Y = [torch.FloatTensor(Y['std'].iloc[train_idx, :].to_numpy()), torch.FloatTensor(Y['ori'].iloc[train_idx, :].to_numpy()), torch.FloatTensor(Y['pro'].iloc[train_idx, :].to_numpy())]
        tensor_val_Y = [torch.FloatTensor(Y['std'].iloc[val_idx, :].to_numpy()), torch.FloatTensor(Y['ori'].iloc[val_idx, :].to_numpy()), torch.FloatTensor(Y['pro'].iloc[val_idx, :].to_numpy())]
        
        model = LAMOI(input_dim = indim, 
                      h_dim = paras['h_dim'], 
                      z_dim = paras['z_dim'], 
                      temperature = paras['temperature'],
                      dropout = paras['dropout'],
                      device = paras['device'],
                      attention = paras['attention'],
                      Weight = paras['Weight'],
                      k = paras['k'],
                      channel = paras['channel'])
        
        optimizer_model = torch.optim.Adam(model.parameters(), lr = paras['lr']['model'])
        if len(indim) > 1 and paras['attention']:
            reg = Regression(tensor_train_Y[0].shape[1], paras['z_dim'] * len(indim) * (len(indim) - 1), paras['cls_dim'])
        else:
            reg = Regression(tensor_train_Y[0].shape[1], paras['z_dim'] * len(indim), paras['cls_dim'])
            
        optimizer_reg = torch.optim.Adam(reg.parameters(), lr = paras['lr']['reg'])
        
        trainDataset = torch.utils.data.TensorDataset(*tensor_train_X, *tensor_train_Y)
        trainLoader = torch.utils.data.DataLoader(dataset = trainDataset, batch_size = paras['batch_size'], shuffle = False, num_workers = 0)
        
        loss_MSE = nn.MSELoss()
        e_name = "thres({})_epoch({})_bs({})_lr({})_dropout({})_zdim({})_tem({})_k({})_ch({})_fold({})".format(paras['thres'],
                                                                                                               paras['epochs'],
                                                                                                               paras['batch_size'], 
                                                                                                               paras['lr'],
                                                                                                               paras['dropout'],  
                                                                                                               paras['z_dim'], 
                                                                                                               paras['temperature'],
                                                                                                               paras['k'], 
                                                                                                               paras['channel'], k)
        group_name = "dataset({})_method({})_weight({})_attn({})_std({})".format(
            paras['name'], 
            paras['method'], 
            paras['Weight'],
            paras['attention'],
            paras['drug_std'])
        
        wandb.login(key = "3c35754d4df7f0ecbaa130aaa78ad8060bd3822d")
        wandb.init(project = "final_test", group = group_name, name = e_name, entity = 'taurus')

        min_loss = 1000
        for epoch in range(paras['epochs']):
            model.to(paras['device'])
            model.train()
            reg.to(paras['device'])
            reg.train()
                
            acc = 0
            all_bs = 0
            train_loss = []

            pbar = tqdm(enumerate(trainLoader), total = len(trainLoader))
            time = 0
            for _, (data) in pbar:
                if len(np.array(data[0])) == 1:
                    break
                
                x = [data[i].to(paras['device']) for i in range(len(indim))]
                
                encode, concat_data = model(x)
                
                mask_eval, label_reg, label_std = [], [], []
                for i in range(len(data[-2])):
                    mask_eval.append([0 if np.isnan(data[-2][i][j]) else 1 for j in range(len(data[-2][i]))])
                    label_reg.append([0 if np.isnan(data[-2][i][j]) else data[-2][i][j] for j in range(len(data[-2][i]))])
                    label_std.append([0 if np.isnan(data[-2][i][j]) else data[-3][i][j] for j in range(len(data[-3][i]))])
                    
                if paras['method'] != None:
                    label_reg = data[-1]
                    
                label_reg = torch.FloatTensor(label_reg)
                label_std = torch.FloatTensor(label_std)
                                      
                Pred_reg = reg(concat_data)
                Pred_reg *= torch.FloatTensor(mask_eval).to(paras['device'])
                
                loss = loss_MSE(Pred_reg, label_reg.to(paras['device']))
                if paras['drug_std']:
                    loss = loss_MSE(Pred_reg, label_std.to(paras['device'])) 
                
                loss_LA = model.loss(x)
                
                loss = loss + loss_LA
                    
                train_loss.append(loss.item())
                
                optimizer_model.zero_grad()
                optimizer_reg.zero_grad()
#                 loss.backward(retain_graph=True)
                loss.backward()
                optimizer_model.step()
                optimizer_reg.step()

                P = Pred_reg.detach().cpu().numpy().copy()
                GT = np.array(label_reg).copy()
                if paras['drug_std']:
                    P = inverse_normalize(P, paras['scaler'], paras['N_type'])
                    
                for i in range(len(P)):
                    for j in range(len(P[i])):
                        if P[i][j] < paras['median'][j]:
                            P[i][j] = 1
                        else:
                            P[i][j] = 0
                        if GT[i][j] < paras['median'][j]:
                            GT[i][j] = 1
                        else:
                            GT[i][j] = 0
                
                for i in range(len(mask_eval)):
                    for j in range(len(mask_eval[i])):
                        if mask_eval[i][j] != 0:
                            all_bs += 1
                            if P[i][j] == GT[i][j]:
                                acc += 1

                pbar.set_description(
                    f"Train Iter: {epoch+1}. "
                    f"Train Loss: {loss.item():.4f}. ")
                pbar.update()
            
            try:
                acc /= all_bs
            except:
                acc = 0
                
            train_loss = np.average(train_loss)
            val_loss, val_acc, val_precision, val_recall, val_f1, val_auc = val_model(tensor_val_X, tensor_val_Y, model, reg, paras)
            if min_loss > val_loss:
                min_loss = val_loss
                torch.save(model, dir_name + '/LAMOI/hlayer({})_thres({})_epoch({})_bs({})_lr({})_dropout({})_tem({})_weight({})_zdim({})_k({})_ch({})_attn({}).pth'.format(
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
                torch.save(reg, dir_name + '/reg/hlayer({})_thres({})_epoch({})_bs({})_lr({})_dropout({})_tem({})_weight({})_zdim({})_k({})_ch({})_attn({}).pth'.format(
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
            wandb.log({
                'train_loss' : train_loss,
                'train_acc' : acc,
                'val_loss' : val_loss,
                'val_acc' : val_acc.mean(),
                'val_precision' : val_precision.mean(),
                'val_recall' : val_recall.mean(),
                'val_f1' : val_f1.mean(),
                'val_auc' : val_auc
            })
        wandb.finish()

def val_model(x, y, model, reg, paras):
    loss_MSE = nn.MSELoss()
        
    with torch.no_grad():
        model.to(paras['device'])
        model.eval()
        reg.to(paras['device'])
        reg.eval()
        
        mask_eval = []
        for i in range(len(y[0])):
            mask_eval.append([0 if np.isnan(y[1][i][j]) else 1 for j in range(len(y[1][i]))])
             
        tensor_x = [x[i].to(paras['device']) for i in range(len(x))]

        encode, concat_data = model(tensor_x)
            
        Pred_reg = reg(concat_data)
                
        P_reg = Pred_reg.detach().cpu().numpy().copy()
        GT_reg = [[0 if np.isnan(y[1][i][j]) else y[1][i][j] for j in range(len(y[1][i]))] for i in range(len(y[1]))]
        if paras['drug_std']:
            P_reg = inverse_normalize(P_reg, paras['scaler'], paras['N_type'])
        
        P = np.array([[1 if P_reg[i][j] < paras['median'][j] else 0 for j in range(len(P_reg[i]))] for i in range(len(P_reg))]) * mask_eval
        GT = [[0 if np.isnan(y[1][i][j]) else 1 if y[2][i][j] < paras['median'][j] else 0 for j in range(len(y[2][i]))] for i in range(len(y[2]))]
        P_reg = P_reg * mask_eval
        
        loss = loss_MSE(torch.FloatTensor(P_reg), torch.FloatTensor(GT_reg))
        acc, precision, recall, f1 = evaluate(P.T, np.array(GT).T, np.array(mask_eval).T)
        auc = get_auc(P.T, np.array(GT).T, np.array(mask_eval).T)
        
        return loss, acc, precision, recall, f1, auc