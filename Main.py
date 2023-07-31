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
from Preprocessing import data_preprocessing
from Train import train_model
from Test import test_model
import torch
import random

import argparse
import warnings
import os
        
warnings.filterwarnings('ignore')

def str2bool(v):
    if v.lower() in ['yes', '1', 'true', 't', 'y']:
        return True
    elif v.lower() in ['no', '0', 'false', 'f', 'n']:
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported Value encountered.')

parser = argparse.ArgumentParser(description = "training")

# Training parameters
parser.add_argument('--epochs', default = 200, type = int, help = 'Number of total epochs to run (default = 200)')
parser.add_argument('--batch_size', default = 2, type = int, help = 'mini-batch size (default: 2)')
parser.add_argument('--k_fold', default = 3, type = int, help = 'number of fold validation (default : 3)')
parser.add_argument('--train', default = 'True', type = str2bool, help = 'train or test')

# Dataset 
parser.add_argument('--Methylation', default = 'True', type = str2bool, help = 'using methylation dataset or not (default : True)')
parser.add_argument('--Mutation', default = 'True', type = str2bool, help = 'using Mutation dataset or not (default : True)')
parser.add_argument('--Expression', default = 'True', type = str2bool, help = 'using Expression dataset or not (default : True)')
parser.add_argument('--CNV', default = 'True', type = str2bool, help = 'using CNV dataset or not (default : True)')
parser.add_argument('--seed', default = 42, type = int, help = 'random seed (default: 42)')
parser.add_argument('--origin_data', default = 'False', type = str2bool, help = 'Using original data or not (default : False)')

# Preprocessing settings
parser.add_argument('--drug_std', default = 'False', type = str2bool, help = 'Min-Max normalize on drug dataset (default : False)')
parser.add_argument('--method', default = None, type = str, help = 'drug preprocessing method (default : None)')
parser.add_argument('--N_type', default = 1, type = int, help = 'Type of feature preprocessing (default : 1 (Min-Max)')
parser.add_argument('--thres', default = 0.3, type = float, help = 'threshold to filter missing values (default : 0.3)')

# Model parameters
parser.add_argument('--model_lr', default = 1e-6, type = float, help = 'featrue extractor initial learning rate (default = 1e-6)')
parser.add_argument('--reg_lr', default = 1e-5, type = float, help = 'regression initial learning rate (default = 1e-5)')
parser.add_argument('--dropout', default = 0.2, type = float, help = 'dropout rate in model (default = 0.2)')
parser.add_argument('--attention', default = 'True', type = str2bool, help = 'using attention module or not (default : True)')
parser.add_argument('--Weight', default = 1.0, type = float, help = 'weight of Latent alignment loss(default : 1.0)')
parser.add_argument('--temperature', default = 1.0, type = float, help = 'parameter in Latent alignment loss(default : 1.0)')
parser.add_argument('--max_hdim', default = 4096, type = int, help = 'the first number of hidden layer dimension (default : 4096)')
parser.add_argument('--h_layer', default = 2, type = int, help = 'hidden layer (default : 2)')
parser.add_argument('--k', default = 40, type = int, help = 'projected dimension of affinity matrix (default : 40)')
parser.add_argument('--z_dim', default = 1024, type = int, help = 'hidden dimension (default : 1024)')
parser.add_argument('--channel', default = 2, type = int, help = 'channels of attention map (default : 2)')
        
def main():
    global args
    args = parser.parse_args()

    if args.origin_data:
        dir_path = './data/'
    else:
        dir_path = './data/preprocessing/'
    X_dir = [dir_path + 'methylation.csv', dir_path + 'expression.csv', dir_path + 'cnv.csv', dir_path + 'mutation.csv']
    Y_dir = dir_path + 'IC50(log).csv'
        
    seed = random.randint(1, 10000) if args.seed == None else int(args.seed)
    
    X, Y, median, std, scaler = data_preprocessing(X_dir, Y_dir, args, seed)
    device = 'cuda:0' if torch.cuda.is_available()==True else 'cpu'
    
    dataset = []
    data_name = ['Me', 'Ex', 'CNV', 'Mu']
    if args.Methylation:
        dataset.append(0)
    if args.Expression:
        dataset.append(1)
    if args.CNV:
        dataset.append(2)
    if args.Mutation:
        dataset.append(3)
    
    paras = {
        'lr' : {'model' : args.model_lr, 'reg' : args.reg_lr}, 
        'median' : median, 
        'std' : std, 
        'seed' : seed, 
        'device' : device, 
        'epochs' : args.epochs, 
        'dropout' : args.dropout, 
        'batch_size' : args.batch_size, 
        'N_type' : args.N_type, 
        'k_fold' : args.k_fold, 
        'thres' : args.thres, 
        'Weight' : args.Weight,
        'method' : args.method,
        'drug_std' : args.drug_std, 
        'temperature' : args.temperature, 
        'h_layer' : args.h_layer, 
        'max_hdim' : args.max_hdim,
        'scaler' : scaler,
        'k' : args.k, 
        'z_dim' : args.z_dim, 
        'channel' : args.channel, 
        'attention' : args.attention
    }
    
    paras['name'] = '('
    for i in range(len(dataset)):
        if i == len(dataset) - 1:
            paras['name'] += data_name[dataset[i]] + ')'
        else:
            paras['name'] += data_name[dataset[i]] + ', '
    
    x_train = [X['train'][i] for i in dataset]
    x_test = [X['test'][i] for i in dataset]
    
    paras['h_dim'] = [None for i in range(len(x_train))]
    for i in range(len(x_train)):
        paras['h_dim'][i] = []
        max_dim = paras['max_hdim']
        while(1):
            if x_train[i].shape[1] > max_dim:
                paras['h_dim'][i] = [int(max_dim / pow(2, j)) for j in range(paras['h_layer'])]
                break
            elif max_dim < 1024: break
            else: max_dim = int(max_dim / 2)
        
    max_dim = paras['max_hdim']
    while(1):
        if paras['z_dim'] > max_dim:
            paras['cls_dim'] = [int(max_dim / pow(2, j)) for j in range(paras['h_layer'])]
            break
        elif max_dim < 128: break
        else: max_dim = int(max_dim / 2)
    
    if args.train:
        train_model(x_train, Y['train'], paras)
    else:
        test_model(x_test, Y['test'], paras)
    
if __name__ == '__main__':
    main()
    

