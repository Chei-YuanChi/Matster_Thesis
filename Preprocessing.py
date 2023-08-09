# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_selection import VarianceThreshold

# Overlapping set of samples
def Overlap_samples(data_omics, data_drug):
    """
    Parameters:
        data_omics (list)       -- Multi-omics dataset
        data_drug (list)        -- Drug dataset
    """
    # Get a list of samples for the multi-omics and drug datasets
    sample_list = [data.index.to_list() for data in data_omics]
    sample_list.append(data_drug.index.to_list())

    # Turn into sets and get their intersection
    sample_set = set(sample_list[0])
    for i in range(1, len(sample_list)):
        sample_set = sample_set & set(sample_list[i])
    
    # Keep only the intersection
    for i in range(len(data_omics)):
        data_omics[i] = data_omics[i].loc[data_omics[i].index.isin(list(sample_set))].sort_index()
    data_drug = data_drug.loc[data_drug.index.isin(list(sample_set))].sort_index()
    
    return data_omics, data_drug

# Filter samples with more than "thres" missing values (Here thres = 0.3)
def Filter_data(data, thres):
    """
    Parameters:
        data (list)       -- Multi-omics dataset
        thres (float)     -- Threshold to filter missing values
    """
    data_copy = data.copy()
    for feature in data.columns:
        if data[feature].isna().sum() > len(data[feature]) * thres:
            data_copy = data_copy.drop([feature], axis = 1)
    return data_copy

# Remove features with "thres" variance (Here thres = 0)
def Delete(data, thres):
    """
    Parameters:
        data (list)       -- Multi-omics dataset
        thres (float)     -- Filter thresholds that are all the same value
    """
    variance = VarianceThreshold(threshold = thres)
    variance.fit(data)
    columns = [column for column in data.columns if column not in data.columns[variance.get_support()]]
    return data.drop(labels = columns, axis = 1) 
    

# Normalize using min-max or standardize
def Normalize(data, idx_train, N_type):
    """
    Parameters:
        data (list)       -- Drug or Multi-omics dataset
        idx_train (list)  -- List of the index of training set
    """
    scaler = {'s' : None, 'mm' : None}
    
    if N_type == 0: # standardize
        scaler['s'] = StandardScaler().fit(data.iloc[idx_train, :])
        data_df = pd.DataFrame(scaler['s'].transform(data))
    elif N_type == 1: # Normalize
        scaler['mm'] = MinMaxScaler().fit(data.iloc[idx_train, :])
        data_df = pd.DataFrame(scaler['mm'].transform(data))
    else: return data, scaler # origin
        
    data_df.columns = data.columns
    data_df.index = data.index
    return data_df, scaler

# Restore normalized data
def inverse_normalize(data, scaler, N_type):
    """
    Parameters:
        data (list)       -- Drug or multi-omics dataset
        scaler (list)     -- List of values recording the conversion process
        N_type(int)       -- Method of normalization (or standardize) 
    """
    if N_type != 0 and N_type != 1:
        return data
    s = scaler['s'] if N_type == 0 else scaler['mm']
    original_data = s.inverse_transform(data)
    return original_data

# Compensation for missing values
def Drug_preprocessing(x, drug, method, idx_train):
    """
    Parameters:
        x (list)          -- Multi-omics dataset
        drug (list)       -- Drug dataset
        method(str)       -- Method of compensation for missing values
        idx_train (list)  -- List of the index of training set
    """
    drug_pre = drug.copy()
    if method == 'mean': # use the mean of the data for imputation
        mean = np.array(drug[idx_train, :].describe().T['mean'])
        for i in range(len(drug.columns)):
            for j in range(len(drug.iloc[:, i])):
                if np.isnan(drug.iloc[j, i]): 
                    drug.iloc[j, i] = mean[i]
                    
    elif method == 'sim': # use the data of the most similar sample for imputation
        x = pd.DataFrame(np.concatenate((x[0], x[1], x[2]), axis=1))
        sim = squareform(pdist(x))
        for i in range(len(drug.columns)):
            for j in range(len(drug.iloc[:, i])):
                if np.isnan(drug.iloc[j, i]):
                    search = np.array(pd.DataFrame(sim[j], columns = ['sim']).sort_values(by = 'sim').index) # Make sure the sample's drug response is not nan
                    for k in search:
                        if not np.isnan(drug.iloc[k, i]) and k != j and k in idx_train:
                            drug.iloc[j, i] = drug.iloc[k, i]
                            break
    return drug_pre, drug


def data_preprocessing(X_dir, Y_dir, args, seed):
    """
    Parameters:
        X_dir (list)       -- The path name where the multi-omics dataset is stored
        Y_dir (str)        -- The path name where the drug dataset is stored
        args (str)         -- All arguments entered by the user or default
        seed (int)         -- Random seed
    """
    # Read csv
    data_X = [pd.read_csv(file_name).set_index('Unnamed: 0') for file_name in X_dir]
    data_Y = pd.read_csv(Y_dir).set_index('Unnamed: 0')
    
    # If the data is original data, perform preprocessing such as overlapping, filtering, and deletion
    if args.origin_data:
        data_X, data_Y = Overlap_samples(data_X, data_Y)
        data_X = [Delete(Filter_data(data_X[i], 0.3), 0) for i in range(len(data_X))]
        data_Y = Filter_data(data_Y, 0.3)
        data_X = [data_X[i].fillna(data_X[i].mean()) for i in range(len(data_X))] # Missing values are imputed as mean
        
    # Split train & test data
    index = [i for i in range(len(data_Y))]
    idx_train, idx_test, _, _ = train_test_split(index, data_Y, test_size = 0.2, random_state = seed) # The ratio of training and testing is 8:2
    
    drug_ori, drug_pro = Drug_preprocessing(data_X, data_Y, args.method, idx_train) # ori: original data, pro : data after imputation

    X = {'train' : [], 'test' : []}
    
    # Normalize
    for i in range(len(data_X)):
        std_data, _ = Normalize(data_X[i], idx_train, args.N_type)
        X['train'].append(std_data.iloc[idx_train, :])
        X['test'].append(std_data.iloc[idx_test, :])
    
    std_data, scaler = Normalize(drug_ori, idx_train, args.N_type)
    Y = {'train' :  {'pro' : drug_pro.iloc[idx_train, :], 'ori' : drug_ori.iloc[idx_train, :], 'std' : std_data.iloc[idx_train, :]},  
         'test' : {'pro' : drug_pro.iloc[idx_test, :], 'ori' : drug_ori.iloc[idx_test, :], 'std' : std_data.iloc[idx_test, :]}} # std : Normalize data
    
    return X, Y, Y['train']['ori'].describe().T['50%'], Y['train']['ori'].describe().T['std'], scaler


