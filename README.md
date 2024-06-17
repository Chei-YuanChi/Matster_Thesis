# A latent alignment-based multi-omics approach to predict drug response
![image](https://github.com/Chei-YuanChi/Matster_Thesis/assets/87289035/eca94730-6e51-4697-bbc0-8052327bb996)

# Operation method
 1. Clone from Github
```
git clone https://github.com/Chei-YuanChi/Matster_Thesis.git
```

 2. Change directory
```
cd Matster_Thesis/
```

 3. Install requirements
```
pip install -r requirements.txt
```

In Linux (non-container), the torch version that needs to modify the requirements is as follows:
```
torch==1.12.1+cu116
```

 4. Enter the file named `script.sh`.
Fill in the needed arguments.
The format of the instruction is as follows:
```
python Main.py --train True --epochs 100 ...
```

 5. Go back to the terminal and enter the following command (convenient to execute multiple times)
```
./script.sh
```
* Note: The user needs to download the dataset and model first.

6. Or the user can enter the following command directly in the terminal
```
python Main.py --train True --epochs 100 ...
```

# Dataset - Depmap
## [Custom donwload](https://depmap.org/portal/download/custom/)
* Mutation : Hotspot_Mutations
* CNV : Copy_Number_(Absolute)
* Methylation : Methylation_(1kb_upstream_TSS)
* Expression : Expression_22Q2_Public
* Drug response : Drug sensitivity IC50 (Sanger GDSC1)


# Main
## **Arguments**

> |    Name     |                                 help                                               |   default    |
> |:-----------:|:----------------------------------------------------------------------------------:|:------------:|
> |   epochs    |    Model training times                                           |     200      |
> |  model_lr   |    learning rate of feature extractor     |     1e-6     |
> |   reg_lr    |     learning rate of regression model       |     1e-5     |
> |   dropout   |     Dropout rate of feature extractor      |     0.2      |
> | batch_size  |During each training, the number of samples put in   |      2       |
> |   N_type    |Normalization used during data preprocessing  | 1 (Min-max) |
> |    seed     |The random seed used when cutting the data set  |      42      |
> |   k_fold    |Determine k to group dataset into k parts          |      3       |
> | Methylation |Using Methylation dataset or not                    |     True     |
> |  Mutation   |Using Mutation dataset or not                        |     True     |
> | Expression  |Using Expression dataset or not                     |     True     |
> |     CNV     |           Using CNV dataset or not               |     True     |
> |  drug_std   |Whether to predict a value between 0 and 1 when predicting   |    False     |
> |   method    |           Normalization used in drug response dataset             |     None     |
> |  attention  |         Using attention module or not           |     True     |
> |    thres    |Threshold for filter missing value           |     0.3      |
> |   Weight    |Weight of latent alignment loss |     1.0      |
> | temperature |When calculating the sample similarity of two features on latent space, adjust the hyperparameters of their value ranges. |     1.0      |
> |  max_hdim   |  The maximal number of neurons in the first layer in feature extractor    |     4096     |
> |   h_layer   |  Layer number of feature extractor        |      2       |
> |      k      |     hidden dimension of affinity matrix         |      40      |
> |    z_dim    |  hidden dimension of feature extractor     |     1024     |
> |   channel   |     hidden dimension in every attention map          |      2       |
> | origin_data |        Whether to use data sets that have not been pre-processed    |    False     |
> |    train    |          Is it a training phase?              |     True     |
> | get_weight  |  In test, choose whether to store the weight of each gene or omics of a drug.     |    None     |
> |  cell_line  |      cell line index of case study            |     -1      |
> | drug_index  |       Drug index of case study            |      0      |

 # Preprocessing
 ## **Process**

1. Identify samples that exist in all datasets
``` python = 
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
```

 2.Filter out data with more than 30% missing values

``` python = 
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
```
3. Delete features that have the same value across all samples

``` python = 
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
```

 4. Fill missing values with the mean of each feature

``` python =
data_X = [data_X[i].fillna(data_X[i].mean()) for i in range(len(data_X))]
```

5. Split the dataset into training and testing sets

``` python = 
# Split train & test data
    index = [i for i in range(len(data_Y))]
    idx_train, idx_test, _, _ = train_test_split(index, data_Y, test_size = 0.2, random_state = seed) # The ratio of training and testing is 8:2
```

 6. Handle missing values in the drug response concentration dataset

``` python = 
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
```

7. Normalize the data

``` python = 
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
```

## Overview
![image](https://github.com/Chei-YuanChi/Matster_Thesis/assets/87289035/61d05274-70ba-437a-8906-ff78d322c9fd)

``` python = 
def data_preprocessing(X_dir, Y_dir, args, seed):
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
         'test' : {'pro' : drug_pro.iloc[idx_test, :], 'ori' : drug_ori.iloc[idx_test, :]}, 'std' : std_data.iloc[idx_test, :]} # std : Normalize data
    
    return X, Y, Y['train']['ori'].describe().T['50%'], Y['train']['ori'].describe().T['std'], scaler
```


# Model
## **Process**


1. Use different feature extractors to extract important features from each omics dataset

``` python =
class Encoder(nn.Module):
    def __init__(self, input_dim, h_dim = [1024, 512], z_dim = 256, dropout = 0):
        """
        Parameters:
            input_dim (int)    -- Input dimension
            h_dim (list)       -- hidden dimension (default : [1024, 512])
            z_dim (int)        -- Output dimension (default : 256)
            dropout (float)    -- Dropout rate (default : 0)
        """
        super(Encoder, self).__init__()
        
        if len(h_dim) == 0:
            self.encoder = FullyConnectedLayer(input_dim, z_dim, dropout) # Only output layer
        else: 
            self.en_layers = OrderedDict()
            self.en_layers['InputLayer'] = FullyConnectedLayer(input_dim, h_dim[0], dropout)
            for num in range(1, len(h_dim)):
                self.en_layers['Layer{}'.format(num)] = FullyConnectedLayer(h_dim[num - 1], h_dim[num], dropout)
            self.en_layers['OutputLayer'] = FullyConnectedLayer(h_dim[-1], z_dim, dropout)
            self.encoder = nn.Sequential(self.en_layers)
        
    def forward(self, x):
        encode = self.encoder(x)
        return encode
```
2. Use latent alignment to adjust the feature extractor, aligning features from different omics in the latent space

![image](https://github.com/Chei-YuanChi/Matster_Thesis/assets/87289035/d6d10a2c-03aa-431f-90d7-4eeb185d9b08)


``` python = 
def loss(self, x):
    encode, concat_data = self.forward(x)
    loss_LA = 0
        
    if self.Weight > 0:
        for i in range(len(encode)):
            for j in range(i + 1, len(encode)):
                loss_LA = loss_LA + cal_LA_loss(encode[i], encode[j], self.temperature) # Calculate the latent alignment loss between all omics
                    
        loss_LA = self.Weight * loss_LA
                    
    return loss_LA
```

```python = 
def cal_LA_loss(embedding1, embedding2, temperature):
    logits = (embedding1 @ embedding2.T) / temperature # Calculate the similarity of two matrices and use temperature to adjust the range 
    similarity1 = embedding1 @ embedding1.T # An omics similarity matrix
    similarity2 = embedding2 @ embedding2.T # Another omics similarity matrix
    targets = F.softmax((similarity1 + similarity2) / 2 * temperature, dim = -1) # Obtain the target matrix by computing two matrices
    loss1 = cross_entropy(logits, targets, reduction = 'none')
    loss2 = cross_entropy(logits.T, targets.T, reduction = 'none')
    return ((loss1 + loss2) / 2.0).mean()
```

3. Learn the correlation information between different omics data

![image](https://github.com/Chei-YuanChi/Matster_Thesis/assets/87289035/82196026-f1b8-4fc1-b0cf-8a5bd08fb70c)

``` python = 
# Divide the features on the latent space into N blocks
for i in range(len(temp_encode)):
    temp_encode[i] = torch.unsqueeze(temp_encode[i], 2).reshape(-1, int(self.z_dim / self.channel), self.channel)
# Initialize
affinity_matrix, H, A, Att_encode = [], [], [], []
for i in range(len(temp_encode)):
    index = 0
    affinity_matrix.append([])
    H.append([])
    A.append([])
    Att_encode.append([])
    for j in range(len(temp_encode)):
        if i <= j:
            affinity_matrix[i].append(torch.swapaxes(temp_encode[i], 1, 2) @ self.aff_weight[index] @ temp_encode[j]) # Use the aff_weight to learn the information between two omcis
            H[i].append(self.tanh(self.W[j] @ temp_encode[j] + self.W[i] @ temp_encode[i] @ affinity_matrix[i][j])) # Use W to weight all omics and add up
            if i < j: index += 1
        else:
            affinity_matrix[i].append(torch.swapaxes(affinity_matrix[j][i], 1, 2))
            H[i].append(self.tanh(self.W[i] @ temp_encode[i] + self.W[j] @ temp_encode[j] @ affinity_matrix[i][j]))
        A[i].append(self.softmax(self.W_h[i][j].T @ H[i][j])) # Use softmax to get the omics weights containing all omics data information
        if i != j:
            Att_encode[i].append(A[i][j] * temp_encode[i]) # Multiply the attention weight back to each physical data

concat_data = Att_encode[0][0].reshape(-1, self.z_dim)
for i in range(len(Att_encode)):
    for j in range(len(Att_encode[i])):
        if i != 0 or j != 0:
            concat_data = torch.cat((concat_data, Att_encode[i][j].reshape(-1, self.z_dim)), 1)
```

4. Predict drug response concentration

![image](https://github.com/Chei-YuanChi/Matster_Thesis/assets/87289035/f46399f6-0dc3-4dcb-8931-efa9da0549c0)

``` python = 
class Regression(nn.Module):
    def __init__(self, n_class, latent_dim, class_dim):
        """
        Construct a multi-layer fully-connected classifier
        Parameters:
            n_class (int)         -- the number of class
            latent_dim (int)      -- the dimensionality of the latent space and the input layer of the classifier
            class_dim (int)       -- the dimensionality of the hidden layer
        """
        super(Regression, self).__init__()
        
        self.reg = nn.Sequential(
            nn.Linear(latent_dim, class_dim[0]),
            nn.ReLU(),
            nn.Linear(class_dim[0], class_dim[1]),
            nn.ReLU(),
            nn.Linear(class_dim[1], n_class),
        )
        
    def forward(self, x):
        return self.reg(x)
```
# Case Study
 1. Conduct test

```
python Main.py --train False
```

2. Obtain the importance of each omics data in the prediction

```
python Main.py --train False --get_weight omics
```

 3. Obtain the importance of a specific cell line's genes for the prediction of individual drugs

```
python Main.py --train False --get_weight omics --cell_line 25(cell line index) --drug_index 26(drug index)
```

 * After obtaining the importance of a specific cell line's genes for the prediction of individual drugs, the top 2000 genes need to be analyzed using the [reactome](https://reactome.org/) website. The process is as follows:

![1](https://github.com/Chei-YuanChi/Matster_Thesis/assets/87289035/19777b76-bb67-49fc-9653-cebe258ecfa0)

![2](https://github.com/Chei-YuanChi/Matster_Thesis/assets/87289035/b9381766-f4f8-4f3a-8ad4-24ca394de3b0)

 4.Enter the names of the top 2000 genes

![345](https://github.com/Chei-YuanChi/Matster_Thesis/assets/87289035/fbbe754d-20e1-4b35-8d01-d8d5b88f9ad1)

![6](https://github.com/Chei-YuanChi/Matster_Thesis/assets/87289035/d2d62e0c-2e74-4ad8-9721-d661c4d06ae3)

![7](https://github.com/Chei-YuanChi/Matster_Thesis/assets/87289035/d6f54a09-f821-41df-981f-e2fc0935af5f)

 8. Enter the drug name (example: panobinostat)

![8](https://github.com/Chei-YuanChi/Matster_Thesis/assets/87289035/473ed234-4456-494e-95b3-9c09a7929675)

![9](https://github.com/Chei-YuanChi/Matster_Thesis/assets/87289035/cd62456a-5e06-4af0-82bc-e77491411a1c)

10. You can see the full view of the corresponding pathways

![10](https://github.com/Chei-YuanChi/Matster_Thesis/assets/87289035/cb62c0fd-4a45-46c6-930b-c13cae33e5ba)

![11-12](https://github.com/Chei-YuanChi/Matster_Thesis/assets/87289035/63359561-d144-44ba-b096-7618a87d50cf)

13. After downloading the file, find the corresponding pathway name and Entities pValue to proceed with the analysis

