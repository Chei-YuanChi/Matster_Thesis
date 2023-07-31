

# A latent alignment-based multi-omics approach to predict drug response
![image](https://github.com/Chei-YuanChi/Matster_Thesis/assets/87289035/22c84fc7-7275-4b0e-9080-3437f0edc210)


# Operation method
1. 進入到 script.sh檔案
輸入需要調整的 argument
格式如下
```
python Main.py --train True --epochs 100 ...
```
2. 回到 terminal 輸入以下指令即可 (方便執行多次)
```
./script.sh
```

3. 或是直接在 terminal 輸入以下指令也可
```
python Main.py --train True --epochs 100 ...
```

# Dataset - Depmap
## [Custom donwload](https://depmap.org/portal/download/custom/)
* Mutation : Hotspot_Mutations
* CNV : Copy_Number_(Absolute)
* Methylation : Methylation_(1kb_upstream_TSS)
* Expression : Expression_22Q2_Public
* Drug response : Drug_sensitivity_(PRISM_Repurposing_Primary_Screen)_19Q4


# Main
## **Arguments**

|    Name     |                                 help                                 |   default    |
|:-----------:|:--------------------------------------------------------------------:|:------------:|
|   epochs    |                             模型訓練次數                             |     200      |
|  model_lr   |                  feature extractor 的 learning rate                  |     1e-6     |
|   reg_lr    |                  regression model 的 learning rate                   |     1e-5     |
|   dropout   |                  feature extractor 的 Dropout rate                   |     0.2      |
| batch_size  |                     每一次訓練時，放入的樣本數量                     |      2       |
|   N_type    |                  資料前處理時，使用的 normalization                  | 1 (Min-max) |
|    seed     |                   切割資料集時所使用的random seed                    |      42      |
|   k_fold    |                     k-fold validation 的切割數量                     |      3       |
| Methylation |                        使用 Methylation 資料集與否                         |     True     |
|  Mutation   |                          使用 Mutation 資料集與否                          |     True     |
| Expression  |                         使用 Expression 資料集與否                         |     True     |
|     CNV     |                            使用 CNV 資料集與否                             |     True     |
|  drug_std   |                    預測時是否為預測 0 ~ 1 間的值                     |    False     |
|   method    |                  藥物反應資料集使用的Normalization                   |     None     |
|  attention  |                       是否使用attention module                       |     True     |
|    thres    |                            過濾缺失值閥值                            |     0.3      |
|   Weight    |                     latent alignment loss 的權重                     |     1.0      |
| temperature | 計算兩個特徵在 latent space 上的樣本相似度時，調整其值域範圍的超參數 |     1.0      |
|  max_hdim   |               feature extractor 第一層最大的神經元數量               |     4096     |
|   h_layer   |                       feature extractor 的層數                       |      2       |
|      k      |                 affinity matrix 的 hidden dimension                  |      40      |
|    z_dim    |                feature extractor 的 hidden dimension                 |     1024     |
|   channel   |               每一個 attention map 的 hidden dimension               |      2       |
| origin_data |                    是否使用尚未經過前處理的資料集                    |    False     |
|    train    |                            是否為訓練階段                            |     True     |


# Preprocessing
## **流程**
1. 找出所有資料集中皆存在的樣本
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

2. 過濾 30% 缺失值以上資料

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
3. 刪除全為同一個值的特徵

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

4. 缺失值補用各特徵平均值補值
``` python =
data_X = [data_X[i].fillna(data_X[i].mean()) for i in range(len(data_X))]
```

5. 切分訓練與測試資料集

``` python = 
# Split train & test data
    index = [i for i in range(len(data_Y))]
    idx_train, idx_test, _, _ = train_test_split(index, data_Y, test_size = 0.2, random_state = seed) # The ratio of training and testing is 8:2
```

6. 處理藥物反應濃度資料集之缺失值

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

7. 對資料進行 Normalize

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
## **流程**
1. 利用不同的 feature extractor 來提取各個體學的重要特徵
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
2. 使用 latent alignment 調整 feature extractor 使不同體學特徵在latent space上對齊

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

3. 學習不同體學資料間的關聯性資訊
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

4. 預測藥物反應濃度
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
