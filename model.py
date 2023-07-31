import torch 
import torch.nn as nn
from collections import OrderedDict
from torch.optim import Adam
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

class FullyConnectedLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout = 0, task = 'None'):
        """
        Parameters:
            input_dim (int)    -- Input dimension
            output_dim (int)   -- Output dimension
            dropout (float)    -- Dropout rate (default : 0)
            task (str)         -- Classification or regression (default : regerssion)
        """
        super(FullyConnectedLayer, self).__init__()
        
        # Linear
        self.fc_block = [nn.Linear(input_dim, output_dim)]

        # Dropout
        if 0 < dropout <= 1:
            self.fc_block.append(nn.Dropout(p = dropout))
        
        # Activation function
        if task == 'cls':
            self.fc_block.append(nn.Sigmoid())
        else:
            self.fc_block.append(nn.ReLU())
            
        
        self.fc_block = nn.Sequential(*self.fc_block)

    def forward(self, x):
        y = self.fc_block(x)
        return y

    
# Feature extractor
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

def cross_entropy(preds, targets, reduction = 'none'):
    """
    Parameters:
        preds (int)       -- Prediction result
        targets (list)    -- Ground truth
        reduction (int)   -- Loss recording method (default : none)
    """
    log_softmax = nn.LogSoftmax(dim = -1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()    

def cal_LA_loss(embedding1, embedding2, temperature):
    """
    Parameters:
        embedding1 (list)     -- An omics feature
        embedding2 (list)     -- Another omcis feature 
        temperature (float)   -- To adjust the range of similarity between two matrices
    """
    logits = (embedding1 @ embedding2.T) / temperature # Calculate the similarity of two matrices and use temperature to adjust the range 
    similarity1 = embedding1 @ embedding1.T # An omics similarity matrix
    similarity2 = embedding2 @ embedding2.T # Another omics similarity matrix
    targets = F.softmax((similarity1 + similarity2) / 2 * temperature, dim = -1) # Obtain the target matrix by computing two matrices
    loss1 = cross_entropy(logits, targets, reduction = 'none')
    loss2 = cross_entropy(logits.T, targets.T, reduction = 'none')
    return ((loss1 + loss2) / 2.0).mean()

def C(k):
    """
    Parameters:
        k (int)     -- Input to calculate the number of combinations
    """
    return int(k * (k - 1) / 2)

# Latent Alginment Multi-Omics Integration
class LAMOI(nn.Module):
    def __init__(self, input_dim, h_dim, z_dim, temperature, dropout, device, attention = True, Weight = 1.0, k = 40, channel = 2):
        """
        Parameters:
            input_dim (int)         -- Number of features per omics data
            h_dim (list)            -- Hidden dimension
            z_dim (list)            -- Output dimension of feature extractor
            temperature (float)     -- To adjust the range of similarity between two matrices
            dropout (float)         -- Dropout rate 
            device (float)          -- CPU or GPU
            attention (bool)        -- Using attention module or not
            Weight (float)          -- Weight of latent alignment loss
            k (float)               -- Hidden dimension of omcis weights
            channel (int)           -- Hidden dimension of affinity weights
        """
        super(LAMOI, self).__init__()
        
        self.encoder = nn.ModuleList([Encoder(input_dim[i], h_dim[i], z_dim, dropout) for i in range(len(input_dim))]) # Initial feature extractors of these omics data
        
        self.aff_weight =  nn.ParameterList([nn.Parameter(torch.randn(int(z_dim / channel), int(z_dim / channel))) for i in range(C(len(input_dim)))]) # The affinity matrix under the pairwise information in these omics data 
        self.W = nn.ParameterList([nn.Parameter(torch.randn(k, int(z_dim / channel))) for i in range(len(input_dim))]) # The weight of these omics data
        self.W_h =  nn.ParameterList([nn.ParameterList([nn.Parameter(torch.randn(k, 1)) for i in range(len(input_dim))]) for i in range(len(input_dim))]) # The weight under the pairwise these omics data 
        
        self.Weight = Weight # lambda : weight of latent alignment loss
        self.device = device
        self.temperature = temperature # Adjust hyperparameter for similarity range
        self.softmax = nn.Softmax(2)
        self.tanh = nn.Tanh()
        self.z_dim = z_dim
        self.channel = channel # Adjust hyperparameter for affinity weight
        self.att = attention # Using attention or not
        
    def forward(self, x):
        # feature extraction
        encode = [self.encoder[i](x[i]) for i in range(len(x))]
        
        concat_data = encode[0]
        for i in range(1, len(x)):
            concat_data = torch.cat((concat_data, encode[i]), 1)
        
        if self.att:
            if len(x) > 1: # The number of omics data is greater than one
                temp_encode = encode.copy()
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
        
        return encode, concat_data
                
    def loss(self, x):
        reconstruct_loss = nn.MSELoss()
        encode, concat_data = self.forward(x)
        loss_LA = 0
        
        if self.Weight > 0:
            for i in range(len(encode)):
                for j in range(i + 1, len(encode)):
                    loss_LA = loss_LA + cal_LA_loss(encode[i], encode[j], self.temperature) # Calculate the latent alignment loss between all omics
                    
            loss_LA = self.Weight * loss_LA
                    
        return loss_LA

# Predictor
class Regression(nn.Module):
    def __init__(self, n_class, latent_dim, class_dim):
        """
        Construct a multi-layer fully-connected classifier
        Parameters:
            n_class (int)         -- Ouput dimension (number of drugs)
            latent_dim (int)      -- The dimensionality of the latent space and the input layer of the classifier
            class_dim (int)       -- Hidden dimension
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
