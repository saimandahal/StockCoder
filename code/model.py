
import torch
import torch.nn as nn
import random

import pandas as pd
import numpy as np

import torch.nn.functional as F

import math
import time

import os

import dataloader as dataloader

if torch.cuda.is_available(): 
    dev = "cuda:0" 
else: 
    dev = "cpu" 
device = torch.device(dev)

# Data import
folder_path = '/local/data/sdahal_p/stock/data/Finance/'

files = os.listdir(folder_path)

csv_files = [file for file in files if file.endswith('.csv')]
csv_path = [os.path.join(folder_path, csv_file) for csv_file in csv_files]
print(len(csv_path))

stock_loader = dataloader.StockData(csv_path)

stock_loader.cleanData()
# Data
test_input,test_output=stock_loader.getTestingData()

train_input , train_output = stock_loader.getTrainingData()

# Positional Encoding
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos = torch.arange(max_len).unsqueeze(1).to(device=device)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)).to(device = device)
        pe = torch.zeros(max_len, 1, d_model).to(device=device)

        pe[:, 0, 0::2] = torch.sin(pos * div).to(device=device)
        pe[:, 0, 1::2] = torch.cos(pos * div).to(device=device)

        self.register_buffer('pe', pe)

    def forward(self, x):

        batch_size, seguence_length , embed_dimension = x.shape
        
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        
        seguence_length , batch_size, embed_dimension = x.shape
        
        return x.reshape(batch_size,seguence_length,embed_dimension)

# Transfomer 
class TransformerLayer(nn.Module):
   def __init__(self, dimension_model, n_heads):
        super(TransformerLayer, self).__init__()
        self.multihead_attn = MultiHeadAttention(dimension_model, n_heads)
        self.feed_forward = FeedForward(dimension_model, dimension_model * 4)
        self.layer_norm1 = nn.LayerNorm(dimension_model)
        self.layer_norm2 = nn.LayerNorm(dimension_model)
        self.dropout = nn.Dropout(0.1)
   def forward(self, x):
        # Multi-head attention
        residual = x
        x = self.layer_norm1(x + self.dropout(self.multihead_attn(x, x, x)))

        # Feed-forward
        x = self.layer_norm2(x + self.dropout(self.feed_forward(x)))

        return x

class FeedForward(nn.Module):
    def __init__(self, dimension_model, ff_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(dimension_model, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dimension_model)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dimension_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.dimension_model = dimension_model
        self.n_heads = n_heads
        self.head_dim = dimension_model // n_heads
        assert self.head_dim * n_heads == dimension_model, "Model"
        self.q_linear = nn.Linear(dimension_model, dimension_model)
        self.v_linear = nn.Linear(dimension_model, dimension_model)
        self.k_linear = nn.Linear(dimension_model, dimension_model)
        self.out = nn.Linear(dimension_model, dimension_model)

    def forward(self, query, key, value):
        batch_size = query.size(0)
        # Linear projections
        query = self.q_linear(query)
        key = self.k_linear(key)
        value = self.v_linear(value)

        query = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores, dim=-1)

        output = torch.matmul(scores, value)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.dimension_model)
        # Final linear layer
        output = self.out(output)
        return output 


class StockCoder(nn.Module):
    def __init__(self, input_dimension = 20, dimension_model = 64, n_output_heads = 1,window = 3,
                  seq_length = 10):
        super().__init__()
        
        self.dimension_model = dimension_model
        self.input_dimension = input_dimension
        self.n_output_heads = n_output_heads
        self.seq_length = seq_length
        
        # Embedding Layer.
        self.input_embed_1 = torch.nn.Linear(self.input_dimension , int(self.dimension_model/2))
        self.input_embed_2 = torch.nn.Linear(int(self.dimension_model/2) , self.dimension_model)

        self.input_dropout = torch.nn.Dropout(p = 0.10)
        
        # Positional Encoding
        self.pos_embed = PositionalEncoding(self.dimension_model,0.2)
           
        # Transformer model definition.
        self.transformer_layers = nn.ModuleList([TransformerLayer(dimension_model=self.dimension_model, n_heads=8) for _ in range(16)])
        
        # Dimension Reduction.
        
        initial_dim  = self.dimension_model
        
        self.dim_red_1 = torch.nn.Linear(initial_dim * 2 , int(initial_dim/2))
        self.dim_red_2 = torch.nn.Linear(int(initial_dim/2) , int(initial_dim/2))
        self.dim_red_3 = torch.nn.Linear(int(initial_dim/2) , int(initial_dim/4))
        self.dim_red_4 = torch.nn.Linear(int(initial_dim/4) , int(initial_dim/8))
        
        self.dim_red_dropout = torch.nn.Dropout(p = 0.05)
        
        # Final linear layer for the model.
        
        self.final_linear_1 = torch.nn.Linear(self.seq_length * int(self.dimension_model/8) ,self.seq_length * int(self.dimension_model/16))    
        self.final_linear_2 = torch.nn.Linear(self.seq_length * int(self.dimension_model/16), self.seq_length)
        
        # Activation Functions
        
        self.activation_relu = torch.nn.ReLU()
        self.activation_identity = torch.nn.Identity()
        self.activation_gelu = torch.nn.GELU()
        self.activation_tanh = torch.nn.Tanh()
        self.activation_sigmoid = torch.nn.Sigmoid()
        
        # Dropout Functions 
        
        self.dropout_5 = torch.nn.Dropout(p = 0.05)
        self.dropout_10 = torch.nn.Dropout(p = 0.10)
        self.dropout_15 = torch.nn.Dropout(p = 0.15)
        self.dropout_20 = torch.nn.Dropout(p = 0.20)
        
    def forward(self,encoder_inputs):
        
        # Converting to the torch array.
        encoder_inputs = encoder_inputs.astype(np.float32)

        encoder_inputs = torch.from_numpy(encoder_inputs).to(dtype= torch.float32,device=device)
                
        encoder_batch_size=1
        
        encoder_batch_size,encoder_sequence_length , input_dimension = encoder_inputs.shape

        embed_input_x = encoder_inputs.reshape(-1,self.input_dimension)
        
        embed_input_x = self.input_embed_1(embed_input_x)
        embed_input_x = self.activation_gelu(embed_input_x)
        
        embed_input_x = self.input_embed_2(embed_input_x)
        embed_input_x = self.activation_gelu(embed_input_x)
        
        embed_input_x = embed_input_x.reshape(encoder_batch_size, encoder_sequence_length, self.dimension_model)
        
        # Applying encoding.
        
        x = self.pos_embed(embed_input_x)

        for layer in self.transformer_layers:
            x = layer(x)
        
        x = x.reshape(-1, self.dimension_model)
        embed_input_x = embed_input_x.reshape(-1,self.dimension_model)
        
        x = torch.cat((x , embed_input_x),1)
        
        # Dim reduction layer.
        
        x = self.dim_red_1(x) 
        x= self.dropout_20(x)
        
        x = self.dim_red_2(x)
        x = self.activation_relu(x)
        x= self.dropout_20(x)

        x = self.dim_red_3(x)
        x= self.dropout_20(x)
        
        x= self.dim_red_4(x)
        x = self.activation_relu(x)
        x= self.dropout_20(x)
        
        x = x.reshape(-1, encoder_sequence_length * int(self.dimension_model/8))
        
        x= self.final_linear_1(x)
        x= self.activation_gelu(x)
        x = self.dropout_10(x)
        
        x = self.final_linear_2(x)
        x = self.activation_identity(x)
        
        x= x.reshape(encoder_batch_size , encoder_sequence_length , self.n_output_heads)
        
        return x
# Model
stock_model = StockCoder(input_dimension = 6, dimension_model = 512, n_output_heads = 1, seq_length = 125)

stock_model = stock_model.to(device = device)

mean_squared_error_stock = nn.MSELoss()

# Optimizer
optimizer_stock = torch.optim.AdamW(stock_model.parameters(), lr= 0.0001, weight_decay = 0.0001)
scheduler_stock = torch.optim.lr_scheduler.StepLR(optimizer_stock, step_size = 3 ,gamma = 0.6, last_epoch= -1, verbose=False)

# Implementation Main
def TrainStock(train_inputs, train_outputs, epoch_number):

    total_loss = 0
    total_batches = 0

    for input_index , batch_input in enumerate(train_inputs):
                
        total_batches+=1

        batch_size , sequence_length , feature_dim = train_outputs[input_index].shape  

        optimizer_stock.zero_grad()
            
        output = stock_model(batch_input)
            
        loss = mean_squared_error_stock(output, torch.from_numpy(train_outputs[input_index]).to(dtype=torch.float32,device=device))

        loss_in_batch = loss.item() * batch_input.shape[0]
            
        total_loss+=loss_in_batch

        loss.backward()

        optimizer_stock.step()

    print('Epoch Number: {} => Avg loss value : {} '.format(epoch_number, total_loss / (total_batches * 1 )))

    final_loss = total_loss / (total_batches * 1 )
    
    return final_loss


# Training Section
stock_model.train(True)


train_epoch_avg_losses_stock = []
start_time = time.time()

total_iterations=0

loss_all_1 = []
for index in range(32):

    temp_holder_stock = list(zip(train_input, train_output))
    random.shuffle(temp_holder_stock)

    epoch_number = index+ 1
       
    train_input_batches_stock, train_output_batches_stock = zip(*temp_holder_stock)
    error1= TrainStock(train_input_batches_stock, train_output_batches_stock,epoch_number )
    loss_all_1.append(error1)
   
    scheduler_stock.step()

    
end_time = time.time()
print("Time Elapsed", end_time - start_time)

loss_df1 = pd.DataFrame()

loss_df1['loss_main1'] =pd.Series(loss_all_1)

loss_df1.to_csv('/local/data/sdahal_p/stock/result/transloss4.csv', index= False)


# Testing
stock_model.eval()

def TestStock(test_inputs, test_outputs):
    losses = []
    
    mse_stock = nn.MSELoss()

    outputs = []
    outputs_loss = []
    actual_outputs = []
    
    
    with torch.no_grad():
        for input_index , batch_input in enumerate(test_inputs):
            
            output = stock_model(batch_input)
                        
            loss = mse_stock(output, torch.from_numpy(test_outputs[input_index]).to(dtype=torch.float32,device=device))
            
            outputs.append(output.detach().cpu().numpy())
            outputs_loss.append(loss.item())
            
            actual_outputs.append(test_outputs[input_index])
            
            losses.append(loss.item())
    
    return (np.array(outputs), np.array(actual_outputs), np.array(outputs_loss))


test_outputs_1_stock, test_outputs_actual_1_stock, test_losses_1_stock = TestStock(test_input,test_output)


reshaped_data_output = [item[0] for item in test_outputs_1_stock]
reshaped_data_actual = [item[0] for item in test_outputs_actual_1_stock]

df_actual = pd.DataFrame()
df_predicted = pd.DataFrame()

print(len(reshaped_data_output))
for i in range(125):
    column_name = f'Stock{i+1}' 
    df_predicted[column_name] = [row[i][0] for row in reshaped_data_output]
    df_actual[column_name] = [row[i][0] for row in reshaped_data_actual]

mean_actual = [df_actual[col].mean() for col in df_actual.columns]
mean_A = pd.DataFrame([mean_actual], columns=df_actual.columns)

mean_predicted = [df_predicted[col].mean() for col in df_predicted.columns]
mean_p = pd.DataFrame([mean_predicted], columns=df_predicted.columns)

final_output = pd.concat([mean_A, mean_p], axis=0)
final_output = final_output.T

final_output.to_csv('/local/data/sdahal_p/stock/result/trans4.csv')