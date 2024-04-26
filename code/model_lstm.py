import torch
import torch.nn as nn
import random
import pandas as pd
import numpy as np
import torch.nn.functional as F
import math
import time
import dataloader as dataloader

if torch.cuda.is_available(): 
    dev = "cuda:0" 
else: 
    dev = "cpu" 
device = torch.device(dev)

csv_paths= [ '/local/data/sdahal_p/stock/data/stocks/ATNI.csv',
            '/local/data/sdahal_p/stock/data/stocks/ATO.csv',
            '/local/data/sdahal_p/stock/data/stocks/ATR.csv',
            '/local/data/sdahal_p/stock/data/stocks/ATRC.csv',
            '/local/data/sdahal_p/stock/data/stocks/ATRI.csv',
            '/local/data/sdahal_p/stock/data/stocks/ATRO.csv',
            '/local/data/sdahal_p/stock/data/stocks/ATRS.csv',
            '/local/data/sdahal_p/stock/data/stocks/ATSG.csv',
            '/local/data/sdahal_p/stock/data/stocks/ATV.csv',
            '/local/data/sdahal_p/stock/data/stocks/ATVI.csv',
            '/local/data/sdahal_p/stock/data/stocks/AU.csv',
            '/local/data/sdahal_p/stock/data/stocks/AUB.csv',
            '/local/data/sdahal_p/stock/data/stocks/AUBN.csv',
            '/local/data/sdahal_p/stock/data/stocks/AUDC.csv',
            '/local/data/sdahal_p/stock/data/stocks/AUTO.csv',
            '/local/data/sdahal_p/stock/data/stocks/AUY.csv',
            '/local/data/sdahal_p/stock/data/stocks/AVA.csv',
            '/local/data/sdahal_p/stock/data/stocks/AVAV.csv',#
            '/local/data/sdahal_p/stock/data/stocks/AVB.csv',
            '/local/data/sdahal_p/stock/data/stocks/AVD.csv',
            '/local/data/sdahal_p/stock/data/stocks/AVDL.csv',
            '/local/data/sdahal_p/stock/data/stocks/AVID.csv',
            '/local/data/sdahal_p/stock/data/stocks/AVK.csv',
            '/local/data/sdahal_p/stock/data/stocks/AVT.csv',
            '/local/data/sdahal_p/stock/data/stocks/AVY.csv',
            '/local/data/sdahal_p/stock/data/stocks/AWF.csv',
            '/local/data/sdahal_p/stock/data/stocks/AWRE.csv',
            '/local/data/sdahal_p/stock/data/stocks/AWX.csv',
            '/local/data/sdahal_p/stock/data/stocks/AXAS.csv',
            '/local/data/sdahal_p/stock/data/stocks/AXDX.csv',
            '/local/data/sdahal_p/stock/data/stocks/AXE.csv',
            '/local/data/sdahal_p/stock/data/stocks/AXGN.csv',
            '/local/data/sdahal_p/stock/data/stocks/AXL.csv',
            '/local/data/sdahal_p/stock/data/stocks/AXO.csv',
            '/local/data/sdahal_p/stock/data/stocks/AXS.csv',
            '/local/data/sdahal_p/stock/data/stocks/AXTI.csv',
            '/local/data/sdahal_p/stock/data/stocks/AYI.csv',
            '/local/data/sdahal_p/stock/data/stocks/AZN.csv',
            '/local/data/sdahal_p/stock/data/stocks/AXTI.csv',

            ]
            
# stock_loader = dataloader.StockData(csv_paths)

# stock_loader.cleanData()

# test_input,test_output=stock_loader.getTestingData()

# train_input , train_output = stock_loader.getTrainingData()

# print(len(train_input))
# print(len(test_input))
folder_path = '/local/data/sdahal_p/stock/data/technology/'
import os
files = os.listdir(folder_path)

csv_files = [file for file in files if file.endswith('.csv')]
csv_path = [os.path.join(folder_path, csv_file) for csv_file in csv_files]
print(len(csv_path))
stock_loader = dataloader.StockData(csv_path)

stock_loader.cleanData()

test_input,test_output=stock_loader.getTestingData()

train_input , train_output = stock_loader.getTrainingData()

print(len(train_input))
print(len(test_input))
class StockFormer(nn.Module):
    def __init__(self, encoder_input_dim=6, model_dim=512, n_output_heads=1, seq_length=63):
        super().__init__()

        self.model_dim = model_dim
        self.encoder_input_dim = encoder_input_dim
        self.n_output_heads = n_output_heads
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=self.encoder_input_dim, hidden_size=self.model_dim, batch_first=True)
        self.fc1 = nn.Linear(self.model_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)
        # self.fc = nn.Linear(self.model_dim, self.seq_length * self.n_output_heads)

    def forward(self, encoder_inputs):
        encoder_inputs = encoder_inputs.astype(np.float32)
        encoder_inputs = torch.from_numpy(encoder_inputs).to(dtype=torch.float32, device=device)

        encoder_batch_size, encoder_sequence_length, encoder_input_dim = encoder_inputs.shape

        lstm_output, _ = self.lstm(encoder_inputs)
        lstm_output = lstm_output.contiguous().view(-1, self.model_dim)

        

        # print(output.shape)
        # quit(0)
        x = self.dropout(F.relu(self.fc1(lstm_output)))
        x = self.dropout(F.relu(self.fc2(x)))
        output = self.fc3(x)
        # quit(0)
        output = output.reshape(encoder_batch_size, self.seq_length, self.n_output_heads)

        return output

stock_model = StockFormer(encoder_input_dim=6, model_dim=512, n_output_heads=1, seq_length=63)
stock_model = stock_model.to(device=device)

mean_squared_error_stock = nn.MSELoss()

# Optimizer
optimizer_stock = torch.optim.AdamW(stock_model.parameters(), lr=0.0001, weight_decay=0.0001)
scheduler_stock = torch.optim.lr_scheduler.StepLR(optimizer_stock, step_size=3, gamma=0.6, last_epoch=-1, verbose=False)

# Training
def TrainModelSP(train_inputs, train_outputs, epoch_number, final_prev):
    total_loss = 0
    total_batches = 0

    for input_index, batch_input in enumerate(train_inputs):
        total_batches += 1

        batch_size, sequence_length, feature_dim = train_outputs[input_index].shape

        optimizer_stock.zero_grad()

        output = stock_model(batch_input)

        loss = mean_squared_error_stock(output, torch.from_numpy(train_outputs[input_index]).to(dtype=torch.float32, device=device))

        loss_in_batch = loss.item() * batch_input.shape[0]
        total_loss += loss_in_batch

        loss.backward()
        optimizer_stock.step()

    print('Epoch Number: {} => Avg loss value : {} '.format(epoch_number, total_loss / (total_batches * 1)))

    return total_loss / (total_batches * 1)

# Training Section
final_prev = torch.from_numpy(np.array([])).to(dtype=torch.float32, device=device)
stock_model.train(True)

train_epoch_avg_losses_stock = []
start_time = time.time()

total_iterations = 0
loss_all_1 = []
for index in range(60):
    temp_holder_stock = list(zip(train_input, train_output))
    random.shuffle(temp_holder_stock)

    epoch_number = index + 1
       
    train_input_batches_stock, train_output_batches_stock = zip(*temp_holder_stock)
    error1 = TrainModelSP(train_input_batches_stock, train_output_batches_stock, epoch_number, final_prev)
    loss_all_1.append(error1)
   
    scheduler_stock.step()

end_time = time.time()
print("Time Elapsed", end_time - start_time)

loss_df1 = pd.DataFrame()
loss_df1['loss_main1'] = pd.Series(loss_all_1)
loss_df1.to_csv('/local/data/sdahal_p/stock/result/lstm1.csv', index=False)

# Testing
def TestStock(test_inputs, test_outputs):
    losses = []
    
    mse_stock = nn.MSELoss()

    outputs = []
    outputs_loss = []
    actual_outputs = []
    
    with torch.no_grad():
        for input_index, batch_input in enumerate(test_inputs):
            output = stock_model(batch_input)
                        
            loss = mse_stock(output, torch.from_numpy(test_outputs[input_index]).to(dtype=torch.float32, device=device))
            
            outputs.append(output.detach().cpu().numpy())
            outputs_loss.append(loss.item())
            actual_outputs.append(test_outputs[input_index])
            losses.append(loss.item())
    
    return np.array(outputs), np.array(actual_outputs), np.array(outputs_loss)

test_outputs_1_stock, test_outputs_actual_1_stock, test_losses_1_stock = TestStock(test_input, test_output)

reshaped_data_output = [item[0] for item in test_outputs_1_stock]
reshaped_data_actual = [item[0] for item in test_outputs_actual_1_stock]

df_actual = pd.DataFrame()
df_predicted = pd.DataFrame()
# print(reshaped_data_output)
# quit(0)
for i in range(46):
    column_name = f'Stock{i+1}' 
    df_predicted[column_name] = [row[i][0] for row in reshaped_data_output]
    df_actual[column_name] = [row[i][0] for row in reshaped_data_actual]

mean_actual = [df_actual[col].mean() for col in df_actual.columns]
mean_A = pd.DataFrame([mean_actual], columns=df_actual.columns)

mean_predicted = [df_predicted[col].mean() for col in df_predicted.columns]
mean_p = pd.DataFrame([mean_predicted], columns=df_predicted.columns)

final_output = pd.concat([mean_A, mean_p], axis=0)
final_output = final_output.T

final_output.to_csv('/local/data/sdahal_p/stock/result/lstm_all.csv')
