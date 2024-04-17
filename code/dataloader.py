import pandas as pd
import numpy as np

class StockData:

    def __init__(self, csv_paths):
        self.data = {}
        for path in csv_paths:
            stock_name = path.split('/')[-1].split('.')[0]
            self.data[stock_name] = pd.read_csv(path , dtype='unicode')
            
            self.data[stock_name].rename(columns=lambda x: x.strip(), inplace=True)
            
            self.data[stock_name] = self.data[stock_name].applymap(lambda x: x.strip() if isinstance(x, str) and x.startswith(' ') else (x.strip() if isinstance(x, str) else x))

    def normalizeCols(self,data, cols):
        temp_df = data.copy(deep = True)
        temp_df.loc[:,cols] = (temp_df.loc[:, cols] - temp_df.loc[:,cols].min())/(temp_df.loc[:, cols].max() - temp_df.loc[:,cols].min())
    
        return temp_df
    def cleanData(self):
        all_dates = []
        for stock_name, stock_data in self.data.items():

            cols = ['Open','High','Low','Close','Adj Close','Volume']
            
            for col in cols:
                stock_data[col] = pd.to_numeric(stock_data[col])

            stock_data[cols] = stock_data[cols].fillna(0)

            stock_data['Date'] = pd.to_datetime(stock_data['Date'])  


            all_dates.append(stock_data['Date'])

            self.data[stock_name] = stock_data
        sets = [set(lst) for lst in all_dates]

        included_values = set.intersection(*sets)

        for stock_name, stock_data in self.data.items():
            self.data[stock_name] = stock_data.loc[stock_data['Date'].isin(included_values),:]

    def getModelInputsAndOutputs(self):
        model_inputs = []
        model_outputs = []
        
        for stock_name, stock_data in self.data.items():

            required_columns = ['Open','High','Low','Close','Adj Close','Volume']

            stock_data[required_columns] = stock_data[required_columns].replace('', '0')

            inputs = stock_data[required_columns].values
            outputs = stock_data['Close'].values.reshape(-1, 1)

            model_inputs.append(inputs)
            model_outputs.append(outputs)
        
        max_length = max(len(arr) for arr in model_inputs)
        model_inputs_padded = [np.pad(arr, ((0, max_length - len(arr)), (0, 0)), mode='constant', constant_values=0) for arr in model_inputs]

        max_length_1 = max(len(arr) for arr in model_outputs)
        model_output_padded = [np.pad(arr, ((0, max_length_1 - len(arr)), (0, 0)), mode='constant', constant_values=0) for arr in model_outputs]


        stocks , days , input_feature_dim = np.array(model_inputs_padded).shape
        stocks , days , output_feature_dim = np.array(model_output_padded).shape

        model_inputs_padded = np.concatenate(model_inputs_padded, axis = 1).reshape(days,stocks,input_feature_dim)
        model_output_padded =np.concatenate(model_output_padded, axis = 1).reshape(days,stocks,output_feature_dim)


        return (model_inputs_padded, model_output_padded)
    
    def generateBatches(self, data, batch_size):
        # Generate batches of data
        num_batches = len(data) // batch_size
        return np.array_split(data, num_batches)
    
    def getTrainingData(self, batch_size=1):
        model_inputs, model_outputs = self.getModelInputsAndOutputs()

        input_batches = self.generateBatches(model_inputs, batch_size)
        output_batches = self.generateBatches(model_outputs, batch_size)

        input_batches_shuffled = []
        for batch in input_batches:
            shuffled_batch = np.random.permutation(batch)
            input_batches_shuffled.append(shuffled_batch)

        output_batches_shuffled = []
        for batch in output_batches:
            shuffled_batch = np.random.permutation(batch)
            output_batches_shuffled.append(shuffled_batch)

        return input_batches_shuffled, output_batches_shuffled
        
        
    
    def getTestingData(self):
        test_data = {}
        test_model_inputs = []
        test_model_outputs = []

        for stock_name, stock_data in self.data.items():
           

            test_data[stock_name] = stock_data[stock_data['Date'].dt.year.isin([2018, 2020, 2015])] 
            self.data[stock_name] = self.data[stock_name][~self.data[stock_name]['Date'].dt.year.isin([2018, 2020 , 2015])]

            required_columns = ['Open','High','Low','Close','Adj Close','Volume']

            inputs = test_data[stock_name][required_columns].values
            outputs = test_data[stock_name]['Close'].values.reshape(-1, 1)

            test_model_inputs.append(inputs)
            test_model_outputs.append(outputs)

        max_length = max(len(arr) for arr in test_model_inputs)
        test_model_inputs_padded = [np.pad(arr, ((0, max_length - len(arr)), (0, 0)), mode='constant', constant_values=0) for arr in test_model_inputs]

        max_length_1 = max(len(arr) for arr in test_model_outputs)
        test_model_output_padded = [np.pad(arr, ((0, max_length_1 - len(arr)), (0, 0)), mode='constant', constant_values=0) for arr in test_model_outputs]

        stocks , days , input_feature_dim = np.array(test_model_inputs_padded).shape
        stocks , days , output_feature_dim = np.array(test_model_output_padded).shape

        test_model_inputs_padded = np.concatenate(test_model_inputs_padded, axis = 1).reshape(days,stocks,input_feature_dim)
        test_model_output_padded =np.concatenate(test_model_output_padded, axis = 1).reshape(days,stocks,output_feature_dim)

        test_input_batches = self.generateBatches(test_model_inputs_padded, 1)
        test_output_batches = self.generateBatches(test_model_output_padded, 1)

        return (test_input_batches, test_output_batches)
