#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np

import os
import sys

import math

class TemporalDataLoader:
    
    def normalizeCols(self,data, cols):
        temp_df = data.copy(deep = True)
        temp_df.loc[:,cols] = (temp_df.loc[:, cols] - temp_df.loc[:,cols].min())/(temp_df.loc[:, cols].max() - temp_df.loc[:,cols].min())
    
        return temp_df
    
    def cleanDataJunk(self):
        lists = [self.VE_19_daily['Date'].values,self.VE_37_daily['Date'].values,self.Precip_daily['Date'].values,
                self.Max_temp_daily['Date'].values,self.Min_temp_daily['Date'].values,self.Obs_temp_daily['Date'].values,
                self.TB_diff_daily['Date'].values,self.SWE_values_data['Date'].values]
        
        sets = [set(lst) for lst in lists]
        
        included_values = set.intersection(*sets)
                
        self.VE_19_daily = self.VE_19_daily.loc[self.VE_19_daily['Date'].isin(included_values),:] 
        self.VE_19_data = self.VE_19_data.loc[self.VE_19_data['Date'].isin(included_values),:]
        self.VE_19_data_H_3 = self.VE_19_data_H_3.loc[self.VE_19_data_H_3['Date'].isin(included_values),:]
        
        self.VE_37_daily = self.VE_37_daily.loc[self.VE_37_daily['Date'].isin(included_values),:] 
        self.VE_37_data = self.VE_37_data.loc[self.VE_37_data['Date'].isin(included_values),:]
        self.VE_37_data_H_3 = self.VE_37_data_H_3.loc[self.VE_37_data_H_3['Date'].isin(included_values),:]
        
        self.TB_diff_daily = self.TB_diff_daily.loc[self.TB_diff_daily['Date'].isin(included_values),:] 
        self.TB_diff_data = self.TB_diff_data.loc[self.TB_diff_data['Date'].isin(included_values),:]
        self.TB_diff_data_H_3 = self.TB_diff_data_H_3.loc[self.TB_diff_data_H_3['Date'].isin(included_values),:]

        self.Precip_daily = self.Precip_daily.loc[self.Precip_daily['Date'].isin(included_values),:]
        self.Precip_data_avg = self.Precip_data_avg.loc[self.Precip_data_avg['Date'].isin(included_values),:]
        self.Precip_data_data_avg_H_3 = self.Precip_data_data_avg_H_3.loc[self.Precip_data_data_avg_H_3['Date'].isin(included_values),:]

        self.Max_temp_daily = self.Max_temp_daily.loc[self.Max_temp_daily['Date'].isin(included_values),:]
        self.Max_temp_data_avg = self.Max_temp_data_avg.loc[self.Max_temp_data_avg['Date'].isin(included_values),:]
        self.Max_temp_data_avg_H_3 = self.Max_temp_data_avg_H_3.loc[self.Max_temp_data_avg_H_3['Date'].isin(included_values),:]

        self.Min_temp_daily = self.Min_temp_daily.loc[self.Min_temp_daily['Date'].isin(included_values),:]
        self.Min_temp_data_avg = self.Min_temp_data_avg.loc[self.Min_temp_data_avg['Date'].isin(included_values),:]
        self.Min_temp_data_avg_H_3 = self.Min_temp_data_avg_H_3.loc[self.Min_temp_data_avg_H_3['Date'].isin(included_values),:]

        self.Obs_temp_daily = self.Obs_temp_daily.loc[self.Obs_temp_daily['Date'].isin(included_values),:]
        self.Obs_temp_data_avg = self.Obs_temp_data_avg.loc[self.Obs_temp_data_avg['Date'].isin(included_values),:]
        self.Obs_temp_data_avg_H_3 = self.Obs_temp_data_avg_H_3.loc[self.Obs_temp_data_avg_H_3['Date'].isin(included_values),:]

        self.SWE_values_data = self.SWE_values_data.loc[self.SWE_values_data['Date'].isin(included_values),:]

        self.dates = np.array(list(self.VE_19_data_H_3['Date'].values))

    def getModelInputsAndOutputs(self):
        model_inputs = []
        model_outputs = []

        for index,location_info in self.snotel_locations_info.iterrows():
    
            elev = (location_info['Elevation_x'] - self.snotel_locations_info['Elevation_x'].mean()) / (self.snotel_locations_info['Elevation_x'].std())
            lat = (location_info['Latitude_x'] - self.snotel_locations_info['Latitude_x'].mean()) / (self.snotel_locations_info['Latitude_x'].std())
            long = (location_info['Longitude_x'] - self.snotel_locations_info['Longitude_x'].mean()) / (self.snotel_locations_info['Longitude_x'].std())
            southness = (location_info['Southness'] - self.snotel_locations_info['Southness'].mean()) / (self.snotel_locations_info['Southness'].std())
            landCover = long = (location_info['LandCover'] - self.snotel_locations_info['LandCover'].mean()) / (self.snotel_locations_info['LandCover'].std())
            spatial_feature = np.array([elev, lat , long, southness , landCover])

            self.daily_info_VE_37 = self.VE_37_daily.loc[:, self.VE_37_daily.columns.isin([location_info['Station Name']])].T.to_numpy()[0].reshape(-1,1)
            self.daily_info_VE_19 = self.VE_19_daily.loc[:, self.VE_19_daily.columns.isin([location_info['Station Name']])].T.to_numpy()[0].reshape(-1,1)
            self.daily_info_TB_diff = self.TB_diff_daily.loc[:, self.TB_diff_daily.columns.isin([location_info['Station Name']])].T.to_numpy()[0].reshape(-1,1)

            self.daily_info_precip = self.Precip_daily.loc[:, self.Precip_daily.columns.isin([location_info['Station Name']])].T.to_numpy()[0].reshape(-1,1)
            self.daily_info_max_temp = self.Max_temp_daily.loc[:, self.Max_temp_daily.columns.isin([location_info['Station Name']])].T.to_numpy()[0].reshape(-1,1)
            self.daily_info_min_temp = self.Min_temp_daily.loc[:, self.Min_temp_daily.columns.isin([location_info['Station Name']])].T.to_numpy()[0].reshape(-1,1)
            self.daily_info_obs_temp = self.Obs_temp_daily.loc[:, self.Obs_temp_daily.columns.isin([location_info['Station Name']])].T.to_numpy()[0].reshape(-1,1)


            self.historial_info_VE_37 = self.VE_37_data_H_3.loc[:, self.VE_37_data_H_3.columns.isin([location_info['Station Name']])].T.to_numpy()[0].reshape(-1,1)
            self.historial_info_VE_19 = self.VE_19_data_H_3.loc[:, self.VE_19_data_H_3.columns.isin([location_info['Station Name']])].T.to_numpy()[0].reshape(-1,1)
            self.historial_info_TB_diff = self.TB_diff_data_H_3.loc[:, self.TB_diff_data_H_3.columns.isin([location_info['Station Name']])].T.to_numpy()[0].reshape(-1,1)

            self.historial_info_precip = self.Precip_data_data_avg_H_3.loc[:, self.Precip_data_data_avg_H_3.columns.isin([location_info['Station Name']])].T.to_numpy()[0].reshape(-1,1)
            self.historial_info_max_temp = self.Max_temp_data_avg_H_3.loc[:, self.Max_temp_data_avg_H_3.columns.isin([location_info['Station Name']])].T.to_numpy()[0].reshape(-1,1)
            self.historial_info_min_temp = self.Min_temp_data_avg_H_3.loc[:, self.Min_temp_data_avg_H_3.columns.isin([location_info['Station Name']])].T.to_numpy()[0].reshape(-1,1)
            self.historial_info_obs_temp = self.Obs_temp_data_avg_H_3.loc[:, self.Obs_temp_data_avg_H_3.columns.isin([location_info['Station Name']])].T.to_numpy()[0].reshape(-1,1)

            merged_values = np.concatenate((self.daily_info_VE_37,self.daily_info_VE_19,self.daily_info_TB_diff,self.daily_info_precip,
                                            self.daily_info_max_temp,self.daily_info_min_temp,self.daily_info_obs_temp,
                                            self.historial_info_VE_37,self.historial_info_VE_19,self.historial_info_TB_diff,self.historial_info_precip,
                                           self.historial_info_max_temp,self.historial_info_min_temp,self.historial_info_obs_temp),axis = 1)

            merged_values = np.insert(merged_values,[0,1,2,3,4],spatial_feature, axis= 1)

            location_swe_values = self.SWE_values_data.loc[:,self.SWE_values_data.columns.isin([location_info['Station Name']])].T.to_numpy()[0].reshape(-1,1)

            model_outputs.append(location_swe_values)
            model_inputs.append(merged_values)
    
    
        locations , days , input_feature_dim = np.array(model_inputs).shape
        locations , days , output_feature_dim = np.array(model_outputs).shape

        model_inputs = np.concatenate(model_inputs, axis = 1).reshape(days,locations,input_feature_dim)
        model_outputs =np.concatenate(model_outputs, axis = 1).reshape(days,locations,output_feature_dim)
        
        return (model_inputs,model_outputs)
    
    def getFilteredDateIndices(self):
        filtered_dates = []
        filtered_dates_indices = []
        
        years = [value for value in range(2001,2019)]

        for year in years:
            indices = np.where(np.logical_and(self.dates >= '{}-10-01'.format(year) , self.dates<='{}-07-01'.format(year + 1)))[0]
    
            filtered_dates.append(self.dates[indices[:270]])
            filtered_dates_indices.append(indices[:270])
    

        filtered_dates = np.concatenate(filtered_dates)
        filtered_dates_indices = np.concatenate(filtered_dates_indices)
            
        return (filtered_dates,filtered_dates_indices)
    
    def generateBatches(self,data,batch_size,seq_length = 270):
        start_index = 0
    
        splitted_data = []
    
        batch_data = []
    
        while start_index < len(data):
            current_split = data[start_index: (start_index + seq_length)]
            start_index = start_index + seq_length
        
            splitted_data.append(current_split)
        
        num_batches = int((len(data)/seq_length)/batch_size)
    
        batch_start = 0
        while batch_start < len(splitted_data):
            batch_data.append(np.array(splitted_data[batch_start:batch_start+batch_size]))
            batch_start = batch_start+batch_size
            
        return batch_data
 
    
    def getTrainingData(self):
        filtered_dates,filtered_dates_indices = self.getFilteredDateIndices()
        
        test_yr_indices_1 = np.where(np.logical_and(filtered_dates >= "{}-10-01".format(self.year_1) , filtered_dates<= "{}-07-01".format(self.year_1+1)))[0]
        test_yr_indices_2 = np.where(np.logical_and(filtered_dates >= "{}-10-01".format(self.year_2) , filtered_dates<= "{}-07-01".format(self.year_2+1)))[0]
        test_yr_indices_3 = np.where(np.logical_and(filtered_dates >= "{}-10-01".format(self.year_3) , filtered_dates<= "{}-07-01".format(self.year_3+1)))[0]
        test_yr_indices_4 = np.where(np.logical_and(filtered_dates >= "{}-10-01".format(self.year_4) , filtered_dates<= "{}-07-01".format(self.year_4+1)))[0]
        test_yr_indices_5 = np.where(np.logical_and(filtered_dates >= "{}-10-01".format(self.year_5) , filtered_dates<= "{}-07-01".format(self.year_5+1)))[0]
        
        train_yr_indices = np.array(list(set([index for index in range(len(filtered_dates))]).difference(np.concatenate((test_yr_indices_1,test_yr_indices_2,test_yr_indices_3,test_yr_indices_4,test_yr_indices_5)))))
        
        train_inputs = self.model_inputs[filtered_dates_indices[train_yr_indices]]
        train_outputs = self.model_outputs[filtered_dates_indices[train_yr_indices]]
        
        days,locations,input_feature_dim = train_inputs.shape
        days,locations,output_feature_dim = train_outputs.shape

        training_data_input = []
        training_data_output = []

        for index in range(len(self.snotel_locations_info['Station Name'])):
            batch_input = self.generateBatches(train_inputs[:,index,:],1)
            batch_output = self.generateBatches(train_outputs[:,index,:], 1)
        
            training_data_input.append(batch_input)
            training_data_output.append(batch_output)
        
        training_data_input = np.array(training_data_input).reshape(-1,1,270,19)
        training_data_output = np.array(training_data_output).reshape(-1,1,270,1)

        return (training_data_input,training_data_output)
    
    def getTestingData(self):
        filtered_dates,filtered_dates_indices = self.getFilteredDateIndices()
        
        test_yr_indices_1 = np.where(np.logical_and(filtered_dates >= "{}-10-01".format(self.year_1) , filtered_dates<= "{}-07-01".format(self.year_1+1)))[0]
        test_yr_indices_2 = np.where(np.logical_and(filtered_dates >= "{}-10-01".format(self.year_2) , filtered_dates<= "{}-07-01".format(self.year_2+1)))[0]
        test_yr_indices_3 = np.where(np.logical_and(filtered_dates >= "{}-10-01".format(self.year_3) , filtered_dates<= "{}-07-01".format(self.year_3+1)))[0]
        test_yr_indices_4 = np.where(np.logical_and(filtered_dates >= "{}-10-01".format(self.year_4) , filtered_dates<= "{}-07-01".format(self.year_4+1)))[0]
        test_yr_indices_5 = np.where(np.logical_and(filtered_dates >= "{}-10-01".format(self.year_5) , filtered_dates<= "{}-07-01".format(self.year_5+1)))[0]
        
        test_inputs_1 = self.model_inputs[filtered_dates_indices[test_yr_indices_1]]
        test_outputs_1 = self.model_outputs[filtered_dates_indices[test_yr_indices_1]]
        
        test_inputs_2 = self.model_inputs[filtered_dates_indices[test_yr_indices_2]]
        test_outputs_2 = self.model_outputs[filtered_dates_indices[test_yr_indices_2]]
        
        test_inputs_3 = self.model_inputs[filtered_dates_indices[test_yr_indices_3]]
        test_outputs_3 = self.model_outputs[filtered_dates_indices[test_yr_indices_3]]
        
        test_inputs_4 = self.model_inputs[filtered_dates_indices[test_yr_indices_4]]
        test_outputs_4 = self.model_outputs[filtered_dates_indices[test_yr_indices_4]]
        
        test_inputs_5 = self.model_inputs[filtered_dates_indices[test_yr_indices_5]]
        test_outputs_5 = self.model_outputs[filtered_dates_indices[test_yr_indices_5]]
        
        testing_data_input_1 = []
        testing_data_output_1 = []

        testing_data_input_2 = []
        testing_data_output_2 = []

        testing_data_input_3 = []
        testing_data_output_3 = []

        testing_data_input_4 = []
        testing_data_output_4 = []

        testing_data_input_5 = []
        testing_data_output_5 = []

        for index in range(len(self.snotel_locations_info['Station Name'])):
            batch_input_1 = self.generateBatches(test_inputs_1[:,index,:],1)    
            batch_output_1 = self.generateBatches(test_outputs_1[:,index,:], 1)
    
            batch_input_2 = self.generateBatches(test_inputs_2[:,index,:],1)
            batch_output_2 = self.generateBatches(test_outputs_2[:,index,:], 1)
    
            batch_input_3 = self.generateBatches(test_inputs_3[:,index,:],1)
            batch_output_3 = self.generateBatches(test_outputs_3[:,index,:], 1)
    
            batch_input_4 = self.generateBatches(test_inputs_4[:,index,:],1)
            batch_output_4 = self.generateBatches(test_outputs_4[:,index,:], 1)
    
            batch_input_5 = self.generateBatches(test_inputs_5[:,index,:],1)
            batch_output_5 = self.generateBatches(test_outputs_5[:,index,:], 1)
    
            testing_data_input_1.append(batch_input_1)
            testing_data_output_1.append(batch_output_1)
    
            testing_data_input_2.append(batch_input_2)
            testing_data_output_2.append(batch_output_2)
    
            testing_data_input_3.append(batch_input_3)
            testing_data_output_3.append(batch_output_3)
    
            testing_data_input_4.append(batch_input_4)
            testing_data_output_4.append(batch_output_4)
    
            testing_data_input_5.append(batch_input_5)
            testing_data_output_5.append(batch_output_5)
        
        testing_data_input_1 = np.array(testing_data_input_1).reshape(-1,1,270,19)
        testing_data_output_1 = np.array(testing_data_output_1).reshape(-1,1,270,1)

        testing_data_input_2 = np.array(testing_data_input_2).reshape(-1,1,270,19)
        testing_data_output_2 = np.array(testing_data_output_2).reshape(-1,1,270,1)

        testing_data_input_3 = np.array(testing_data_input_3).reshape(-1,1,270,19)
        testing_data_output_3 = np.array(testing_data_output_3).reshape(-1,1,270,1)

        testing_data_input_4 = np.array(testing_data_input_4).reshape(-1,1,270,19)
        testing_data_output_4 = np.array(testing_data_output_4).reshape(-1,1,270,1)

        testing_data_input_5 = np.array(testing_data_input_5).reshape(-1,1,270,19)
        testing_data_output_5 = np.array(testing_data_output_5).reshape(-1,1,270,1)
        
        return ((testing_data_input_1,testing_data_output_1),(testing_data_input_2,testing_data_output_2),(testing_data_input_3,testing_data_output_3),(testing_data_input_4,testing_data_output_4),(testing_data_input_5,testing_data_output_5))

    def prepareInputsAndOutputs(self):
        self.model_inputs, self.model_outputs = self.getModelInputsAndOutputs()

    def __init__(self):
        self.VE_37_daily = pd.read_csv('/local/data/sdahal_p/data/Data/VE_37_Collection.csv')
        self.VE_19_daily = pd.read_csv('/local/data/sdahal_p/data/Data/VE_19_Collection.csv')
        self.TB_diff_daily = pd.read_csv('/local/data/sdahal_p/data/Data/TB_Diff_Collection.csv')

        self.Precip_daily = pd.read_csv('/local/data/sdahal_p/data/Data/Precipitation_Collection.csv')
        self.Max_temp_daily = pd.read_csv('/local/data/sdahal_p/data/Data/Max_Temp_Collection.csv')
        self.Min_temp_daily = pd.read_csv('/local/data/sdahal_p/data/Data/Min_Temp_Collection.csv')
        self.Obs_temp_daily = pd.read_csv('/local/data/sdahal_p/data/Data/Obs_Temp_Collection.csv')
        
        self.snotel_locations_info = pd.read_csv('/local/data/sdahal_p/data/Data/Snotel_Locations_Filtered_v3.csv')
        self.snotel_locations_info['Southness'] = [ math.sin(value['Slope_tif1_x']) * math.cos(value['Aspect_tif_x']) for index,value in self.snotel_locations_info.iterrows()]
        
        self.VE_37_daily = self.normalizeCols(self.VE_37_daily , list(self.snotel_locations_info['Station Name'].values))
        self.VE_19_daily = self.normalizeCols(self.VE_19_daily, list(self.snotel_locations_info['Station Name'].values))
        self.TB_diff_daily = self.normalizeCols(self.TB_diff_daily, list(self.snotel_locations_info['Station Name'].values))

        self.Precip_daily = self.normalizeCols(self.Precip_daily, list(self.snotel_locations_info['Station Name'].values))
        self.Max_temp_daily = self.normalizeCols(self.Max_temp_daily, list(self.snotel_locations_info['Station Name'].values))
        self.Min_temp_daily = self.normalizeCols(self.Min_temp_daily, list(self.snotel_locations_info['Station Name'].values))
        self.Obs_temp_daily = self.normalizeCols(self.Obs_temp_daily, list(self.snotel_locations_info['Station Name'].values))
        
        # Averaged data of the snotel locations.

        self.VE_37_data = pd.read_csv('/local/data/sdahal_p/data/Data/VE_37_data_avg.csv')
        self.VE_37_data = self.normalizeCols(self.VE_37_data , list(self.snotel_locations_info['Station Name'].values))
        
        self.VE_19_data = pd.read_csv('/local/data/sdahal_p/data/Data/VE_19_data_avg.csv')
        self.VE_19_data = self.normalizeCols(self.VE_19_data , list(self.snotel_locations_info['Station Name'].values))
        
        self.TB_diff_data = pd.read_csv('/local/data/sdahal_p/data/Data/TB_diff_data_avg.csv')
        self.TB_diff_data = self.normalizeCols(self.TB_diff_data , list(self.snotel_locations_info['Station Name'].values))
        

        self.Precip_data_avg = pd.read_csv('/local/data/sdahal_p/data/Data/Precipitation_data_avg.csv')
        self.Precip_data_avg = self.normalizeCols(self.Precip_data_avg , list(self.snotel_locations_info['Station Name'].values))
        
        self.Max_temp_data_avg = pd.read_csv('/local/data/sdahal_p/data/Data/Max_temp_data_avg.csv')
        self.Max_temp_data_avg = self.normalizeCols(self.Max_temp_data_avg , list(self.snotel_locations_info['Station Name'].values))
        
        
        self.Min_temp_data_avg = pd.read_csv('/local/data/sdahal_p/data/Data/Min_temp_data_avg.csv')
        self.Min_temp_data_avg = self.normalizeCols(self.Min_temp_data_avg , list(self.snotel_locations_info['Station Name'].values))
        
        self.Obs_temp_data_avg = pd.read_csv('/local/data/sdahal_p/data/Data/Obs_temp_data_avg.csv')
        self.Obs_temp_data_avg = self.normalizeCols(self.Obs_temp_data_avg , list(self.snotel_locations_info['Station Name'].values))
        
        
        # Historical data of the snotel locations.

        self.VE_37_data_H_3 = pd.read_csv('/local/data/sdahal_p/data/Data/VE_37_data_avg_H_3.csv')
        self.VE_37_data_H_3 = self.normalizeCols(self.VE_37_data_H_3 , list(self.snotel_locations_info['Station Name'].values))
        
        self.VE_19_data_H_3 = pd.read_csv('/local/data/sdahal_p/data/Data/VE_19_data_avg_H_3.csv')
        self.VE_19_data_H_3 = self.normalizeCols(self.VE_19_data_H_3 , list(self.snotel_locations_info['Station Name'].values))
        
        self.TB_diff_data_H_3 = pd.read_csv('/local/data/sdahal_p/data/Data/TB_Diff_data_avg_H_3.csv')
        self.TB_diff_data_H_3 = self.normalizeCols(self.TB_diff_data_H_3 , list(self.snotel_locations_info['Station Name'].values))
        

        self.Precip_data_data_avg_H_3 = pd.read_csv('/local/data/sdahal_p/data/Data/Precip_data_avg_H_3.csv')
        self.Precip_data_data_avg_H_3 = self.normalizeCols(self.Precip_data_data_avg_H_3 , list(self.snotel_locations_info['Station Name'].values))
        
        self.Max_temp_data_avg_H_3 = pd.read_csv('/local/data/sdahal_p/data/Data/Max_Temp_data_avg_H_3.csv')
        self.Max_temp_data_avg_H_3 = self.normalizeCols(self.Max_temp_data_avg_H_3 , list(self.snotel_locations_info['Station Name'].values))
        
        self.Min_temp_data_avg_H_3 = pd.read_csv('/local/data/sdahal_p/data/Data/Min_Temp_data_avg_H_3.csv')
        self.Min_temp_data_avg_H_3 = self.normalizeCols(self.Min_temp_data_avg_H_3 , list(self.snotel_locations_info['Station Name'].values))
        
        self.Obs_temp_data_avg_H_3 = pd.read_csv('/local/data/sdahal_p/data/Data/Obs_Temp_data_avg_H_3.csv')
        self.Obs_temp_data_avg_H_3 = self.normalizeCols(self.Obs_temp_data_avg_H_3 , list(self.snotel_locations_info['Station Name'].values))
        
        
        # SWE values
        
        self.SWE_values_data = pd.read_csv('/local/data/sdahal_p/data/Data/SWE_Collection.csv')

        self.SWE_values_data = self.SWE_values_data.loc[:, self.SWE_values_data.columns.isin(self.snotel_locations_info['Station Name'].to_list() + ['Date'])]
       
        # Years for testing
        
        self.year_1 = 2014
        self.year_2 = 2006
        self.year_3 = 2017
        self.year_4 = 2016
        self.year_5 = 2007
        