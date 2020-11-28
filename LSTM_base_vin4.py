#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 02 23:31:33 2020

@author: c00294860

go through the Test set and form the results
Here the normalisation of train used is consider the whole datas' features' minimum and maximum
"""
import os
import csv
import pickle
from copy import deepcopy

#data_dir_main= 'C:/Users/c00294860/Desktop/time_series'
data_dir_main= '/home/C00294860/Documents/Anomaly/Anomally_2020_Fall/'
os.chdir('/')
os.chdir(data_dir_main)
file_name='household_power_consumption_part_1.csv'
#%
def retrive_rows(file_name,omit_rows=[],omit_coloumns=[],type_coloumn={},type_change={}):
    '''
    This is a helper function to data curaitation of CSV files
    omit_rows: list of rows not inclueded in final results
    omit_coloumns: list of coloumns not inclueded in final results
    type_coloumn: Type of the data in the columns if you needed to convert in the order of csv
                If this provided then must be given for all coloumns not string
    '''
    f = open(file_name)
    csv_f = csv.reader(f)
    stack=[]
    for row in csv_f:
        stack.append(row)
    break_occured=False
    final_data=[]
    for i in range(0,len(stack)):
        if i not in omit_rows: 
           final_data_t=[]
           for j in range(0,len(stack[i])):
               if j not in omit_coloumns:
                  if j in list(type_coloumn.keys()):
                      try:
                          final_data_t.append(type_coloumn[j](stack[i][j]))
                      except:
                          print(stack[i])
                          break_occured=True
                          break
                  else:
                      if j in list(type_change.keys()):
                          final_data_t.append(type_change[j][stack[i][j]])
                      else:
                          final_data_t.append(stack[i][j])
           if break_occured:
               break
           final_data.append(final_data_t)
    return final_data
# intialise the dataset to fit 
type_coloumn={}
type_coloumn.update({2: float})
type_coloumn.update({3: float})
type_coloumn.update({4: float})
type_coloumn.update({5: float})
type_coloumn.update({6: int})
type_coloumn.update({7: int})
type_coloumn.update({8: int})

type_change={}#may be we can give the direction as relative to general or give different coloumns asnd assign binary 

dataset = retrive_rows(file_name,omit_rows=[0],omit_coloumns=[0,1],type_coloumn=type_coloumn)
actual_type_data_fetaures=[float,float,float,float,int,int,int]
#%%

'''
 go through time base models
'''
from tensorflow import keras
from Deep_time_series_models import data_feeder


        
from Deep_time_series_models import LSTM_all

load_model_dir=''.join([data_dir_main,'data_results/LSTM_base'])

features=7
for h_size_fac in [0.125,0.0625,0.25,0.5,1,2,4,8]:
    #h_size_fac=2
    time_steps = 45
    #prediction_steps=10
    #        for prediction_steps in [1,2,3]:
    for prediction_steps in [15,30,45,60]:
    #        prediction_steps = 1
        model_intial=LSTM_all(features,time_steps,prediction_steps)
        train_data_obj=data_feeder(data_dir_main,dataset,'LSTM_base',time_steps,features,actual_type_data_fetaures,prediction_steps=prediction_steps)
        model,model_name =model_intial.LSTM_base_hidden(h_size_fac)
        model.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
        history_all=[]
        for itration in range(0,100):
            for fetch in range(0,len(train_data_obj.train_indexes)):
                x_train, y_train=train_data_obj.train_data_get(fetch)
                history = model.fit(x_train, y_train,batch_size=len(x_train),epochs=1,validation_split=0.2,verbose=0)
#        #        print(history.history.keys())
#                print('validation acc: ',round(100*history['val_acc'][-1],2),'%')
#                print('training acc: ',round(100*history['acc'][-1],2),'%')
#                history_all.append(deepcopy(history))
            if itration//10==0:
                c_train_test=0
                x_train_all=0
                for fetch in range(0,len(train_data_obj.train_indexes)):
                    x_train, y_train=train_data_obj.train_data_get(fetch)
                    test_scores = model.evaluate(x_train, y_train,verbose=1)
                    c_train_test = c_train_test +int(test_scores[1]*(len(x_train)))
                    x_train_all = x_train_all + len(x_train)
        print("Accuracy: ",100*c_train_test/x_train_all)
        os.chdir('/')
        os.chdir(load_model_dir)
        model.save(''.join(["timesteps_",str(time_steps),"_predsteps_",str(prediction_steps),model_name]))
        predict_all=[]
        for fetch in range(0,len(train_data_obj.train_indexes)):
            x_train, y_train=train_data_obj.train_data_get(fetch)
            predict_all.append(deepcopy(model.predict(x_train,verbose=0)))
        os.chdir('/')
        os.chdir(''.join([load_model_dir,'/train/']))
        pickle.dump(predict_all, open(''.join(["LSTM_base_vin1_train_timesteps_",str(time_steps),"_predsteps_",str(prediction_steps),'_h_size_',str(int(model_intial.h_size*h_size_fac)),"_fac_",str(h_size_fac),".p"]), "wb"))  
           
        #        test_data_obj=data_feeder(dataset,'LSTM_base',time_steps,features,actual_type_data_fetaures)
        test_data_obj=data_feeder(data_dir_main,dataset,'LSTM_base',time_steps,features,actual_type_data_fetaures,prediction_steps=prediction_steps)
        
        predict_all=[]
        for fetch in range(0,len(test_data_obj.test_indexes)):
            x_test =test_data_obj.test_data_get(fetch)
            predict_all.append(deepcopy(model.predict(x_test,verbose=0)))
        os.chdir('/')
        os.chdir(''.join([load_model_dir,'/test/']))
        pickle.dump(predict_all, open(''.join(["LSTM_base_vin1_test_timesteps_",str(time_steps),"_predsteps_",str(prediction_steps),'_h_size_',str(int(model_intial.h_size*h_size_fac)),"_fac_",str(h_size_fac),".p"]), "wb"))  
