#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:48:45 2020

@author: c00294860

This codes have different architechtures
"""
from tensorflow import keras
from tensorflow.keras import layers
import os
import pickle

'''Defining the swish -activation function'''
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation

import numpy as np
from copy import deepcopy
#import matplotlib.pyplot as plt

class Swish(Activation):
    
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'


def swish(x, beta = 1):
    '''
    "Swish:

    In 2017, Researchers at Google Brain, published a new paper where they proposed their novel activation function named as Swish.
     Swish is a gated version of Sigmoid Activation Function, where the mathematical formula is: f(x, β) = x * sigmoid(x, β) 
     [Often Swish is referred as SILU or simply Swish-1 when β = 1].

    Their results showcased a significant improvement in Top-1 Accuracy for ImageNet at 0.9% better than ReLU for Mobile 
    NASNet-A and 0.7% for Inception ResNet v2." from https://towardsdatascience.com/mish-8283934a72df on  Fri Dec 27 14:44:30 2019
    
    '''
    return (x * sigmoid(beta * x))

get_custom_objects().clear()
get_custom_objects().update({'swish': Swish(swish)})   
 
class LSTM_all:
    
    def __init__(self,features,time_steps=6,prediction_steps=1,activation='relu'):

        self.activation = activation
        self.time_steps=time_steps
        self.prediction_steps=prediction_steps
        self.features = features

        self.o_p_size_predicted=features*prediction_steps
        '''
        This rough calculation may help:
            In order to maintain the nurons capabity with number of feature
            and maintain factor of 128
            
            fetures inter relation-ship can be get by atleast corrlations' matrixs one with digonal portion
            like
            *
            **
            ***
            ****
            1,2,3,4...
            
            it is an arithmatic series summation /2
            (n * (2 + (n - 1))) / 2)
            
            where n: number of features
            number_minimum_neurons_features= (n * (2 + (n - 1)))/4
            like wise relation ship along the time(if time_steps relationship want to prserve)
            number_minimum_neurons_weigthts_time= (time_steps * (2 + (time_steps - 1))) /4
            
            
            number_minimum_neurons_weigthts = number_minimum_neurons_features*number_minimum_neurons_weigthts_time
            
        Always make sure number of train samples*fetaures > trainable_weights*10
        
        Else try to do with minimum weights other wise it will be over fit to the data needed to train
        
        '''
#        number_minimum_neurons_weigthts = (((features * (2 + (features - 1)))*(time_steps* (time_steps + (time_steps - 1)))) /4)
        number_minimum_neurons_weigthts = (((features * (2 + (features - 1)))*(2* (2 + (2 - 1)))) /4)

        h_size_fac=1
        h_size_neurons=2
        h_size=h_size_neurons*h_size_fac

        while features*h_size<number_minimum_neurons_weigthts:
            h_size_fac=h_size_fac+1
            h_size = h_size_neurons**h_size_fac
        self.h_size= h_size
        self.hidden_dence_size=h_size*2

    def LSTM_base(self):
        
        '''
        Simple LSTM with one layer
        '''
        inputs = keras.Input(shape=(self.time_steps,self.features))

        lstm_1 = layers.LSTM(self.h_size)(inputs)
        Hiden_dense = layers.Dense(self.hidden_dence_size)(lstm_1)
        DROP_OUT= layers.Dropout(0.5)(Hiden_dense)
        outputs_b = layers.Dense(self.o_p_size_predicted,activation=self.activation)(DROP_OUT)
        outputs = layers.Reshape((self.prediction_steps, self.features))(outputs_b)
        model_name =''.join(['LSTM_base_h_size',str(self.h_size),'_act_',self.activation,'.h5'])
        model = keras.models.Model(inputs=inputs, outputs=outputs,name=model_name)

        return model,model_name
    
    def Stacked_LSTM_base(self,number_of_stacks=2):
        '''
        Stacked LSTM 
        https://machinelearningmastery.com/stacked-long-short-term-memory-networks/
        
        This is made with only 2/3 layers of stack to see the performance
        '''
        inputs = keras.Input(shape=(self.time_steps,self.features))
        lstm_1 = layers.LSTM(self.h_size, return_sequences=True)(inputs)

        if number_of_stacks==2:
            lstm_2 = layers.LSTM(self.h_size)(lstm_1)
            
            Hiden_dense = layers.Dense(self.hidden_dence_size,activation=self.activation)(lstm_2)

        elif number_of_stacks==3:
            lstm_2 = layers.LSTM(self.h_size, return_sequences=True)(lstm_1)
            lstm_3 = layers.LSTM(self.h_size)(lstm_2)
            
            Hiden_dense = layers.Dense(self.hidden_dence_size,activation=self.activation)(lstm_3)
        else:
            raise("ONly upto 3 stacks are defined needed to contract if more")
        DROP_OUT= layers.Dropout(0.5)(Hiden_dense)#THIS IS ADDED AFTER THE DATA RUN FOR ALL 2-11 IN FIRST DATA EVAL
        outputs_b = layers.Dense(self.o_p_size_predicted,activation='sigmoid')(DROP_OUT)
        outputs = layers.Reshape((self.prediction_steps, self.features))(outputs_b)

        model_name =''.join(['Stacked_LSTM_base_with_',str(number_of_stacks),'_stacks_','_h_size',str(self.h_size),'_act_',self.activation,'.h5'])
        model = keras.models.Model(inputs=inputs, outputs=outputs,name=model_name)

        return model,model_name
    
    def Time_dist_LSTM(self):
        '''
        time distributed LSTM 
        https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/
        
        This will be helpful to findout the better number for
        timesteps needed to predict the next o/p
        '''
        inputs = keras.Input(shape=(self.time_steps,self.features))

        lstm_1 = layers.LSTM(self.h_size, return_sequences=True)(inputs)
        Hiden_dense = layers.TimeDistributed(layers.Dense(self.hidden_dence_size,activation=self.activation))(lstm_1)
        DROP_OUT= layers.Dropout(0.5)(Hiden_dense)#THIS IS ADDED AFTER THE DATA RUN FOR ALL 2-11 IN FIRST DATA EVAL
        outputs = layers.TimeDistributed(layers.Dense(self.features,activation='sigmoid'))(DROP_OUT)
#        outputs = layers.TimeDistributed(layers.Dense(self.features))(Hiden_dense)

        model_name =''.join(['Time_dist_LSTM_',str(self.time_steps),'_h_size',str(self.h_size),'_act_',self.activation,'.h5'])
        model = keras.models.Model(inputs=inputs, outputs=outputs,name=model_name)

        return model,model_name

    def TIME_dist_Stacked_LSTM(self,number_of_stacks=2):
        '''
        Stacked LSTM 
        https://machinelearningmastery.com/stacked-long-short-term-memory-networks/
        
        This is made with only 2/3 layers of stack to see the performance
        '''
        inputs = keras.Input(shape=(self.time_steps,self.features))
        lstm_1 = layers.LSTM(self.h_size, return_sequences=True)(inputs)

        if number_of_stacks==2:
            lstm_2 = layers.LSTM(self.h_size)(lstm_1)
            
            Hiden_dense = layers.Dense(self.hidden_dence_size,activation=self.activation)(lstm_2)

        elif number_of_stacks==3:
            lstm_2 = layers.LSTM(self.h_size, return_sequences=True)(lstm_1)
            lstm_3 = layers.LSTM(self.h_size)(lstm_2)
            
            Hiden_dense = layers.Dense(self.hidden_dence_size,activation=self.activation)(lstm_3)
        else:
            raise("ONly upto 3 stacks are defined needed to contract if more")
        DROP_OUT= layers.Dropout(0.5)(Hiden_dense)#THIS IS ADDED AFTER THE DATA RUN FOR ALL 2-11 IN FIRST DATA EVAL
        outputs = layers.Dense(self.o_p_size_predicted,activation='sigmoid')(DROP_OUT)
        outputs = layers.TimeDistributed(layers.Dense(self.features,activation='sigmoid'))(DROP_OUT)
#        outputs = layers.TimeDistributed(layers.Dense(self.features))(Hiden_dense)

        model_name =''.join(['Time_dist_Stacked_LSTM_h_with_',str(number_of_stacks),'_stacks_','_h_size',str(self.h_size),'_act_',self.activation,'.h5'])

        model = keras.models.Model(inputs=inputs, outputs=outputs,name=model_name)

        return model,model_name            
      
    def LSTM_base_hidden(self,h_size_fac):
        
        '''
        Simple LSTM with one layer
        '''
        inputs = keras.Input(shape=(self.time_steps,self.features))
        h_size=int(self.h_size*h_size_fac)
        
        lstm_1 = layers.LSTM(h_size)(inputs)
        Hiden_dense = layers.Dense(self.hidden_dence_size)(lstm_1)
        DROP_OUT= layers.Dropout(0.5)(Hiden_dense)
        outputs_b = layers.Dense(self.o_p_size_predicted,activation=self.activation)(DROP_OUT)
        outputs = layers.Reshape((self.prediction_steps, self.features))(outputs_b)
        model_name =''.join(['LSTM_base_h_size',str(h_size),'_act_',self.activation,'.h5'])
        model = keras.models.Model(inputs=inputs, outputs=outputs,name=model_name)

        return model,model_name
         
    def Time_dist_LSTM_given_hidden(self,h_size_fac):
        
        '''
        time distributed LSTM 
        https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/
        
        This will be helpful to findout the better number for
        timesteps needed to predict the next o/p
        '''
        h_size=int(self.h_size*h_size_fac)
        inputs = keras.Input(shape=(self.time_steps,self.features))

        lstm_1 = layers.LSTM(h_size, return_sequences=True)(inputs)
        Hiden_dense = layers.TimeDistributed(layers.Dense(self.hidden_dence_size,activation=self.activation))(lstm_1)
        DROP_OUT= layers.Dropout(0.5)(Hiden_dense)#THIS IS ADDED AFTER THE DATA RUN FOR ALL 2-11 IN FIRST DATA EVAL
        outputs = layers.TimeDistributed(layers.Dense(self.features,activation='sigmoid'))(DROP_OUT)
#        outputs = layers.TimeDistributed(layers.Dense(self.features))(Hiden_dense)

        model_name =''.join(['Time_dist_LSTM_',str(self.time_steps),'_h_size',str(h_size),'_act_',self.activation,'.h5'])
        model = keras.models.Model(inputs=inputs, outputs=outputs,name=model_name)

        return model,model_name

    def Stacked_LSTM_base_hidden(self,h_size_fac,number_of_stacks=2):
        '''
        Stacked LSTM 
        https://machinelearningmastery.com/stacked-long-short-term-memory-networks/
        
        This is made with only 2/3 layers of stack to see the performance
        '''
        h_size=int(self.h_size*h_size_fac)

        inputs = keras.Input(shape=(self.time_steps,self.features))
        lstm_1 = layers.LSTM(h_size, return_sequences=True)(inputs)

        if number_of_stacks==2:
            lstm_2 = layers.LSTM(h_size)(lstm_1)
            
            Hiden_dense = layers.Dense(self.hidden_dence_size,activation=self.activation)(lstm_2)

        elif number_of_stacks==3:
            lstm_2 = layers.LSTM(h_size, return_sequences=True)(lstm_1)
            lstm_3 = layers.LSTM(h_size)(lstm_2)
            
            Hiden_dense = layers.Dense(self.hidden_dence_size,activation=self.activation)(lstm_3)
        else:
            raise("ONly upto 3 stacks are defined needed to contract if more")
        DROP_OUT= layers.Dropout(0.5)(Hiden_dense)#THIS IS ADDED AFTER THE DATA RUN FOR ALL 2-11 IN FIRST DATA EVAL
        outputs_b = layers.Dense(self.o_p_size_predicted,activation='sigmoid')(DROP_OUT)
        outputs = layers.Reshape((self.prediction_steps, self.features))(outputs_b)

        model_name =''.join(['Stacked_LSTM_base_with_',str(number_of_stacks),'_stacks_','_h_size',str(h_size),'_act_',self.activation,'.h5'])
        model = keras.models.Model(inputs=inputs, outputs=outputs,name=model_name)

        return model,model_name
    
       


  

class data_feeder:
    def __init__(self,data_dir_main,data,model_name,time_steps,features,actual_type_data_fetaures,prediction_steps=1,train_size_frac=0.7,batch_size=32,name_set='train'):       
        self.prediction_steps=prediction_steps
        self.time_steps=time_steps
        self.features = features
        
        self.batch_size = 32
        self.train_data_index_position = 0 #to retrieve the data where left
        self.test_data_index_position = 0 #to retrieve the data where left
    
        self.model_name=model_name# inorder to create the targetsw depends on the model given
        self.data= data
        train_size=int(train_size_frac*len(data))
        test_size=len(data)-train_size
    #        valid_size=int(valid_size_frac*len(data))
    #        test_size=len(data)-train_size-valid_size
        self.train_indexes,self.train_size_data_last=self.index_former(train_size)
        self.test_indexes,self.test_size_data_last=self.index_former(test_size)
        self.train_size=train_size
        self.calc_min_max_whole()
        self.simple_normaliser_train_data()
        self.actual_type_data_fetaures=actual_type_data_fetaures
        number_minimum_neurons_weigthts = (((features * (2 + (features - 1)))*(2* (2 + (2 - 1)))) /4)
    
        h_size_fac=1
        h_size_neurons=2
        h_size=h_size_neurons*h_size_fac
    
        while features*h_size<number_minimum_neurons_weigthts:
            h_size_fac=h_size_fac+1
            h_size = h_size_neurons**h_size_fac
        self.h_size= h_size
        
        '''check the training and testing saving directory'''
        os.chdir('/')
        self.saving_dir_train = ''.join([data_dir_main,'/data_results/',model_name,'/train'])
        self.saving_dir_test = ''.join([data_dir_main,'/data_results/',model_name,'/test'])
    
        if not os.path.isdir(self.saving_dir_train):
            os.makedirs(self.saving_dir_train)
            print("Making train direcory: ",self.saving_dir_train)
        if not os.path.isdir(self.saving_dir_test):
            os.makedirs(self.saving_dir_test)
            print("Making test direcory: ",self.saving_dir_test)   
          
        self.data_dir=''.join([data_dir_main,'/data_results/',model_name,'/'])
        os.chdir('/')
        os.chdir(self.data_dir)
        self.name_set=name_set
        
        print('')
        print('')
        print('')
        print('Now in ',name_set,' directory')
        print('')
        print('')
        print('')
        print('')
            
    def hidden_size_calc(self):
        features=self.features
    #    number_minimum_neurons_weigthts = (((features * (2 + (features - 1)))*(2* (2 + (2 - 1)))) /4)
        number_minimum_neurons_weigthts=((features +1)* features)*(2* (2 + 1)) /4
        h_size_fac=1
        h_size_neurons=2
        h_size=h_size_neurons*h_size_fac
    
        while features*h_size<number_minimum_neurons_weigthts:
            h_size_fac=h_size_fac+1
            h_size = h_size_neurons**h_size_fac
        return h_size
    
    def calc_min_max_whole(self):
        '''just normalise the data with mninimum and maximum'''     
        numpy_data=np.array(self.data)
        self.whole_feature_max=np.amax(numpy_data, axis=0)
        self.whole_feature_min=np.amin(numpy_data, axis=0)
        
    def simple_normaliser_train_data(self):
        '''just normalise the data with mninimum and maximum'''
        self.data_given = deepcopy(self.data)
        self.test_data_given = deepcopy(self.data[self.train_size:len(self.data)])
        
        feature_max=np.amax(self.data_given, axis=0)
        feature_min=np.amin(self.data_given, axis=0)
        
        numpy_data=np.array(deepcopy(self.data))

        for i in range(0,numpy_data.shape[0]):
            for j in range(0,numpy_data.shape[1]):
                if feature_min[j]==feature_max[j]:
                    numpy_data[i,j]= 0
                    if i<5:
                        print('the coloumn j ',j,' is not providing any training info always same value')
                else:
                    if numpy_data[i,j]<=feature_min[j]:
                        numpy_data[i,j]=0
                    elif numpy_data[i,j]>=feature_max[j]:
                        numpy_data[i,j]=1
                    else:
                        numpy_data[i,j]= (numpy_data[i,j]-feature_min[j])/(feature_max[j]-feature_min[j])
        self.feature_max=feature_max      
        self.feature_min=feature_min
        self.data=numpy_data
        self.test_data=numpy_data[self.train_size:len(numpy_data),:]

    def index_former(self,train_size):
        '''
        This function help to creaate the batch splited data
        '''
        iter_train = (train_size-(self.time_steps+self.prediction_steps))//self.batch_size
 
        if (train_size-(self.time_steps+self.prediction_steps))%self.batch_size > self.time_steps+self.prediction_steps:
            iter_train =  iter_train + 1
            size_data_last =(train_size-(self.time_steps+self.prediction_steps))%self.batch_size
        else:
            size_data_last=self.batch_size
        indexes = np.arange(0,iter_train)
        return indexes,size_data_last
       
    def train_data_get(self,index,train_data_sel=True):
        '''
        the selector 
        train_data_sel
            decides to train or analyse
        '''
        if index< len(self.train_indexes)-1:
            size_data = self.batch_size
        elif index== (len(self.train_indexes)-1):
            size_data =self.train_size_data_last

        x = np.empty((size_data,self.time_steps,self.features))
        if self.model_name=='Time_dist_LSTM':
            y = np.empty((size_data,self.time_steps,self.features), dtype=float)
            y_given = np.empty((size_data,self.time_steps,self.features), dtype=float)
        elif self.model_name=='LSTM_base' or self.model_name=='Stacked_LSTM_base':
#            y = np.empty((size_data,self.prediction_steps*self.features), dtype=float)
            y = np.empty((size_data,self.prediction_steps,self.features), dtype=float)
            y_given = np.empty((size_data,self.prediction_steps,self.features), dtype=float)
 
        else:
          raise ("Miss use of Function __data_generation")
        # Generate data
        for i in range(self.train_data_index_position,self.train_data_index_position+size_data):
            # Store sample
            for j in range(0,self.time_steps):
                x[i-self.train_data_index_position,j,:] = deepcopy(self.data[i+j])
                if self.model_name=='Time_dist_LSTM':
                    y[i-self.train_data_index_position,j,:] = deepcopy(self.data[i+j+1])
                    y_given[i-self.train_data_index_position,j,:] = deepcopy(self.data_given[i+j+1])

                else:
                    if j == (self.time_steps-1):#at the end add the prediction steps
                        for k in range(0,self.prediction_steps):
#                            y[i-self.train_data_index_position,k:(self.features+k)] = deepcopy(self.data[i+j+k+1])
#                            y_given[i-self.train_data_index_position,k:(self.features+k)] = deepcopy(self.data_given[i+j+k+1])
                            y[i-self.train_data_index_position,k,:] = deepcopy(self.data[i+j+k+1])
                            y_given[i-self.train_data_index_position,k,:] = deepcopy(self.data_given[i+j+k+1])

        self.train_data_index_position = self.train_data_index_position + size_data
        if index== len(self.train_indexes)-1:
            self.train_data_index_position=0
        if train_data_sel: 
            return x,y
        else:
            return y,y_given

    def test_data_get(self,index,test_data_sel=True):
        '''
        the selector 
        test_data_sel
            decides to test or analyse the test data
        '''
        if index< len(self.test_indexes)-1:
            size_data = self.batch_size
        elif index== (len(self.test_indexes)-1):
            size_data =self.test_size_data_last

        x = np.empty((size_data,self.time_steps,self.features))
        if self.model_name=='Time_dist_LSTM':
            y = np.empty((size_data,self.time_steps,self.features), dtype=float)
            y_given = np.empty((size_data,self.time_steps,self.features), dtype=float)
        elif self.model_name=='LSTM_base' or self.model_name=='Stacked_LSTM_base':
#            y = np.empty((size_data,self.prediction_steps*self.features), dtype=float)
            y = np.empty((size_data,self.prediction_steps,self.features), dtype=float)
            y_given = np.empty((size_data,self.prediction_steps,self.features), dtype=float)
 
        else:
          raise ("Miss use of Function __data_generation")
        # Generate data
        for i in range(self.test_data_index_position,self.test_data_index_position+size_data):
            # Store sample
            for j in range(0,self.time_steps):
                x[i-self.test_data_index_position,j,:] = deepcopy(self.test_data[i+j])
                if self.model_name=='Time_dist_LSTM':
                    y[i-self.test_data_index_position,j,:] = deepcopy(self.test_data[i+j+1])
                    y_given[i-self.test_data_index_position,j,:] = deepcopy(self.test_data_given[i+j+1])

                else:
                    if j == (self.time_steps-1):#at the end add the prediction steps
                        for k in range(0,self.prediction_steps):
#                            y[i-self.test_data_index_position,k:(self.features+k)] = deepcopy(self.test_data[i+j+k+1])
#                            y_given[i-self.test_data_index_position,k:(self.features+k)] = deepcopy(self.test_data_given[i+j+k+1])
                            y[i-self.test_data_index_position,k,:] = deepcopy(self.test_data[i+j+k+1])
                            y_given[i-self.test_data_index_position,k,:] = deepcopy(self.test_data_given[i+j+k+1])
                            
        self.test_data_index_position = self.test_data_index_position + size_data
        if index== len(self.test_indexes)-1:
            self.test_data_index_position=0
        if test_data_sel: 
            return x
        else:
            return y,y_given
    
    def Error_retrieve(self,stack=0,h_size_fac=1):
        '''
        Inorder to retrieve the MSE for time distributed LSTM
        Here the model implemented predicting 1 step ahead with the given time steps
        prediction_steps like
           
        since each time 
        
        **  set-1
        given        t_1,t_2,..,t_11
        preidicted       p_2,p_3...,p_12
        
        **  set-2
        given        t_2,t_3,..,t_13
        preidicted       p_3,p_4...,p_14
        
        thats why time steps are given as prediction as well
        '''
        os.chdir('/')
        os.chdir(self.data_dir)
        
        name_set=self.name_set
        batch_size=self.batch_size
        features=self.features
        model_name=self.model_name
        if  self.model_name=='Time_dist_LSTM' and h_size_fac!=1:
            prediction_steps=self.time_steps#since the Time distribution analysis predict just one step ahead for the given distrbution
            predicted= pickle.load(open(''.join(["Time_dist_LSTM_vin1_predict_all_",name_set,"_timesteps_",str(self.time_steps),'_h_size_',str(int(self.h_size*h_size_fac)),"_fac_",str(h_size_fac),".p"]), "rb"))

        elif  self.model_name=='Time_dist_LSTM':
            prediction_steps=self.time_steps#since the Time distribution analysis predict just one step ahead for the given distrbution
            predicted= pickle.load(open(''.join(["Time_dist_LSTM_vin1_predict_all_",name_set,"_timesteps_",str(prediction_steps),".p"]), "rb"))       
        elif model_name=='LSTM_base':
            prediction_steps = self.prediction_steps
            if stack==0:
                predicted= pickle.load(open(''.join(["LSTM_base_vin1_",name_set,"_timesteps_",str(self.time_steps),"_predsteps_",str(prediction_steps),'_h_size_',str(int(self.h_size*h_size_fac)),"_fac_",str(h_size_fac),".p"]), "rb"))

        print(prediction_steps)
        #    SE_in_normalised_scale_all=np.empty((len(predicted),batch_size,prediction_steps,features))
        SE_given_scale=np.empty((len(predicted),batch_size,prediction_steps,features))
        E_given_scale=np.empty((len(predicted),batch_size,prediction_steps,features))
        E_in_normalised_scale_all=np.empty((len(predicted),batch_size,prediction_steps,features))

        new_shape=0
        for i in range(0,len(predicted)):
            taken_pred=predicted[i]
            y=np.empty((taken_pred.shape[0],prediction_steps,features))
            if name_set=='train':
                given_y_norm,given_y = self.train_data_get(i,train_data_sel=False)
            elif name_set=='test':
                given_y_norm,given_y = self.test_data_get(i,test_data_sel=False)

            for j in range(0,taken_pred.shape[0]):
                for k in range(0,prediction_steps):
                    for m in range(0,self.features):
                        y[j,k,m]=self.actual_type_data_fetaures[m]((taken_pred[j,k,m]*(self.feature_max[m]-self.feature_min[m]))+self.feature_min[m])
                        SE_given_scale[i,j,k,m]=deepcopy((given_y[j,k,m]-y[j,k,m])**2)
        #                    SE_in_normalised_scale_all[i,j,k,m]=deepcopy((given_y_norm[j,k,m]-taken_pred[j,k,m])**2)
                        E_given_scale[i,j,k,m]=deepcopy(given_y[j,k,m]-y[j,k,m])
                        E_in_normalised_scale_all[i,j,k,m]=deepcopy(given_y_norm[j,k,m]-taken_pred[j,k,m])
            new_shape=new_shape+taken_pred.shape[0]
         
        '''
        To remove the batch axis and string the matrix
        
        since each time 
        
        **  set-1
        given        t_1,t_2,..,t_6
        preidicted       p_2,p_3...,p_7
        
        **  set-2
        given        t_2,t_3,..,t_7
        preidicted       p_3,p_4...,p_8
        
        '''
        SE_given_scale_fin=np.empty((new_shape,prediction_steps,features))
    #    SE_in_normalised_scale_all_fin = np.empty((new_shape,prediction_steps,features))
        E_given_scale_fin=np.empty((new_shape,prediction_steps,features))
        E_in_normalised_scale_all_fin = np.empty((new_shape,prediction_steps,features))
        for i in range(0,SE_given_scale.shape[0]):
            for j in range(0,predicted[i].shape[0]):
                if i>0:
                    SE_given_scale_fin[i*predicted[i-1].shape[0]+j,:,:] = deepcopy(SE_given_scale[i,j,:,:])
    #                SE_in_normalised_scale_all_fin[i*predicted[i-1].shape[0]+j,:,:] = deepcopy(SE_in_normalised_scale_all[i,j,:,:])
                    E_given_scale_fin[i*predicted[i-1].shape[0]+j,:,:] = deepcopy(E_given_scale[i,j,:,:])
                    E_in_normalised_scale_all_fin[i*predicted[i-1].shape[0]+j,:,:] = deepcopy(E_in_normalised_scale_all[i,j,:,:])
                else:
                    SE_given_scale_fin[j,:,:] = deepcopy(SE_given_scale[i,j,:,:])
    #                SE_in_normalised_scale_all_fin[j,:,:] = deepcopy(SE_in_normalised_scale_all[i,j,:,:])
                    E_given_scale_fin[j,:,:] = deepcopy(E_given_scale[i,j,:,:])
                    E_in_normalised_scale_all_fin[j,:,:] = deepcopy(E_in_normalised_scale_all[i,j,:,:])  
           
        if name_set=='test':
            '''
            To make fair in test set evaluation
            leave the first set since the LSTM model need to track the flow
            '''
            SE_given_scale_fin=SE_given_scale_fin[1:SE_given_scale_fin.shape[0],:,:]
            E_given_scale_fin=E_given_scale_fin[1:E_given_scale_fin.shape[0],:,:]
            E_in_normalised_scale_all_fin = E_in_normalised_scale_all_fin[1:E_in_normalised_scale_all_fin.shape[0],:,:]

        MSE_given_scale_fin= np.average(SE_given_scale_fin, axis=0)
        RMSE_given_scale_fin=np.empty((MSE_given_scale_fin.shape))
        for i in range(0,MSE_given_scale_fin.shape[0]):
            for j in range(0,MSE_given_scale_fin.shape[1]):
                RMSE_given_scale_fin[i,j]=(MSE_given_scale_fin[i][j])**(0.5)
        
        os.chdir('/')
        if name_set=='train':
            os.chdir(self.saving_dir_train)
        elif name_set=='test':
            os.chdir(self.saving_dir_test)
            
        if stack==0:
             if model_name=='Time_dist_LSTM':
                pickle.dump(RMSE_given_scale_fin, open(''.join([name_set,'_RMSE_given_scale_fin_',model_name,'_',str(prediction_steps),'_h_size_',str(int(self.h_size*h_size_fac)),'.p']), "wb" ))
                pickle.dump(SE_given_scale_fin, open(''.join([name_set,'_SE_given_scale_fin_',model_name,'_',str(prediction_steps),'_h_size_',str(int(self.h_size*h_size_fac)),'.p']), "wb" ))
                pickle.dump(E_given_scale_fin, open(''.join([name_set,'_E_given_scale_fin_',model_name,'_',str(prediction_steps),'_h_size_',str(int(self.h_size*h_size_fac)),'.p']), "wb" ))
                pickle.dump(E_in_normalised_scale_all_fin, open(''.join([name_set,'_E_in_normalised_scale_all_fin_',model_name,'_',str(prediction_steps),'_h_size_',str(int(self.h_size*h_size_fac)),'.p']), "wb" ))
             elif model_name=='LSTM_base':
                pickle.dump(RMSE_given_scale_fin, open(''.join([name_set,'_RMSE_given_scale_fin_',model_name,'_time_step_given_',str(self.time_steps),'_pred_',str(prediction_steps),'_h_size_',str(int(self.h_size*h_size_fac)),'.p']), "wb" ))
                pickle.dump(SE_given_scale_fin, open(''.join([name_set,'_SE_given_scale_fin_',model_name,'_time_step_given_',str(self.time_steps),'_pred_',str(prediction_steps),'_h_size_',str(int(self.h_size*h_size_fac)),'.p']), "wb" ))
                pickle.dump(E_given_scale_fin, open(''.join([name_set,'_E_given_scale_fin_',model_name,'_time_step_given_',str(self.time_steps),'_pred_',str(prediction_steps),'_h_size_',str(int(self.h_size*h_size_fac)),'.p']), "wb" ))
                pickle.dump(E_in_normalised_scale_all_fin, open(''.join([name_set,'_E_in_normalised_scale_all_fin_',model_name,'_time_step_given_',str(self.time_steps),'_pred_',str(prediction_steps),'_h_size_',str(int(self.h_size*h_size_fac)),'.p']), "wb" ))     
             else:
                raise("Build here stack-0")

        return SE_given_scale_fin,E_given_scale_fin,E_in_normalised_scale_all_fin,RMSE_given_scale_fin#SE_in_normalised_scale_all_fin
 
    def Error_load(self,stack=0,h_size_fac=1):
        '''
        Inorder to retrieve the MSE for time distributed LSTM
        Here the model implemented predicting 1 step ahead with the given time steps
        prediction_steps like
           
        since each time 
        
        **  set-1
        given        t_1,t_2,..,t_11
        preidicted       p_2,p_3...,p_12
        
        **  set-2
        given        t_2,t_3,..,t_13
        preidicted       p_3,p_4...,p_14
        
        thats why time steps are given as prediction as well
        '''
        name_set=self.name_set
        model_name=self.model_name
        
        if  self.model_name=='Time_dist_LSTM':
            prediction_steps=self.time_steps#since the Time distribution analysis predict just one step ahead for the given distrbution
        elif model_name=='LSTM_base':
            prediction_steps = self.prediction_steps
           
        os.chdir('/')
        if name_set=='train':
            os.chdir(self.saving_dir_train)
        elif name_set=='test':
            os.chdir(self.saving_dir_test)
            
        if stack==0:
			if model_name=='Time_dist_LSTM':
#                RMSE_given_scale_fin=pickle.load(open(''.join([name_set,'_RMSE_given_scale_fin_',model_name,'_',str(prediction_steps),'_h_size_',str(int(self.h_size*h_size_fac)),"_fac_",str(h_size_fac),'.p']), "rb"))   
#                SE_given_scale_fin=pickle.load(open(''.join([name_set,'_SE_given_scale_fin_',model_name,'_',str(prediction_steps),'_h_size_',str(int(self.h_size*h_size_fac)),"_fac_",str(h_size_fac),'.p']), "rb"))   
                E_given_scale_fin=pickle.load(open(''.join([name_set,'_E_given_scale_fin_',model_name,'_',str(prediction_steps),'_h_size_',str(int(self.h_size*h_size_fac)),"_fac_",str(h_size_fac),'.p']), "rb"))   
                E_in_normalised_scale_all_fin=pickle.load(open(''.join([name_set,'_E_in_normalised_scale_all_fin_',model_name,'_',str(prediction_steps),'_h_size_',str(int(self.h_size*h_size_fac)),"_fac_",str(h_size_fac),'.p']), "rb"))   
            elif model_name=='LSTM_base':
#                RMSE_given_scale_fin=pickle.load(open(''.join([name_set,'_RMSE_given_scale_fin_',model_name,'_time_step_given_',str(self.time_steps),'_pred_',str(prediction_steps),'_h_size_',str(int(self.h_size*h_size_fac)),"_fac_",str(h_size_fac),'.p']), "rb"))   
#                SE_given_scale_fin=pickle.load(open(''.join([name_set,'_SE_given_scale_fin_',model_name,'_time_step_given_',str(self.time_steps),'_pred_',str(prediction_steps),'_h_size_',str(int(self.h_size*h_size_fac)),"_fac_",str(h_size_fac),'.p']), "rb"))   
                E_given_scale_fin=pickle.load(open(''.join([name_set,'_E_given_scale_fin_',model_name,'_time_step_given_',str(self.time_steps),'_pred_',str(prediction_steps),'_h_size_',str(int(self.h_size*h_size_fac)),"_fac_",str(h_size_fac),'.p']), "rb"))   
                E_in_normalised_scale_all_fin=pickle.load(open(''.join([name_set,'_E_in_normalised_scale_all_fin_',model_name,'_time_step_given_',str(self.time_steps),'_pred_',str(prediction_steps),'_h_size_',str(int(self.h_size*h_size_fac)),"_fac_",str(h_size_fac),'.p']), "rb"))        


            else:
                raise("Build here stack-0")
        else:
            if model_name=='LSTM_base':
#                RMSE_given_sacle_fin=pickle.load(open(''.join([name_set,'_RMSE_given_scale_fin_',model_name,'_stacks_',str(stack),'_time_step_given_',str(self.time_steps),'_pred_',str(prediction_steps),'.p']), "rb"))   
#                SE_given_scale_fin=pickle.load(open(''.join([name_set,'_SE_given_scale_fin_',model_name,'_stacks_',str(stack),'_time_step_given_',str(self.time_steps),'_pred_',str(prediction_steps),'.p']), "rb"))   
                E_given_scale_fin=pickle.load(open(''.join([name_set,'_E_given_scale_fin_',model_name,'_stacks_',str(stack),'_time_step_given_',str(self.time_steps),'_pred_',str(prediction_steps),'.p']), "rb"))   
                E_in_normalised_scale_all_fin=pickle.load(open(''.join([name_set,'_E_in_normalised_scale_all_fin_',model_name,'_stacks_',str(stack),'_time_step_given_',str(self.time_steps),'_pred_',str(prediction_steps),'.p']), "rb"))   
            else:
                raise("Build here stack else")


        dict_average_errors={}
        SE_normalised_scale = np.square(E_in_normalised_scale_all_fin)

        dict_average_errors['MSE_normal_scale']=np.average(SE_normalised_scale)
        dict_average_errors['MAE_normal_scale']=np.average(np.absolute(E_in_normalised_scale_all_fin))    
        dict_average_errors['RMSE_normal_scale']=np.sqrt(dict_average_errors['MSE_normal_scale'])

    
        MAE_error_t=np.average(np.absolute(E_given_scale_fin),axis=0)
        dict_average_errors['MAE_given_scale_feature']=np.average(MAE_error_t,axis=0)
        
        pickle.dump(dict_average_errors, open(''.join([name_set,'_dict_average_errors_',model_name,'_stacks_',str(stack),'_time_step_given_',str(self.time_steps),'_pred_',str(prediction_steps),'.p']), "wb" ))

        return dict_average_errors

    def dict_main_info(self):
        
        os.chdir('/')
        os.chdir(self.saving_dir_test)
        '''
        Feature error also displayed in actual scale with 
        Actual training set range presented
        And Train+Test set range
        '''
        dict_main_info={}
        dict_main_info['feature_max_train_set'] = self.feature_max      
        dict_main_info['feature_min_train_set'] = self.feature_min
        '''
        This is the right way to implemet it
        Recalculate theminimum and maximum and the model is trained again for training set based on the new information 
        
        '''
        dict_main_info['feature_max_whole_set'] = self.whole_feature_max  
        dict_main_info['feature_min_whole_set'] = self.whole_feature_min
        pickle.dump(dict_main_info, open('dict_main_info_.p', "wb" ))

