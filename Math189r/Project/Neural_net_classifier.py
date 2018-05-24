# -*- coding: utf-8 -*-
"""
Created on Tue May 22 21:41:56 2018

@author: aminv
"""
print("Importing libraries...")
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, GRU, TimeDistributed, PReLU
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder

print("Importing data...")
wdir='C:/Users/aminv/Desktop/UCI_Smartphone_Dataset'

X_train=pd.read_csv(wdir+'/Train/X_train.txt',' ')
y_train=pd.read_csv(wdir+'/Train/y_train.txt')
X_test=pd.read_csv(wdir+'/Test/X_test.txt',' ')
y_test=pd.read_csv(wdir+'/Test/y_test.txt')

y_train=y_train-1
y_test=y_test-1


subject_id_train=pd.read_csv(wdir+'/Train/subject_id_train.txt')
subject_id_test=pd.read_csv(wdir+'/Test/subject_id_test.txt')

print('Encoding data...')
enc = OneHotEncoder()
enc.fit(y_test.values)  
y_test=enc.transform(y_test.values).toarray()
y_train=enc.transform(y_train.values).toarray()

print('Creating validation data...')
num_points=X_train.values.shape[0]
X_validation=X_train.values[int(.9*num_points):,:]
y_validation=y_train[int(.9*num_points):,:]
X_train=X_train.values[:int(.9*num_points),:]
y_train=y_train[:int(.9*num_points),:]



print('Building Neural Network Model...')

input_shape = X_train.shape
number_of_x_columns = int(input_shape[1])  
dropout_pct = 0.10
default_activation_type = 'relu' # 'linear' , 'tanh' , 'softmax' , 'sigmoid' , 'relu'
default_init_type = 'uniform' # 'uniform' , 'lecun_uniform' , 'normal' , 'glorot_normal' , 'he_normal'
default_batch_size = 5
default_layer_width = 80
default_optimizer = 'adam' # 'SGD' , 'RMSprop' , 'Adagrad' , 'Adam' , 'Adamax' , 'Nadam'' , 'RMSprop' , 'Adagrad' , 'Adam' , 'Adamax' , 'Nadam' 
default_loss_function = 'categorical_crossentropy' # 'mae' , 'mse' , 'mape' , 'msle' , 'squared_hinge' , 'kld'
default_regularization_alpha = 0.000
output_classes=12
#default_weight_regularizer = l1l2(l1=default_regularization_alpha, l2=default_regularization_alpha)
#default_activity_regularizer = activity_l1l2(l1=default_regularization_alpha, l2=default_regularization_alpha)

nn_model = Sequential()
nn_model.add(Dense(default_layer_width, input_dim=number_of_x_columns, kernel_initializer=default_init_type))
nn_model.add(Activation(default_activation_type))
nn_model.add(Dropout(dropout_pct))
nn_model.add(Dense(default_layer_width, kernel_initializer=default_init_type))
nn_model.add(Activation(default_activation_type))
nn_model.add(Dropout(dropout_pct))
nn_model.add(Dense(default_layer_width, kernel_initializer=default_init_type))
nn_model.add(Activation(default_activation_type))
nn_model.add(Dropout(dropout_pct))
nn_model.add(Dense(int(round(default_layer_width*0.85)), kernel_initializer=default_init_type))
nn_model.add(Activation(default_activation_type))
nn_model.add(Dropout(dropout_pct))
nn_model.add(Dense(int(round(default_layer_width*0.75)), kernel_initializer=default_init_type))
nn_model.add(Activation(default_activation_type))
nn_model.add(Dropout(dropout_pct))
nn_model.add(Dense(int(round(default_layer_width*0.60)), kernel_initializer=default_init_type))
nn_model.add(Activation(default_activation_type))
nn_model.add(Dropout(dropout_pct))
nn_model.add(Dense(activation="softmax", units=output_classes))

nn_model.compile(optimizer=default_optimizer, loss=default_loss_function, metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=30)
print('Fitting Neural Network Model...')
nn_history = nn_model.fit(X_train, y_train, callbacks=[early_stopping],\
                          validation_data=(X_validation, y_validation), epochs= 50, \
                          batch_size=default_batch_size, verbose = 1, shuffle=True) 

score = nn_model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
