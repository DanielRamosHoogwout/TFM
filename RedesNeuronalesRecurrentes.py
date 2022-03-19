# -*- coding: utf-8 -*-
"""
Trabajo Final de Máster:
    Redes Neuronales Recurrentes aplicadas a criptomonedas.
"""

# Parte 1 - Preprocesado de datos

#Librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Cargar datos
dataset_train = pd.read_csv("C:/Users/Daniel/Desktop/TFM/Datos/BTC_USD_Train.csv")

#Se toma solo el valor de apertura
training_set = dataset_train.iloc[:, 1:2].values #Dataframe de 1 columna

#Escalado de características
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

# Crear una estrucutra de datos con 60 timesteps y 1 salida
X_train = []
y_train = []

for i in range(60, 1963): #2618 tendré que cambiarlo por el ultimo dato de train
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])

X_train, y_train = np.array(X_train), np.array(y_train)
# Redimensión de los datos
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))                            
#Parte 2 - Construcción de la RNR
from keras.models import Sequential #Definir capas de red neuronal
from keras.layers import Dense, LSTM, Dropout 
#La ultima capa de la red neuronal
#Retroalimentar la red neuronal
#Regular y prevenir overfitting

#Inicialización del modelo
regressor = Sequential()

#Aplicar LSTM y Dropout
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1],1) ))
regressor.add(Dropout(0.2)) #20% de las neuronas no se va a utilizar
#Parte 3 - Ajustar las predicciones y visualizar los resultados
