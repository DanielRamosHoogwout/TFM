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
dataset_train = pd.read_csv("C:/Users/Daniel/Desktop/TFM/Datos/BTC_train_nov2019.csv")

#Se toma solo el valor de apertura
training_set = dataset_train.iloc[:, [1,4]].values #Dataframe de 1 columna

#Escalado de características
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

# Crear una estrucutra de datos con 60 timesteps y 1 salida
X_train = []
X_train2 = []
y_train = []

for i in range(60, len(training_set)): #2618 tendré que cambiarlo por el ultimo dato de train
    X_train.append(training_set_scaled[i-60:i,0])
    X_train2.append(training_set_scaled[i-60:i,1])
    y_train.append(training_set_scaled[i,0])

X_train, X_train2, y_train = np.array(X_train), np.array(X_train2), np.array(y_train)
# Redimensión de los datos
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_train2 = np.reshape(X_train2, (X_train2.shape[0], X_train2.shape[1], 1))
 
# Unimos todas las capas de nuestros datos tridimensionales
X_train = np.append(X_train, X_train2, axis = 2)
                           
#Parte 2 - Construcción de la RNR
from keras.models import Sequential #Definir capas de red neuronal
from keras.layers import Dense, LSTM, Dropout 
#La ultima capa de la red neuronal
#Retroalimentar la red neuronal
#Regular y prevenir overfitting

#Inicialización del modelo
regressor = Sequential()

#Aplicar LSTM y Dropout

#Primera capa
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
regressor.add(Dropout(0.2)) #20% de las neuronas no se va a utilizar

#Segunda capa
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2)) #20% de las neuronas no se va a utilizar

#Tercera capa
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2)) #20% de las neuronas no se va a utilizar

#Cuarta capa
regressor.add(LSTM(units = 50, return_sequences = False))
regressor.add(Dropout(0.2)) #20% de las neuronas no se va a utilizar

#Capa de salida
regressor.add(Dense(units = 1))

#Compilar
regressor.compile(optimizer= 'adam', loss = 'mean_squared_error') #Documentación RNR Optimizador: RMSprop vs ADAM adam mejor

#Conjunto entrenamiento
regressor.fit(X_train, y_train, epochs= 100, batch_size = 32)

#Parte 3 - Ajustar las predicciones y visualizar los resultados

#Cargamos los datos del mes de enero-marzo de 2020
dataset_test = pd.read_csv("C:/Users/Daniel/Desktop/TFM/Datos/BTC_test_feb2020.csv")
real_price = dataset_test.iloc[:, [1,4]].values #Dataframe de 1 columna

#Predecir las acciones de enero-marzo de 2020
dataset_total = pd.concat((dataset_train[['Open','Close']], dataset_test[['Open','Close']]), axis = 0)
inputs  = dataset_total[len(dataset_total)-len(dataset_test)-60: ].values #Formato fila

inputs = sc.transform(inputs) #Reescalamos los datos (el mínimo y el máximo se han obtenido del sc anterior.)

X_test = []
X_test2 = []
for i in range(60, 60+len(dataset_test)): #2618 tendré que cambiarlo por el ultimo dato de train
    X_test.append(inputs[i-60:i,0])
    X_test2.append(inputs[i-60:i,1])

X_test, X_test2= np.array(X_test), np.array(X_test2)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)) #Tridimensionalizamos
X_test2 = np.reshape(X_test2, (X_test2.shape[0], X_test2.shape[1], 1))

#Unimos
X_test = np.append(X_test, X_test2, axis = 2)
                    
predicted_price = regressor.predict(X_test)
predicted_price = np.append(predicted_price, ([[0]]*91), axis = 1)
predicted_price = sc.inverse_transform(predicted_price)

#Visualizacion de los datos
plt.plot(real_price[ : , 0], color = 'red', label = 'Real Bitcoin Price')
plt.plot(predicted_price[ : , 0], color = 'blue', label = 'Predicted Bitcoin Price')
plt.title('Bitcoin Price Prediction')
plt.xlabel('Time')
plt.ylabel('Bitcoin Price')
plt.legend()
plt.show()

#Metricas de error
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_price[:,0], predicted_price[:,0]))
print(rmse)