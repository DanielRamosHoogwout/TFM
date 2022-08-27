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
training_set = dataset_train.iloc[:, [1,4]].values #Dataframe de 2 columna

#Escalado de características
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

# Crear una estrucutra de datos con period timesteps y 1 salida
X_train = []
X_train2 = []
y_train = []

period = 90

for i in range(period, len(training_set)): #2618 tendré que cambiarlo por el ultimo dato de train
    X_train.append(training_set_scaled[i-period:i,0])
    X_train2.append(training_set_scaled[i-period:i,1])
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
regressor.compile(optimizer= 'adam', loss = 'mean_squared_error', metrics = ['accuracy']) #Documentación RNR Optimizador: RMSprop vs ADAM adam mejor

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', patience = 50, verbose = 1)
mc =  ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=1, save_best_only=True)   
#Conjunto entrenamiento recursivo
#history = regressor.fit(X_train, y_train, epochs= 1000, batch_size = 32)
history = regressor.fit(X_train, y_train, validation_data = (X_test, testing(y_test)), epochs = 300, batch_size = 32, callbacks = [es, mc])

#Cargamos el mejor modelo
from keras.models import load_model
saved_model = load_model('best_model.h5')
# Evaluating model performance
train_loss, train_acc = saved_model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = saved_model.evaluate(X_test, y_test, verbose=0)
print(f'Train accuracy: {train_acc*100:.3f} % || Test accuracy: {test_acc*100:.3f} %')
print(f'Train loss: {train_loss:.3f} || Test loss: {test_loss:.3f}')

# summarize history for loss
print(history.history.keys())
plt.plot(history.history['loss'])
#plt.plot(min(history.history['loss']), marker="o")
#♂plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()

def testing(y_test):
    prediced_price_list = []
    for j in range(len(X_test)):
        predicted_price = saved_model.predict(X_test, batch_size=1)
        predicted_price_list = np.append(predicted_price, ([[0]]*91), axis = 1)

    predicted_price_list = sc.inverse_transform(predicted_price_list)
    
    return predicted_price_list

#Parte 3 - Ajustar las predicciones y visualizar los resultados

#Cargamos los datos del mes de enero-marzo de 2020
dataset_test = pd.read_csv("C:/Users/Daniel/Desktop/TFM/Datos/BTC_test_feb2020.csv")
real_price = dataset_test.iloc[:, [1,4]].values #Dataframe de 1 columna
y_test = dataset_test.iloc[:, [1,4]].values #Esto tengo que cambiarlo
#Predecir las acciones de enero-marzo de 2020
dataset_total = pd.concat((dataset_train[['Open','Close']], dataset_test[['Open','Close']]), axis = 0)
inputs  = dataset_total[len(dataset_total)-len(dataset_test)-period: ].values #Formato fila

inputs = sc.transform(inputs) #Reescalamos los datos (el mínimo y el máximo se han obtenido del sc anterior.)

X_test = []
X_test2 = []
for i in range(period, period+len(dataset_test)):
    X_test.append(inputs[i-period:i,0])
    X_test2.append(inputs[i-period:i,1])

X_test, X_test2= np.array(X_test), np.array(X_test2)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)) #Tridimensionalizamos
X_test2 = np.reshape(X_test2, (X_test2.shape[0], X_test2.shape[1], 1))

#Unimos
X_test = np.append(X_test, X_test2, axis = 2)

prediced_price_list = []
for j in range(len(X_test)):
    predicted_price = saved_model.predict(X_test, batch_size=1)
    predicted_price_list = np.append(predicted_price, ([[0]]*91), axis = 1)

predicted_price_list = sc.inverse_transform(predicted_price_list)

#Visualizacion de los datos
plt.plot(real_price[ : , 0], color = 'red', label = 'Real Bitcoin Price')
plt.plot(predicted_price_list[ : , 0], color = 'blue', label = 'Predicted Bitcoin Price')
plt.title('Bitcoin Price Prediction')
plt.xlabel('Time')
plt.ylabel('Bitcoin Price')
plt.legend()
plt.show()

#Metricas de error
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_price[:,0], predicted_price_list[:,0]))
print(rmse)