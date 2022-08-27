# -*- coding: utf-8 -*-
"""
Trabajo Final de Máster:
    Redes Neuronales Recurrentes aplicadas a criptomonedas
"""

# Parte 1 - Preprocesado de datos

#Librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# settingt he seed
np.random.seed(0)
tf.random.set_seed(0)

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
days = 90

for i in range(period, len(training_set)):
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
regressor.compile(optimizer= 'adam', loss = 'mean_squared_error', metrics=['mae']) #Documentación RNR Optimizador: RMSprop vs ADAM adam mejor

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', mode = 'min', verbose = 1, patience=10, restore_best_weights=True)   
#Conjunto entrenamiento recursivo
history = regressor.fit(X_train, y_train, epochs= 100, batch_size = 32, callbacks = [es])

#Observamos si los parametros del modelo son optimos


# summarize history for loss

plt.plot(history.history['loss'])
#plt.plot(min(history.history['loss']), marker="o")
#plt.plot(history.history['val_loss'])
#plt.title('Ritmo de aprendizaje')
plt.ylabel('Pérdida')
plt.xlabel('Epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Parte 3 - Ajustar las predicciones y visualizar los resultados

#Cargamos los datos del mes de enero-marzo de 2020
dataset_test = pd.read_csv("C:/Users/Daniel/Desktop/TFM/Datos/BTC_test_feb2020.csv")
real_price = dataset_test.iloc[:, [1,4]].values #Dataframe de 1 columna

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

#Predecimos el precio

predicted_price = regressor.predict(X_test)
predicted_price = np.append(predicted_price, ([[0]]*(91)), axis = 1)
predicted_price = sc.inverse_transform(predicted_price)

#Visualizacion de los datos
plt.plot(real_price[ : , 0], color = 'red', label = 'Precio Real del Bitcoin')
plt.plot(predicted_price[ : , 0], color = 'blue', label = 'Precio estimado del Bitcoin')
plt.title('Predicción del precio del Bitcoin')
plt.xlabel('Dias')
plt.ylabel('Precio del Bitcoin')
plt.legend()
plt.show()

#Metricas de error
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
rmse = math.sqrt(mean_squared_error(real_price[:,0], predicted_price[:,0]))
mae = mean_absolute_error(real_price[:,0], predicted_price[:,0])
print("RMSE:", rmse)
print("MAE:", mae)

#Vamos a validar este modelo

#Cargamos los datos de validación 1 de marzo 21 hasta 27 nov 22
dataset_val = pd.read_csv("C:/Users/Daniel/Desktop/TFM/Datos/BTC_val.csv")
real_price = dataset_val.iloc[:, [1,4]].values #Dataframe de 1 columna
real_price = real_price[days:days*2]

dataset_val = dataset_val[['Open','Close']]
inputs  = dataset_val[:len(dataset_val)].values #Formato fila
inputs = sc.transform(inputs) #Reescalamos los datos (el mínimo y el máximo se han obtenido del sc anterior.)

X_val = []
X_val2 = []
for i in range(0, days):
    X_val.append(inputs[i:i+period,0])
    X_val2.append(inputs[i:i+period,1])

X_val, X_val2= np.array(X_val), np.array(X_val2)

X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1)) #Tridimensionalizamos
X_val2 = np.reshape(X_val2, (X_val2.shape[0], X_val2.shape[1], 1))

#Unimos
X_val = np.append(X_val, X_val2, axis = 2)

#Predecimos el precio

predicted_price = regressor.predict(X_val)
predicted_price = np.append(predicted_price, ([[0]]*days), axis = 1)
predicted_price = sc.inverse_transform(predicted_price)

#Visualizacion de los datos

plt.plot(real_price[ : , 0], color = 'red', label = 'Precio Real del Bitcoin')
plt.plot(predicted_price[ : , 0], color = 'blue', label = 'Precio estimado del Bitcoin')
plt.title('Predicción del precio del Bitcoin')
plt.xlabel('Dias')
plt.ylabel('Precio del Bitcoin')
plt.legend()
plt.show()

rmse = math.sqrt(mean_squared_error(real_price[:,0], predicted_price[:,0]))
mae = mean_absolute_error(real_price[:,0], predicted_price[:,0])
print("RMSE:", rmse)
print("MAE:", mae)

#Ver los errores para DM
real_price = real_price[:,0].tolist()
predicted_price = predicted_price[:,0].tolist()
forecast_errors_30 = [real_price[j]-predicted_price[j] for j in range(len(real_price))]
print(forecast_errors_30)

df_30 = pd.DataFrame(forecast_errors_30)
df_30.to_csv('C:/Users/Daniel/Desktop/TFM/error_30.csv', index=False, header = ["Data"])

dataset_30 = pd.read_csv("C:/Users/Daniel/Desktop/TFM/error_30.csv")
list_30 = dataset_30['Data'].values.tolist()

