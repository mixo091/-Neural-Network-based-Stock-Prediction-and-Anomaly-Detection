import keras
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from keras import layers, optimizers, losses, metrics
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model
import os.path
import sys

#create a model
def createModel(trainX):
  model = Sequential()
  model.add(LSTM(64, activation='relu', return_sequences=True, input_shape=(trainX.shape[1], 1)))
  model.add(Dropout(0.2))
  model.add(LSTM(units = 32, return_sequences = True))
  model.add(Dropout(0.2))
  model.add(LSTM(units = 32))
  model.add(Dropout(0.2))

  model.add(Dense(1))

  model.compile(optimizer='adam', loss='mse')

  return model

def processData(data: np.array,lag,n_ahead=1):

    X,Y = [], []

    if len(data) - lag <= 0:
      X.append(data)
    else:
      for i in range(len(data) - lag - n_ahead):
        X.append(data[i:(i+lag), 0])
        Y.append(data[i+lag, 0])

    return np.array(X), np.array(Y)

# read args
dataset = sys.argv[2]
n = int(sys.argv[4])


# load dataset
series = pd.read_csv(dataset, header=0,sep='\t')
  
random_series = series.sample(n)
random_series = random_series.iloc[:, 1 : ].values

look_back_value = 5
n_ahead = 1

series = series.iloc[:, 1: ].values

# our training set
train_set = []
train_size = int(series.shape[1] * 0.8)

for i in range(series.shape[0]):

  train = series[i].reshape(-1,1)[0:train_size]

  sc = MinMaxScaler(feature_range = (0, 1))
  train = sc.fit_transform(train)

  train_set = np.concatenate((np.array(train_set).reshape(-1,1), train))

# create train set
trainX, trainY = processData(train_set, look_back_value, n_ahead=1)
trainX = np.reshape(trainX, (trainX.shape[0], look_back_value, 1))

#model = createModel(trainX)
model = keras.models.load_model('models/all-series_model.h5')
#model.fit(trainX, trainY, epochs = 100, batch_size=1024, shuffle=True)
model.save("models/all-series_model.h5")
    
for i in range(random_series.shape[0]):
  test = random_series[i]
  test = sc.transform(test.reshape(-1,1))

  testX, testY = processData(test, look_back_value, n_ahead=1)  
  testX = np.reshape(testX, (testX.shape[0], look_back_value, 1))

  predicted = model.predict(testX)
  print("predictions shape:", predicted.shape)

  scaler = MinMaxScaler(feature_range = (0, 1))
  scaler = scaler.fit(predicted)
  predicted = scaler.inverse_transform(predicted)

  plt.figure(figsize=(8, 8))
  plt.plot(test, color = 'red', label = 'Actual Values')
  plt.plot(predicted, color = 'blue', label = 'Predicted Values', alpha=0.6)
  plt.title('Train Dataset')
  plt.legend()
  plt.show()
