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
def createModel(lag):

 model = Sequential()
  #Adding the first LSTM layer and some Dropout regularisation
 model.add(LSTM(units = 64, return_sequences = True, input_shape = (lag, 1)))
 model.add(Dropout(0.2))
  # Adding a second LSTM layer and some Dropout regularisation
#model.add(LSTM(units = 64, return_sequences = True))
# model.add(Dropout(0.2))
  # Adding a third LSTM layer and some Dropout regularisation
#  model.add(LSTM(units = 50, return_sequences = True))
#  model.add(Dropout(0.2))
  # Adding a fourth LSTM layer and some Dropout regularisation
 model.add(LSTM(units = 50))
 model.add(Dropout(0.2))
  # Adding the output layer
 model.add(Dense(units = 1))

  # Compiling the RNN
 model.compile(optimizer = 'adam', loss = 'mean_squared_error')
 #model.compile(optimizer='Adagrad', loss = 'mean_squared_error')

 return model


def processData(data: np.array, lag, size):

  X,Y = [], []

  for i in range(lag, size):
    X.append(data[i-lag:i, 0])
    Y.append(data[i, 0])

  return np.array(X), np.array(Y)

# read args
dataset = sys.argv[2]
n = int(sys.argv[4])


# load dataset
series = pd.read_csv(dataset, header=0,sep='\t')
  
random_series = series.sample(n)
random_series = random_series.iloc[:, 1 : ].values

look_back_value = 10
n_ahead = 1
train_size = int(random_series.shape[1] * 0.8)
test_size = int(random_series.shape[1] - train_size)
    
for i in range(random_series.shape[0]):
  
  # train, test = random_series[i].reshape(-1,1)[0:train_size], random_series[i].reshape(-1,1)[train_size: random_series.shape[1]]
  train = random_series[i][0 : train_size].reshape(-1,1)

  sc = MinMaxScaler(feature_range = (0, 1))
  train = sc.fit_transform(train)
  trainX, trainY = processData(train, look_back_value, train_size)
  trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
  print('trainX=>', trainX.shape)
 
  model = keras.models.load_model('models/my_model.h5')
  #model = createModel(look_back_value)
  #model.fit(trainX, trainY, epochs=100, batch_size = 32)

  model.save("models/my_model.h5")

  # create test_set
  test_set = random_series[i][train_size: random_series.shape[1]]
  
  test = random_series[i][random_series.shape[1] - test_size - look_back_value:]
  test = sc.transform(test.reshape(-1,1))
  print('test=>', test.shape)
  testX = []
  for i in range(look_back_value, test.shape[0]):
    testX.append(test[i-look_back_value:i, 0])
  testX = np.array(testX)
  testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
 
  # testX, testY = processData(test, look_back_value, random_series.shape[1] - test_size - look_back_value)
  testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

  predicted = model.predict(testX)
  predicted = sc.inverse_transform(predicted)

  print('predicted = ', predicted.shape)
  print('dataset =' ,test_set.shape)

  plt.figure(figsize=(12, 8))
  plt.plot(test_set.reshape(-1,1), color = 'red', label = 'Actual Values')
  plt.plot(predicted, color = 'blue', label = 'Predicted Values')
  # plt.plot(testY, color = 'green', label = 'Real Values')
  plt.title('Train Dataset')
  plt.legend()
  plt.show()
