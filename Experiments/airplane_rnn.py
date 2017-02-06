from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

dataframe = read_csv("international-airline-passengers.csv", usecols=[1], engine="python", skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype("float32")
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

np.random.seed(7)

train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

def create_dataset(dataset, look_back=1):
	data_X, data_Y = [], []
	for i in xrange(len(dataset) - look_back - 1):
		a = dataset[i:(i + look_back), 0]
		data_X.append(a)
		data_Y.append(dataset[i + look_back, 0])
	return np.array(data_X), np.array(data_Y)

look_back = 3
train_X, train_Y = create_dataset(train, look_back=look_back)
test_X, test_Y = create_dataset(test, look_back=look_back)

train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

model = Sequential()
model.add(LSTM(4, input_dim=1))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")
model.fit(train_X, train_Y, nb_epoch=100, batch_size=2, verbose=2)

train_predict = model.predict(train_X)
test_predict = model.predict(test_X)

train_predict = scaler.inverse_transform(train_predict)
train_Y = scaler.inverse_transform([train_Y])
test_predict = scaler.inverse_transform(test_predict)
test_Y = scaler.inverse_transform([test_Y])

train_score = math.sqrt(mean_squared_error(train_Y[0], train_predict[:, 0]))
test_score = math.sqrt(mean_squared_error(test_Y[0], test_predict[:, 0]))
print "Train Score: (%.2f RMSE)" % (train_score)
print "Test Score: (%.2f RMSE)" % (test_score)

train_predict_plot = np.empty_like(dataset)
train_predict_plot[:, :] = np.nan
train_predict_plot[look_back:len(train_predict) + look_back, :] = train_predict

test_predict_plot = np.empty_like(dataset)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + 2 * look_back + 1:len(dataset) - 1] = test_predict

plt.plot(scaler.inverse_transform(dataset))
plt.plot(train_predict_plot)
plt.plot(test_predict_plot)
plt.show()