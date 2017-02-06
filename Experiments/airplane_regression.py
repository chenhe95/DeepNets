from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
import math

dataframe = read_csv("international-airline-passengers.csv", usecols=[1], engine="python", skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype("float32")

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

look_back = 1
train_X, train_Y = create_dataset(train, look_back=look_back)
test_X, test_Y = create_dataset(test, look_back=look_back)

model = Sequential()
model.add(Dense(8, input_dim=look_back, activation="relu"))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")
model.fit(train_X, train_Y, nb_epoch=200, batch_size=2, verbose=2)

train_score = model.evaluate(train_X, train_Y, verbose=0)
print "Train Score: %.2f MSE (%.2f RMSE)" % (train_score, math.sqrt(train_score))
test_score = model.evaluate(test_X, test_Y, verbose=0)
print "Test Score: %.2f MSE (%.2f RMSE)" % (test_score, math.sqrt(test_score))

train_predict = model.predict(train_X)
test_predict = model.predict(test_X)

train_predict_plot = np.empty_like(dataset)
train_predict_plot[:, :] = np.nan
train_predict_plot[look_back:len(train_predict) + look_back, :] = train_predict

test_predict_plot = np.empty_like(dataset)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + 2 * look_back + 1:len(dataset) - 1] = test_predict

plt.plot(dataset)
plt.plot(train_predict_plot)
plt.plot(test_predict_plot)
plt.show()