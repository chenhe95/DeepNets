import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

filename = "alice.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

epoch_n = 50
batch_size = 64
gen_length = 1000

n_chars = len(raw_text)
n_vocab = len(chars)
print "Total characters: ", n_chars
print "Total vocab: ", n_vocab

seq_length = 100
data_X = [[char_to_int[char] for char in raw_text[i:i + seq_length]]
	for i in xrange(0, n_chars - seq_length, 1)]

data_Y = [char_to_int[raw_text[i + seq_length]] 
	for i in xrange(0, n_chars - seq_length, 1)]

n_patterns = len(data_X)
print "Total patterns: ", n_patterns

X = np.reshape(data_X, (n_patterns, seq_length, 1))
X = X / float(n_vocab)
y = np_utils.to_categorical(data_Y)

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam")

filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-big.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor="loss", verbose=1, 
	save_best_only=True, mode="min")
callbacks_list = [checkpoint]

model.fit(X, y, nb_epoch=epoch_n, batch_size=batch_size, callbacks=callbacks_list)

filename = None
model.load_weights(filename)
model.compile(loss="categorical_crossentropy", optimizer="adam")
start = np.random.randint(0, len(data_X) - 1)
pattern = data_X[start]
print "Seed: "
print "\"", "".join([int_to_char[value] for value in pattern]), "\""
for i in range(gen_length):
	x = np.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = np.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print "\nDone"