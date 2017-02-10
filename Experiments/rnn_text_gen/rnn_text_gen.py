import numpy as np
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
