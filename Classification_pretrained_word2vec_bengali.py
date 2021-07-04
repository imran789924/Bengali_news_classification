#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 08:37:18 2021

@author: imran
"""


import importlib
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import asarray
from keras.preprocessing.text import Tokenizer
from numpy import zeros
from keras.preprocessing.sequence import pad_sequences
from banglakit import lemmatizer as lem
from banglakit.lemmatizer import BengaliLemmatizer
from tensorflow.keras.preprocessing.text import one_hot
from keras.layers import Embedding
from keras.models import Sequential
import re




def load_embedding(filename):
	# load embedding into memory, skip first line
	file = open(filename,'r')
	lines = file.readlines()[1:]
	file.close()
	# create a map of words to vectors
	embedding = dict()
	for line in lines:
		parts = line.split()
		# key is string word, value is numpy array for vector
		embedding[parts[0]] = asarray(parts[1:], dtype='float32')
	return embedding


tokenizer = Tokenizer()
def get_weight_matrix(embedding, vocab):
	# total vocabulary size plus 0 for unknown words
	vocab_size = len(vocab) + 1
	# define weight matrix dimensions with all 0
	weight_matrix = zeros((vocab_size, 300))
	# step vocab, store vectors using the Tokenizer's integer mapping
	for word, i in vocab.items():
		weight_matrix[i] = embedding.get(word)
	return weight_matrix


def process_data(X_head, X_body, corpus_head, corpus_body):
    for i in range(0, len(X_head)):
        pat = re.compile(r'[^\u0980-\u09E5\u09F0-\u09FF]', re.UNICODE)
        t_head = pat.sub(' ', X_head[i])
        t_body = pat.sub(' ', X_body[i])
        t_head = t_head.split()
        t_body = t_body.split()
        lemmatizer = BengaliLemmatizer()
        #t = [ps.setm(word) for word in t if not word in set(stopwords.word('english'))]
        t_body = [lemmatizer.lemmatize(word, pos = lem.POS_NOUN) for word in t_body if not lemmatizer.lemmatize(word, pos = lem.POS_NOUN) in set(stops_bengali)]
        t_head = [lemmatizer.lemmatize(word, pos = lem.POS_NOUN) for word in t_head if not lemmatizer.lemmatize(word, pos = lem.POS_NOUN) in set(stops_bengali)]

        t_head = [pat.sub('', word) for word in t_head]
        t_head = [pat.sub('', word) for word in t_head]
        t_head = [word for word in t_head if len(word) > 1]
        t_body = [word for word in t_body if len(word) > 1]
        
        t_body = [t for t in t_head if t in vocab]
        t_body = [t for t in t_body if t in vocab]
        
        t_body = ' '.join(t_body)
        t_head = ' '.join(t_head)
        corpus_body.append(t_body)
        corpus_head.append(t_head)

vocab_filename = 'vocab_bangla.txt'
file = open(vocab_filename, 'r')
vocab = file.read()
file.close()
vocab = vocab.split()
vocab = set(vocab)



moduleName = 'stopwords'
stops = importlib.import_module(moduleName)
stops_bengali = stops.stops_bengali_

##############TRAINING DATA, NEWS HEAD AND BODY##########################

df = pd.read_csv('archive/train.csv')

df.loc[df['label'] == 'sport', 'label'] = 'sports'
df.loc[df['label'] == 'nation', 'label'] = 'national'
df.loc[df['label'] == 'world', 'label'] = 'international'
df.loc[df['label'] == 'travel', 'label'] = 'entertainment'

X_head = df.iloc[:, 0]
X_body = df.iloc[:, 1]
y = df.iloc[:, 2].values.reshape(-1, 1)

corpus_head = []
corpus_body = []

process_data(X_head, X_body, corpus_head, corpus_body)


# create the tokenizer
tokenizer_head = Tokenizer()
tokenizer_body = Tokenizer()
# fit the tokenizer on the documents
tokenizer_head.fit_on_texts(corpus_head)
tokenizer_body.fit_on_texts(corpus_body)

# sequence encode
encoded_docs_body = tokenizer_head.texts_to_sequences(X_body)
encoded_docs_head = tokenizer_body.texts_to_sequences(X_head)
# pad sequences
max_length_body = max([len(s.split()) for s in X_body])
max_length_head = max([len(s.split()) for s in X_head])

X_head = pad_sequences(encoded_docs_head, maxlen=max_length_head, padding='post')
X_body = pad_sequences(encoded_docs_body, maxlen=max_length_body, padding='post')

# define training labels

vocab_size_head = len(tokenizer_head.word_index) + 1
vocab_size_body = len(tokenizer_body.word_index) + 1

raw_embedding = load_embedding('embedding_word2vec.txt')

embedding_vectors_head = get_weight_matrix(raw_embedding, tokenizer_head.word_index)
embedding_vectors_body = get_weight_matrix(raw_embedding, tokenizer_body.word_index)


from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
y = enc.fit_transform(y)
y = np_utils.to_categorical(y)


embedding_layer_head = Embedding(vocab_size_head, 300, weights=[embedding_vectors_head], input_length=max_length_head, trainable=False)
embedding_layer_body = Embedding(vocab_size_body, 300, weights=[embedding_vectors_body], input_length=max_length_body, trainable=False)


from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D



model = Sequential()
#model.add(embedding_layer_head)
model.add(embedding_layer_body)
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
#model.add(Dense(10, activation='relu'))
model.add(Dense(6, activation='softmax'))
print(model.summary())
# compile network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(X_body, y, epochs=10, verbose=2)

model.save('trained_model.h5')

from keras.models import load_model

model = load_model('trained_model.h5')
model.summary()



##############TEST DATA, NEWS HEAD AND BODY##########################


df_test = pd.read_csv('archive/valid.csv')

df_test.loc[df_test['label'] == 'sport', 'label'] = 'sports'
df_test.loc[df_test['label'] == 'nation', 'label'] = 'national'
df_test.loc[df_test['label'] == 'world', 'label'] = 'international'
df_test.loc[df_test['label'] == 'travel', 'label'] = 'entertainment'

X_head_test = df_test.iloc[:, 0]
X_body_test = df_test.iloc[:, 1]
y_test = df_test.iloc[:, 2].values.reshape(-1, 1)

corpus_head_test = []
corpus_body_test = []

process_data(X_head_test, X_body_test, corpus_head_test, corpus_body_test)


# sequence encode
encoded_docs_body_test = tokenizer_head.texts_to_sequences(X_body_test)
encoded_docs_head_test = tokenizer_body.texts_to_sequences(X_head_test)
# pad sequences

X_head_test = pad_sequences(encoded_docs_head_test, maxlen=max_length_head, padding='post')
X_body_test = pad_sequences(encoded_docs_body_test, maxlen=max_length_body, padding='post')

y_test = enc.transform(y_test)
y_test = np_utils.to_categorical(y_test)

loss, acc = model.evaluate(X_body_test, y, verbose=0)
print('Test Accuracy: %f' % (acc*100))

y_pred = model.predict(X_body_test)

y_pred = y_pred.argmax(1)
y_test = y_test.argmax(1)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

###############Debugging###################
print(model.layers)
cw1 = np.array(model.layers[1].get_weights())
print(cw1.shape) # 2 weight, 1 weight, 1 bias
print(cw1[0].shape) # 3 channels, 3 by 3 kernels, 32 filters
print(cw1[1].shape) # 32 biases
