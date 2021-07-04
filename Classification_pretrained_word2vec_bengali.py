#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 08:37:18 2021

@author: imran
"""


import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import asarray
from keras.preprocessing.text import Tokenizer
from numpy import zeros
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.models import Sequential




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
        t_body = [t for t in t_body if t in vocab]
        t_body = ' '.join(t_body)
        t_head = ' '.join(t_head)
        corpus_body.append(t_body)
        corpus_head.append(t_head)


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


vocab_filename = 'vocab_bangla.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)


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
 
# define model
model = Sequential()
model.add(embedding_layer_body)
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(6, activation='softmax'))
print(model.summary())
# compile network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(X_body, y, epochs=20, verbose=2)