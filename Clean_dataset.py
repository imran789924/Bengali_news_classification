#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 06:21:21 2021

@author: imran
"""

import importlib
import re
import nltk
import pandas as pd
from banglakit import lemmatizer as lem
from banglakit.lemmatizer import BengaliLemmatizer
from tensorflow.keras.preprocessing.text import one_hot
from collections import Counter

moduleName = 'stopwords'
stops = importlib.import_module(moduleName)


stops_bengali = stops.stops_bengali_


df = pd.read_csv('archive/train.csv')
sentences = list(df.iloc[:, 1])
sentences.extend(list(df.iloc[:,0]))


corpus = []
vocab = Counter()
for i in range(0, len(sentences)):
    pat = re.compile(r'[^\u0980-\u09E5\u09F0-\u09FF]', re.UNICODE)
    t = pat.sub(' ', sentences[i])
    t = t.split()
    lemmatizer = BengaliLemmatizer()
    #t = [ps.setm(word) for word in t if not word in set(stopwords.word('english'))]
    t = [lemmatizer.lemmatize(word, pos = lem.POS_NOUN) for word in t if not lemmatizer.lemmatize(word, pos = lem.POS_NOUN) in set(stops_bengali)]
    #review = ' '.join(t)
    t = [pat.sub('', word) for word in t]
    t = [word for word in t if len(word) > 1]
    corpus.append(t)
    vocab.update(t)


print(len(vocab))
# print the top words in the vocab
print(vocab.most_common(50))


# keep tokens with a min occurrence
min_occurane = 3
tokens = [k for k,c in vocab.items() if c >= min_occurane]
print(len(tokens))


with open('vocab_bangla.txt', 'w') as f:
    for item in vocab:
        f.write("%s\n" % item)
# close file
f.close()


import pickle

with open("corpus.txt", "wb") as fp:   #Pickling
    pickle.dump(corpus, fp)


with open("corpus.txt", "rb") as fp:   # Unpickling
    corpus = pickle.load(fp)

from gensim.models import Word2Vec
model = Word2Vec(sentences = corpus, vector_size=300, window = 5, min_count = 1, workers = 4)

words = list(model.wv.index_to_key)
print('Vocabulary size: %d' % len(words))
 
# save model in ASCII (word2vec) format
filename = 'embedding_word2vec.txt'
model.wv.save_word2vec_format(filename, binary=False)