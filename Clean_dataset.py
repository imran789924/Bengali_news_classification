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


stops_bengali = stops.stops_bengali


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
    t = [lemmatizer.lemmatize(word, pos = lem.POS_NOUN) for word in t if not word in set(stops_bengali)]
    #review = ' '.join(t)
    t = [pat.sub('', word) for word in t]
    t = [word for word in t if len(word) > 1]
    corpus.append(t)
    vocab.update(t)


print(len(vocab))
# print the top words in the vocab
print(vocab.most_common(50))


# keep tokens with a min occurrence
min_occurane = 2
tokens = [k for k,c in vocab.items() if c >= min_occurane]
print(len(tokens))


with open('vocab_bangla.txt', 'w') as f:
    for item in vocab:
        f.write("%s\n" % item)
# close file
file.close()