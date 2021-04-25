import pandas as pd
import numpy as np
import time
import gensim
import logging
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from collections.abc import Iterable
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from gensim.models import Word2Vec 
from pprint import pprint
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest, f_classif, chi2
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from nltk import tokenize
import math
import textstat
from sklearn.feature_extraction.text import CountVectorizer

class util(object):
    def feature_readability(self, data):
        return textstat.automated_readability_index(data)

    def feature_ngram(self, data, numOfGramFeatures):
        ngram_vectorizer_uni = CountVectorizer(max_features=numOfGramFeatures, ngram_range=(1,1)) # unigram 
        return ngram_vectorizer_uni.fit_transform(data)
    
    def feature_tfidf(self, data, numOfGramFeatures):
        Tfidf_vect = TfidfVectorizer(max_features=numOfGramFeatures, ngram_range=(1,1))
        Tfidf_vect.fit(data)
        return Tfidf_vect.transform(data)

    def exam_selectKbest(self, data, X, Y, numOfGramFeatures):
        kval = 0
        if kval > 0:
            selector = SelectKBest(chi2, k=kval)
            selector.fit_transform(X, Y)
            alabels = data.columns.tolist()
            slabels = selector.get_support()
            print('Selected K features: ==============')
            for aindex in range(len(alabels)):
                if slabels[aindex] == True:
                    print(alabels[aindex], ',' ,end = '')
            print('\n')
            