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
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV


class ml_clf(object):
    def __init__(self, config):
        useLabel = 'Label'  # classification label header name in CSV file
        testSize = config['testSize'] # test size in train-test ratio
        Corpus = pd.read_csv(config['file'], encoding='latin-1') # CSV file containing posts
        
        # preprocessing texts: stopwords, lemmatize. 
        Corpus = self.populateFinalText(Corpus)

        # populate TFIDF feature in Corpus
        Corpus = self.populateTfIdf(Corpus)

        # Labels
        corpusY = Corpus[useLabel]

        # features
        Corpus.drop('Text', inplace=True, axis=1)
        Corpus.drop(useLabel, inplace=True, axis=1)
        corpusX = Corpus
        corpusX = corpusX.replace(np.nan, 0)

        # standardise feature value if required
        min_max_scaler = preprocessing.MinMaxScaler()
        corpusX = min_max_scaler.fit_transform(corpusX)

        self.Train_X, self.Test_X, self.Train_Y, self.Test_Y = model_selection.train_test_split(corpusX,corpusY, test_size=testSize, random_state=11, stratify=corpusY)
        self.Train_Y = self.Train_Y.astype(int)
        self.Test_Y = self.Test_Y.astype(int)


    def populateTfIdf(self, Corpus):
        numOfGramFeatures = 500
        Tfidf_vect = TfidfVectorizer(max_features=numOfGramFeatures, ngram_range=(2,2))
        Tfidf_vect.fit(Corpus['text_final'])
        TfidfFeature = Tfidf_vect.transform(Corpus['text_final'])
        Tfidf_vect_uni = TfidfVectorizer(max_features=numOfGramFeatures, ngram_range=(1,1))
        Tfidf_vect_uni.fit(Corpus['text_final'])
        TfidfFeatureUni = Tfidf_vect_uni.transform(Corpus['text_final'])
        tfidfHeaders = Tfidf_vect.vocabulary_
        tfidfHeadersUni = Tfidf_vect_uni.vocabulary_
        keyedIndex = 0
        for key,entry in tfidfHeaders.items():
            Corpus['tf ' + key] = TfidfFeature.getcol(keyedIndex).toarray()
            keyedIndex = keyedIndex + 1
        keyedIndex = 0
        for key,entry in tfidfHeadersUni.items():
            Corpus['tf ' + key] = TfidfFeatureUni.getcol(keyedIndex).toarray()
            keyedIndex = keyedIndex + 1
        Corpus.drop('text_final', inplace=True, axis=1)
        return Corpus


    def populateFinalText(self, Corpus): 
        Corpus['Text'].dropna(inplace=True)
        Corpus['Text'] = [ entry.lower() if isinstance(entry, str) else entry for entry in Corpus['Text'] ]
        Corpus['Text']=Corpus['Text'].replace(to_replace= r'\\', value= '', regex=True)
        Corpus['Text'] = [ entry if isinstance(entry, str) else entry for entry in Corpus['Text'] ]
        tag_map = defaultdict(lambda : wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV
        for index,entry in enumerate(Corpus['Text']):
            Final_words = []
            word_Lemmatized = WordNetLemmatizer()
            if isinstance(entry, Iterable):
                wordList = tokenize.word_tokenize(entry)
                for word, tag in pos_tag(wordList):
                    if word not in stopwords.words('english') and word.isalpha():
                        word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                        Final_words.append(word_Final)
            Corpus.loc[index,'text_final'] = str(Final_words)
        return Corpus


    def nb_clf(self):
        GaussianNaive = naive_bayes.GaussianNB()
        GaussianNaive.fit(self.Train_X,self.Train_Y)
        Gpredictions_NB = GaussianNaive.predict(self.Test_X)
        print("Gaussian Naive Bayes Accuracy Score -> ",accuracy_score(Gpredictions_NB, self.Test_Y))
        print("GaussianNaive Bayes Kappa Score -> ",cohen_kappa_score(Gpredictions_NB, self.Test_Y))
        print("Gaussian Naive Bayes ROC AUC Score -> ",roc_auc_score(self.Test_Y, Gpredictions_NB))
        print("Gaussian Naive Bayes F1 Score -> ",f1_score(Gpredictions_NB, self.Test_Y))


    def svm_clf(self):
        # A sample GridSearched model: 
        # SVM = svm.SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0,
        # decision_function_shape=None, degree=3, gamma=0.01, kernel='rbf',
        # max_iter=-1, probability=False, random_state=None, shrinking=True,
        # tol=0.001, verbose=False)
        SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
        SVM.fit(self.Train_X, self.Train_Y)
        predictions_SVM = SVM.predict(self.Test_X)
        print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, self.Test_Y))
        print("SVM Kappa Score -> ",cohen_kappa_score(predictions_SVM, self.Test_Y))
        print("SVM ROC AUC Score -> ",roc_auc_score(predictions_SVM, self.Test_Y))
        print("SVM F1 Score -> ",f1_score(predictions_SVM, self.Test_Y))

    def rf_clf(self):
        # A sample GridSearched model: 
        # rfc=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
        #     max_depth=None, max_features=0.5, max_leaf_nodes=None,
        #     min_impurity_split=1e-07, min_samples_leaf=1,
        #     min_samples_split=4, min_weight_fraction_leaf=0.0,
        #     n_estimators=250, n_jobs=1, oob_score=False, random_state=None,
        #     verbose=0, warm_start=False)
        rfc=RandomForestClassifier(n_estimators=100)
        rfc.fit(self.Train_X,self.Train_Y)
        predictions_rfc = rfc.predict(self.Test_X)
        print("Random forest Accuracy Score -> ",accuracy_score(predictions_rfc, self.Test_Y))
        print("Random forest Kappa Score -> ",cohen_kappa_score(predictions_rfc, self.Test_Y))
        print("Random forest ROC AUC Score -> ",roc_auc_score(predictions_rfc, self.Test_Y))
        print("Random forest F1 Score -> ",f1_score(predictions_rfc, self.Test_Y))

    def lr_clf(self):
        # lrc = LogisticRegression(C=4.281332398719396, class_weight=None, dual=False,
        #   fit_intercept=True, intercept_scaling=1, max_iter=100,
        #   multi_class='ovr', n_jobs=1, penalty='l1', random_state=None,
        #   solver='liblinear', tol=0.0001, verbose=0, warm_start=False)
        lrc=LogisticRegression(random_state=0)
        lrc.fit(self.Train_X,self.Train_Y)
        predictions_lrc = lrc.predict(self.Test_X)
        print("Logistic regression Accuracy Score -> ",accuracy_score(predictions_lrc, self.Test_Y))
        print("Logistic regression Kappa Score -> ",cohen_kappa_score(predictions_lrc, self.Test_Y))
        print("Logistic regression ROC AUC Score -> ",roc_auc_score(predictions_lrc, self.Test_Y))
        print("Logistic regression F1 Score -> ",f1_score(predictions_lrc, self.Test_Y))
