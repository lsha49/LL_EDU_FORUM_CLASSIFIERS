Education forum post classification
-------------------------------------------------------------------------
The purpose of this repository is to explore text classification methods in 
education forum post classification. This repo is resulted from this [paper](https://files.eric.ed.gov/fulltext/ED615664.pdf): [Which Hammer should I Use? A Systematic Evaluation of Approaches for Classifying Educational Forum Posts](https://scholar.googleusercontent.com/scholar.bib?q=info:w-ycr6cO4LoJ:scholar.google.com/&output=citation&scisdr=CgVL0huFEN6jhKmtSpQ:AAGBfm0AAAAAYzGrUpT0ZqJDrMNreZGfFoumBsDZz9qj&scisig=AAGBfm0AAAAAYzGrUihwH0i3R_KMAkWkU40AkKs0GI48&scisf=4&ct=citation&cd=-1&hl=en). We kindly ask you to reference this paper when applying this repo. 

We also include here the Stanford forum post dataset: stanfordMOOCForumPostsSet.tar.gz. This dataset was originally created and labelled by the Stanford financed by the National Science Foundation to support educational research. The dataset can be used for research purpose.

#### Introduction
Forum post classification is a long-standing task in the field of educational research. To help ease the effort and aid future research, here we provide commonly used ML (Machine Learning) and DL (Deep Learning) model implemenatation code. (note: DL code is modified from repo in here: https://github.com/zackhy/TextClassification, while text preprocessing partially used code by https://medium.com/@bedigunjit/)

Models:
-------------------------------------------------------------------------
1) Naive bayes: ml_classifiers.nb_clf
2) Logistic regression: ml_classifiers.lr_clf
3) Random forest: ml_classifiers.rf_clf
4) Support vector machine: ml_classifiers.svm_clf
5) CLSTM: clstm_classifier
6) BLSTM: rnn_classifier


## Requirements  
-------------------------------------------------------------------------------------------------------
* Python 3.x  
* Tensorflow > 1.5
* Sklearn > 0.19.0  

ML Usage: 
-------------------------------------------------------------------------------------------------------
ML code is contained in Traditional_Machine_Learning.ml_classifiers

create configuration: 
config = dict()
config['testSize'] = 0.2
config['file'] = 'xxx.csv'

* initialise base classifier:
classifier = ml_clf(config)

* create a Naive bayes classifier: 
classifier.nb_clf()

* create a SVM classifier:
classifier.svm_clf()

* create a Logistic regression classifier:
classifier.lr_clf()

* create a Random forest classifier:
classifier.rf_clf()

* to perform a simple grid search with pre-defined parameters:
grid = GridSearchCV(YOUR_MODEL,YOUR_SEARCH_PARAMS,refit=True,verbose=2)
grid.fit(self.Train_X,self.Train_Y)
print(grid.best_estimator_)

* to run a model with hyperparamter, replace model function and add parameter: 
e.g., 
replace: 
rfc=RandomForestClassifier()

with: 
rfc=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
  max_depth=None, max_features=0.5, max_leaf_nodes=None,
  min_impurity_split=1e-07, min_samples_leaf=1,
  min_samples_split=4, min_weight_fraction_leaf=0.0,
  n_estimators=250, n_jobs=1, oob_score=False, random_state=None,
  verbose=0, warm_start=False)



BERT pretrained embedding:
-------------------------------------------------------------------------------------------------------
We used a service called "Bert-as-a-service" (https://github.com/hanxiao/bert-as-service) to generate BERT embeddings of the forum post. 
The embedding is then used as input for DL models


DL Usage: 
-------------------------------------------------------------------------------------------------------
We refer this repo: https://github.com/zackhy/TextClassification, where DL code was modified from. 

1) model is in `xxx_classifier.py`
2) run python `train.py` to train the DL model
3) run python `test.py` to do test.


