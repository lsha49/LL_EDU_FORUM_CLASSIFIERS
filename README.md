Education forum post classification
-------------------------------------------------------------------------
The purpose of this repository is to explore text classification methods in 
education forum post classification

We also include here the Stanford forum post dataset: stanfordMOOCForumPostsSet.tar.gz. This dataset was originally created and labelled by the Stanford financed by the National Science Foundation to support educational research. The dataset can be used for research purpose.


@todo add EDM paper reference 

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
@todo



DL Usage: 
-------------------------------------------------------------------------------------------------------
1) model is in `xxx_classifier.py`
2) run python `train.py` to train the DL model
3) run python `test.py` to do test.

-------------------------------------------------------------------------
Road Map
-------------------------------------------------------------------------------------------------------
One way you can use this repository:
 
step 1: input file format


BERT pretrained embedding:
-------------------------------------------------------------------------------------------------------
@todo 