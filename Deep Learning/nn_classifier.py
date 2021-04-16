# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from bert_serving.client import BertClient
from nltk import tokenize
from pprint import pprint


class nn_clf(object):
    """
    A NN classifier for text classification
    Reference: A NN Neural Network for Text Classification
    """
    def __init__(self, config):
        self.max_length = config.max_length
        self.num_classes = config.num_classes
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.filter_sizes = list(map(int, config.filter_sizes.split(",")))
        self.num_filters = config.num_filters
        self.hidden_size = len(self.filter_sizes) * self.num_filters
        self.num_layers = config.num_layers
        self.l2_reg_lambda = config.l2_reg_lambda

        # Placeholders
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name='batch_size')
        self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, 40000], name='input_x')
        self.input_y = tf.placeholder(dtype=tf.int64, shape=[None], name='input_y')
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
        self.sequence_length = tf.placeholder(dtype=tf.int32, shape=[None], name='sequence_length')

        # L2 loss
        self.l2_loss = tf.constant(0.0)        
        inputs = self.input_x
        
        # Input dropout
        inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob)

        # Softmax output layer
        with tf.name_scope('softmax'):
            # multi label
            W = tf.Variable(tf.zeros([40000, 8])) 
            b = tf.Variable(tf.zeros([8])) # number of classes

            self.logits = tf.nn.softmax(tf.matmul(inputs, W) + b) # Softmax 
            predictions = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(predictions, 1, name='predictions')

        # Loss
        with tf.name_scope('loss'):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            self.cost = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss

        # Accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.correct_num = tf.reduce_sum(tf.cast(correct_predictions, tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')
