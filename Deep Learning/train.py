# -*- coding: utf-8 -*-
import os
import sys
import csv
import time
import json
import datetime
import pickle as pkl
import tensorflow as tf
from tensorflow.contrib import learn

import data_helper
from rnn_classifier import rnn_clf
from cnn_classifier import cnn_clf
from nn_classifier import nn_clf
from clstm_classifier import clstm_clf
from bert_serving.client import BertClient
from nltk import tokenize
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Parameters
# =============================================================================

# Model choices
tf.flags.DEFINE_string('clf', 'cnn', "Type of classifiers. Default: cnn. You have four choices: [cnn, lstm, blstm, clstm]")

tf.flags.DEFINE_string('useEmbed', 'bert_blstm', "Type of embedding to use")

# Data parameters
tf.flags.DEFINE_string('data_file', None, 'Data file path')
tf.flags.DEFINE_string('stop_word_file', None, 'Stop word file path')
tf.flags.DEFINE_string('language', 'en', "Language of the data file. You have two choices: [ch, en]")
tf.flags.DEFINE_integer('min_frequency', 0, 'Minimal word frequency')
tf.flags.DEFINE_integer('num_classes', 0, 'Number of classes')
tf.flags.DEFINE_integer('max_length', 0, 'Max document length')
tf.flags.DEFINE_integer('vocab_size', 0, 'Vocabulary size')
tf.flags.DEFINE_float('test_size', 0, 'Cross validation test size')

# Model hyperparameters
tf.flags.DEFINE_integer('embedding_size', 0, 'Word embedding size.')
tf.flags.DEFINE_string('filter_sizes', '5', 'CNN filter sizes.')

tf.flags.DEFINE_integer('num_filters', 128, 'Number of filters per filter size.')
tf.flags.DEFINE_integer('hidden_size', 128, 'Number of hidden units in the LSTM cell.')
tf.flags.DEFINE_integer('num_layers', 2, 'Number of the LSTM cells.')
tf.flags.DEFINE_float('keep_prob', 0.5, 'Dropout keep probability')  # All
tf.flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate')  # All
tf.flags.DEFINE_float('l2_reg_lambda', 0.001, 'L2 regularization lambda')  # All

# Training parameters
tf.flags.DEFINE_integer('batch_size', 32, 'Batch size')
tf.flags.DEFINE_integer('num_epochs', 50, 'Number of epochs')
tf.flags.DEFINE_float('decay_rate', 1, 'Learning rate decay rate. Range: (0, 1]')  
tf.flags.DEFINE_integer('decay_steps', 100000, 'Learning rate decay steps') 
tf.flags.DEFINE_integer('evaluate_every_steps', 100, 'Evaluate the model on validation set after this many steps')
tf.flags.DEFINE_integer('save_every_steps', 1000, 'Save the model after this many steps')
tf.flags.DEFINE_integer('num_checkpoint', 10, 'Number of models to store')

FLAGS = tf.app.flags.FLAGS

if FLAGS.clf == 'lstm':
    FLAGS.embedding_size = FLAGS.hidden_size
elif FLAGS.clf == 'clstm':
    FLAGS.hidden_size = len(FLAGS.filter_sizes.split(",")) * FLAGS.num_filters

# Output files directory
timestamp = str(int(time.time()))
outdir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
if not os.path.exists(outdir):
    os.makedirs(outdir)

# Load and save data
# =============================================================================

data, labels, lengths, vocab_processor = data_helper.load_data(file_path=FLAGS.data_file,
                                                               sw_path=FLAGS.stop_word_file,
                                                               min_frequency=FLAGS.min_frequency,
                                                               max_length=FLAGS.max_length,
                                                               language=FLAGS.language,
                                                               shuffle=True)

params = FLAGS.flag_values_dict()
# Print parameters
model = params['clf']
if model == 'cnn':
    del params['hidden_size']
    del params['num_layers']
elif model == 'lstm' or model == 'blstm':
    del params['num_filters']
    del params['filter_sizes']
    params['embedding_size'] = params['hidden_size']
elif model == 'clstm':
    params['hidden_size'] = len(list(map(int, params['filter_sizes'].split(",")))) * params['num_filters']

params_dict = sorted(params.items(), key=lambda x: x[0])
print('Parameters:')
for item in params_dict:
    print('{}: {}'.format(item[0], item[1]))
print('')

# Save parameters to file
params_file = open(os.path.join(outdir, 'params.pkl'), 'wb')
pkl.dump(params, params_file, True)
params_file.close()


# Simple Cross validation
x_train, x_valid, y_train, y_valid, train_lengths, valid_lengths = train_test_split(data,
                                                                                    labels,
                                                                                    lengths,
                                                                                    test_size=FLAGS.test_size,
                                                                                    random_state=22)
# Batch iterator
train_data = data_helper.batch_iter(x_train, y_train, train_lengths, FLAGS.batch_size, FLAGS.num_epochs)

# Train
# =============================================================================

with tf.Graph().as_default():
    with tf.Session() as sess:
        if FLAGS.clf == 'cnn':
            classifier = cnn_clf(FLAGS)
        elif FLAGS.clf == 'nn':
            classifier = nn_clf(FLAGS)
        elif FLAGS.clf == 'lstm' or FLAGS.clf == 'blstm':
            classifier = rnn_clf(FLAGS)
        elif FLAGS.clf == 'clstm':
            classifier = clstm_clf(FLAGS)
        else:
            raise ValueError('clf should be one of [cnn, lstm, blstm, clstm]')

        # Train procedure
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Learning rate decay
        starter_learning_rate = FLAGS.learning_rate
        learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                   global_step,
                                                   FLAGS.decay_steps,
                                                   FLAGS.decay_rate,
                                                   staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(classifier.cost)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Summaries
        loss_summary = tf.summary.scalar('Loss', classifier.cost)
        accuracy_summary = tf.summary.scalar('Accuracy', classifier.accuracy)

        # Train summary
        train_summary_op = tf.summary.merge_all()
        train_summary_dir = os.path.join(outdir, 'summaries', 'train')
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Validation summary
        valid_summary_op = tf.summary.merge_all()
        valid_summary_dir = os.path.join(outdir, 'summaries', 'valid')
        valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, sess.graph)

        saver = tf.train.Saver(max_to_keep=FLAGS.num_checkpoint)

        sess.run(tf.global_variables_initializer())


        def run_step(input_data, is_training=True):
            """Run one step of the training process."""
            input_x, input_y, sequence_length = input_data

            bc = BertClient(ip="127.0.0.1")
    
            # useEmbed = 'bert_blstm'
            useEmbed = FLAGS.useEmbed
                
            if useEmbed == 'bert_clstm':
                # n * 768 embedding           
                listofzeros = [0] * 768
                storeToList = []
                for index,post in enumerate(input_x):
                    sentences = tokenize.sent_tokenize(post)
                    embeds = bc.encode(sentences)
                    storeToList.append(0)                
                    embeds = embeds.copy()
                    if len(embeds) > 5:
                        embeds = embeds[0:4,:]
                    appended = np.array([listofzeros] * (6 - len(embeds)))
                    appended = np.array([np.array(x) for x in appended])
                    apped = np.vstack((embeds, appended))
                    storeToList[index] = apped
                input_x = np.array(storeToList)
             
            if useEmbed == 'bert_blstm':
                storeToList = []
                for index,post in enumerate(input_x):
                    sentences = tokenize.sent_tokenize(post)
                    embeds = bc.encode(sentences)
                    embeds = embeds.copy()
                    storeToList.append(np.array(0))           
                    padded_embeddings = np.array(embeds.flatten())
                    if len(padded_embeddings) > 40001:
                        padded_embeddings = padded_embeddings[0:40000]
                    padded_embeddings = np.pad(padded_embeddings, (0, 40000 - len(padded_embeddings)), mode='constant')
                    storeToList[index] = np.array(padded_embeddings)
                    
                input_x = np.array([np.array(x) for x in storeToList])

            if useEmbed == 'glove':
                PAD_TOKEN = 0
                word2idx = { 'PAD': PAD_TOKEN } 
                weights = []
                with open("./data/glove.6B.50d.txt", 'r') as file:
                    for index, line in enumerate(file):
                        values = line.split()
                        word = values[0] 
                        word_weights = np.asarray(values[1:], dtype=np.float32) 
                        word2idx[word] = index + 1 
                        weights.append(word_weights)
                EMBEDDING_DIMENSION = len(weights[0])
                weights.insert(0, np.random.randn(EMBEDDING_DIMENSION))
                UNKNOWN_TOKEN=len(weights)
                word2idx['UNK'] = UNKNOWN_TOKEN
                weights.append(np.random.randn(EMBEDDING_DIMENSION))
                weights = np.asarray(weights, dtype=np.float32)
                finalWordEmbeds = []
                for index,post in enumerate(input_x):
                    words = tokenize.word_tokenize(post)
                    wordIndices = [ int(word2idx[k]) for k in words if k in word2idx ]
                    wordIndices = np.array([np.array(x) for x in wordIndices])
                    finalWordEmbeds.append(np.array(wordIndices))
                mappedDic = np.array(finalWordEmbeds)
                listofzeros = [0] * 50
                finalEmbedsAll = []
                for index,postEmbedindexArr in enumerate(mappedDic):
                    finalEmbeds = [ weights[embIndex] for embIndex in postEmbedindexArr]
                    finalEmbeds = np.array([np.array(x) for x in finalEmbeds])
                    if len(finalEmbeds) == 0: 
                        print('0 glove embed found')
                        finalEmbeds = [ weights['UNK']]
                        finalEmbeds = np.array([np.array(x) for x in finalEmbeds])
                    appended = np.array([listofzeros] * (1000 - len(finalEmbeds)))
                    appended = np.array([np.array(x) for x in appended])
                    apped = np.vstack((finalEmbeds, appended))
                    finalEmbedsAll.append(0)      
                    finalEmbedsAll[index] = apped                
                finalEmbedsAll = np.array(finalEmbedsAll)
                input_x = np.array(finalEmbedsAll)

            fetches = {'step': global_step,
                       'cost': classifier.cost,
                       'accuracy': classifier.accuracy,
                       'learning_rate': learning_rate}
            feed_dict = {classifier.input_x: input_x,
                         classifier.input_y: input_y}

            if FLAGS.clf != 'cnn' and FLAGS.clf != 'nn':
                fetches['final_state'] = classifier.final_state
                feed_dict[classifier.batch_size] = len(input_x)
                feed_dict[classifier.sequence_length] = sequence_length
            if FLAGS.clf == 'nn':
                feed_dict[classifier.batch_size] = len(input_x)
                feed_dict[classifier.sequence_length] = sequence_length

            if is_training:
                fetches['train_op'] = train_op
                fetches['summaries'] = train_summary_op
                feed_dict[classifier.keep_prob] = FLAGS.keep_prob
            else:
                fetches['summaries'] = valid_summary_op
                feed_dict[classifier.keep_prob] = 1.0

            vars = sess.run(fetches, feed_dict)
            step = vars['step']
            cost = vars['cost']
            accuracy = vars['accuracy']
            summaries = vars['summaries']

            # Write summaries to file
            if is_training:
                train_summary_writer.add_summary(summaries, step)
            else:
                valid_summary_writer.add_summary(summaries, step)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step: {}, loss: {:g}, accuracy: {:g}".format(time_str, step, cost, accuracy))

            return accuracy


        print('Start training ...')

        for train_input in train_data:
            run_step(train_input, is_training=True)
            current_step = tf.train.global_step(sess, global_step)

            if current_step % FLAGS.evaluate_every_steps == 0:
                print('\nValidation')
                run_step((x_valid, y_valid, valid_lengths), is_training=False)
                print('')

            if current_step % FLAGS.save_every_steps == 0:
                save_path = saver.save(sess, os.path.join(outdir, 'model/clf'), current_step)

        print('\nAll the files have been saved to {}\n'.format(outdir))
