'''
Created on Mar 18, 2016

@author: minhtran
'''

from __future__ import absolute_import
from __future__ import print_function

import collections
import zipfile

import numpy as np
from six.moves import xrange
import tensorflow as tf

# Read the data into a string.
def read_data(filename):
    f = zipfile.ZipFile(filename)
    for name in f.namelist():
        return f.read(name).split()
    f.close()

vocabulary_size = 5000
def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

data_index = -1
# Function to generate a training batch for the cbow model.
def generate_batch(batch_size, data):
    global data_index
    batch = np.ndarray(shape=(batch_size, 2), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    for i in range(batch_size):
        data_index = data_index + 1
        batch[i] = [data[data_index], data[data_index + 2]]
        labels[i] = data[data_index + 1]
    
    return batch, labels

def main():
    filename = "/tmp/cifar10_data/cifar-10-batches-bin/text8.zip"
    words = read_data(filename)
    print('Data size', len(words))
  
    data, _, _, _ = build_dataset(words)
    del words
    batch_size = 1000
    
    graph = tf.Graph()
    with graph.as_default():

        # Input data
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size, 2])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

        with tf.device('/cpu:0'):
            # Look up embeddings for inputs.
            embeddings = tf.Variable(np.identity(vocabulary_size, dtype = np.float32))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)
            embed1 = tf.reshape(embed, shape=(batch_size, 2*vocabulary_size))
            W_1 = tf.Variable(tf.truncated_normal([2*vocabulary_size, 128], stddev=0.1, 
                                                                            dtype=tf.float32))
            b_1 = tf.Variable(tf.zeros([128], dtype=tf.float32))
            hidden = tf.nn.softmax(tf.matmul(embed1, W_1) + b_1)
            W_2 = tf.Variable(tf.truncated_normal([128, vocabulary_size], stddev=0.1, 
                                                                            dtype=tf.float32))
            b_2 = tf.Variable(tf.zeros([vocabulary_size], dtype=tf.float32))
            y = tf.nn.softmax(tf.matmul(hidden, W_2) + b_2)
            y__ = tf.nn.embedding_lookup(embeddings, train_labels)
            y_ = tf.reshape(y__, shape=(batch_size, vocabulary_size))
    
        cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))    
  
    num_steps = 200
    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        tf.initialize_all_variables().run()
        print("Initialized")

        for step in xrange(num_steps):
            batch_inputs, batch_labels = generate_batch(batch_size, data)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        session.run(train_step, feed_dict=feed_dict)      
    
        if step % 20 == 0:
            train_accuracy = accuracy.eval(feed_dict={train_inputs: batch_inputs, 
                                                                  train_labels: batch_labels})
            print("step %d, training accuracy %g" % (step, train_accuracy))
        
        # Results
        batch_tests, batch_tests_labels = generate_batch(batch_size, data)
        print(session.run(accuracy, feed_dict={train_inputs: batch_tests, 
                                                           train_labels: batch_tests_labels}))

if __name__ == '__main__':
    main()
