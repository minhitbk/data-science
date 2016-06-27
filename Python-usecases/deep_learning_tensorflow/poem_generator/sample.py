'''
Created on Apr 6, 2016

@author: minhtran
'''
import numpy as np
import sys
import pickle
import tensorflow as tf

reload(sys)  
sys.setdefaultencoding('utf8')  # @UndefinedVariable

def load_object(filename):
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
    return obj

def weighted_pick(weights):
    t = np.cumsum(weights)
    s = np.sum(weights)
    return int(np.searchsorted(t, np.random.rand(1) * s))

vocabulary_size = 2000
lstm_size = 128

index_to_word = load_object('save/index_to_word.pkl')
word_to_index = load_object('save/word_to_index.pkl')

lstm = tf.nn.rnn_cell.LSTMCell(lstm_size, vocabulary_size,
                                        initializer=tf.truncated_normal_initializer(0, 0.05))
embeddings = tf.Variable(np.identity(vocabulary_size, dtype=np.float32), name='embeddings', 
                                                                             trainable=False)
softmax_w = tf.Variable(tf.truncated_normal([lstm_size, vocabulary_size], stddev=0.05, 
                                                         dtype=tf.float32), name='softmax_w')
softmax_b = tf.Variable(tf.truncated_normal([vocabulary_size], stddev=0.05, dtype=tf.float32),
                                                                            name='softmax_b')

word__ = tf.nn.embedding_lookup(embeddings, tf.reshape([2], shape=[1, 1]))
with tf.variable_scope('lstm'):
    _, state = lstm(word__[:, 0, :], tf.zeros((1, lstm.state_size)))
    
# Setup a session
sess = tf.Session()

# Load Tensorflow objects
saver = tf.train.Saver(tf.all_variables())
ckpt = tf.train.get_checkpoint_state('save')
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)

#Generate an arbitrary set of sentences
for i in range(1000):
    word_ = tf.nn.embedding_lookup(embeddings, tf.reshape([2], shape=[1, 1]))
    results = []
    with tf.variable_scope('lstm'):
        tf.get_variable_scope().reuse_variables()
        new_state = tf.zeros((1, lstm.state_size))
        for i in range(16):
            pre_words, new_state = lstm(word_[:, 0, :], new_state)
            logits = tf.matmul(pre_words, softmax_w) + softmax_b
            target_prob = tf.nn.softmax(logits)
            target_prob_vals = sess.run(target_prob)
            word = weighted_pick(target_prob_vals)
            results.append(word)
            word_ = tf.nn.embedding_lookup(embeddings, tf.reshape(word, shape=[1, 1])) 
    
    sys.stdout.write('\n')
    sys.stdout.write(index_to_word[2].encode('utf8'))
    sys.stdout.write(' ')
    for w in results:        
        sys.stdout.write(index_to_word[w].encode('utf8'))
        sys.stdout.write(' ')
