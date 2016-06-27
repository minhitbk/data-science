'''
Created on Apr 1, 2016

@author: minhtran
'''
import csv
import itertools
import nltk
import numpy as np
import sys
import pickle
import tensorflow as tf

reload(sys)  
sys.setdefaultencoding('utf8')  # @UndefinedVariable

vocabulary_size = 2000
unknown_token = 'UNKNOWN_TOKEN'
sentence_start_token = 'SENTENCE_START'
sentence_mid_token = 'SENTENCE_MID'
sentence_end_token = 'SENTENCE_END'

lstm_size = 128
batch_size = 64 
num_epoch = 40

def utf_8_encoder(unicode_csv_data):
    for line in unicode_csv_data:
        yield line.encode('utf-8')
        
def weighted_pick(weights):
    t = np.cumsum(weights)
    s = np.sum(weights)
    return int(np.searchsorted(t, np.random.rand(1) * s))

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        
with open('data/input.txt', 'rb') as f:
    reader = csv.reader(utf_8_encoder(f), skipinitialspace=True)
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) 
                                                                            for x in reader])
    sentences = ['%s' % (x) for x in sentences]
    # Append SENTENCE_START
    sentences[0::2] = ['%s %s' % (sentence_start_token, x) for x in sentences[0::2]]
    # Append SENTENCE_END
    sentences[1::2] = ['%s %s' % (x, sentence_end_token) for x in sentences[1::2]]
    # Insert SENTENCE_MID
    sentences = [six + ' ' + sentence_mid_token + ' ' + eight for six, eight in 
                                                       zip(sentences[0::2], sentences[1::2])]
print 'Parsed %d sentences.' % (len(sentences))

# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print 'Found %d unique words tokens.' % len(word_freq.items())

# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

# Save objects for later use
save_object(index_to_word, 'save/index_to_word.pkl')
save_object(word_to_index, 'save/word_to_index.pkl')

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

# Create the training data
x_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

lstm = tf.nn.rnn_cell.LSTMCell(lstm_size, vocabulary_size, 
                                        initializer=tf.truncated_normal_initializer(0, 0.05))
init_state = tf.placeholder(tf.float32, [None, lstm.state_size], name='init_state')
embeddings = tf.Variable(np.identity(vocabulary_size, dtype=np.float32), name='embeddings', 
                                                                             trainable=False)
softmax_w = tf.Variable(tf.truncated_normal([lstm_size, vocabulary_size], stddev=0.05, 
                                                         dtype=tf.float32), name='softmax_w')
softmax_b = tf.Variable(tf.truncated_normal([vocabulary_size], stddev=0.05, dtype=tf.float32),
                                                                            name='softmax_b')
sen_len = len(y_train[0])
input_batch = tf.placeholder(tf.int32, [None, sen_len], name='input_batch')
target_batch = tf.placeholder(tf.int32, [None, sen_len], name='target_batch')

inputs = tf.nn.embedding_lookup(embeddings, input_batch)
outputs = []
with tf.variable_scope('lstm', initializer=tf.random_normal_initializer(0, 0.05)):
    state = init_state
    for j in range(sen_len):
        if j > 0: 
            tf.get_variable_scope().reuse_variables()
        cell_output, state = lstm(inputs[:, j, :], state)
        logits = tf.matmul(cell_output, softmax_w) + softmax_b
        predict_target = tf.nn.softmax(logits)
        outputs.append(predict_target)

output = tf.reshape(tf.concat(1, outputs), shape = [batch_size, sen_len, -1])
target_lookup = tf.nn.embedding_lookup(embeddings, target_batch)
target = tf.reshape(tf.reshape(target_lookup, shape=[batch_size, -1]), shape = [batch_size, 
                                                                                sen_len, -1])
# Define an objective function
cross_entropy = -tf.reduce_sum(target * tf.log(output))
# Define a training step
#train_step = tf.train.GradientDescentOptimizer(0.007).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

# Test
correct_prediction = tf.equal(tf.argmax(output, 2), tf.argmax(target, 2))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# For modeling language with 1 best sentence 
sen_start = tf.placeholder(tf.int32, [None, 1], name='sen_start')
word_ = tf.nn.embedding_lookup(embeddings, sen_start)
results = []
with tf.variable_scope('lstm'):
    tf.get_variable_scope().reuse_variables()
    state = init_state
    for i in range(16):
        pre_words, state = lstm(word_[:, 0, :], state)
        logits = tf.matmul(pre_words, softmax_w) + softmax_b
        predict_target = tf.nn.softmax(logits)
        word = tf.argmax(predict_target, 1)
        results.append(word)
        word_ = tf.nn.embedding_lookup(embeddings, tf.reshape(word, shape=[1, 1]))

num_batch = len(y_train) / batch_size
print 'Number of batches: %d.' % num_batch

# Setup a session
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

# Train model
for epoch in range(num_epoch):
    accList = []
    for i in range(num_batch):
        _, acc, corr_pre = sess.run([train_step, accuracy, correct_prediction], 
                              feed_dict={input_batch: x_train[i*batch_size:(i+1)*batch_size], 
                                        target_batch: y_train[i*batch_size:(i+1)*batch_size],
                                        init_state: np.zeros([batch_size, lstm.state_size])})
        accList.append(acc)

    if epoch % 5 == 0:
        sen_index = np.random.choice(range(len(x_train)), size = batch_size, replace = False)
        acc = accuracy.eval(feed_dict={input_batch: x_train[sen_index], 
                                       target_batch: y_train[sen_index],
                                       init_state: np.zeros([batch_size, lstm.state_size])}) 
        print ('Epoch %d has accuracy: %g and %g' % (epoch, np.mean(accList), acc))

saver = tf.train.Saver(tf.all_variables())
saver.save(sess, 'save/model.ckpt')    

#Generate poems
#Generate 1 best sentence
words_ = sess.run(results, feed_dict={sen_start: np.matrix(1*[1*[2]]),
                                                 init_state: np.zeros([1, lstm.state_size])})
sys.stdout.write('\n')
sys.stdout.write(index_to_word[2].encode('utf8'))
sys.stdout.write(' ')
for w in words_:        
    sys.stdout.write(index_to_word[w].encode('utf8'))
    sys.stdout.write(' ')


