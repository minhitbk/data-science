import nltk
import numpy as np
import tensorflow as tf

vocabulary_size = 2200
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_mid_token = "SENTENCE_MID"
sentence_end_token = "SENTENCE_END"

lstm_size = 128
batch_size = 64
num_epoch = 2

with codecs.open('data/tho_luc_bat.txt', 'rb', 'utf-8') as f:
    sentences = [nltk.sent_tokenize(x)[0] for x in f]

concat_sentences = []
for i in range(len(sentences) - 1):
    concat_sentences.append('{} {} {} {} {}'.format(sentence_start_token,
                                                    sentences[i], sentence_mid_token,
                                                    sentences[i+1], sentence_end_token) )
sentences = concat_sentences

print("Parsed %d sentences." % (len(sentences)))

# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print("Found %d unique words tokens." % len(word_freq.items()))

# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token
                                                          for w in sent]

# Create the training data
x_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
initial_state = tf.placeholder(tf.float32, [None, lstm.state_size], name='initial_state')
sen_len = len(y_train[0])

embeddings = tf.Variable(np.identity(vocabulary_size, dtype=np.float32))
input_batch = tf.placeholder(tf.int32, [None, sen_len], name='input_batch')
target_batch = tf.placeholder(tf.int32, [None, sen_len], name='target_batch')

softmax_w = tf.Variable(tf.truncated_normal([lstm_size, vocabulary_size],
                                           stddev=0.1, dtype=tf.float32))
softmax_b = tf.Variable(tf.truncated_normal([vocabulary_size], stddev=0.1,
                                                        dtype=tf.float32))

inputs = tf.nn.embedding_lookup(embeddings, input_batch)
outputs = []

with tf.variable_scope("lstm"):
    state = initial_state
    for j in range(sen_len):
        if j > 0:
            tf.get_variable_scope().reuse_variables()
        inp = tf.squeeze(tf.slice(inputs, [0, 1, 0], [-1, 1, -1]), [1])
        cell_output, state = lstm(inp, state)
        logits = tf.matmul(cell_output, softmax_w) + softmax_b
        predict_target = tf.nn.softmax(logits)
        outputs.append(tf.expand_dims(predict_target, 1))

output = tf.concat(1, outputs)
target_lookup = tf.nn.embedding_lookup(embeddings, target_batch)
print('Input shape: {}, target lookup shape: {}'.format(output.get_shape(), target_lookup.get_shape()))

# Train and evaluate the model
# Define an objective function
cross_entropy = -tf.reduce_sum(target_lookup * tf.log(output))

# Define a training step
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Test
correct_prediction = tf.equal(tf.argmax(output, 2), tf.argmax(target_lookup, 2))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

num_batch = len(y_train) / batch_size
print("Number of batches: %d." % num_batch)

# Setup a session
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

# Train model
initial_values = np.zeros([batch_size, lstm.state_size])

for epoch in range(num_epoch):
    batch_acc = 0
    for i in range(num_batch):
        _, acc = sess.run([train_step, accuracy],
                          feed_dict={input_batch: x_train[i*batch_size:(i+1)*batch_size],
                                     target_batch: y_train[i*batch_size:(i+1)*batch_size],
                                     initial_state: initial_values})
        batch_acc += acc

    sen_index = np.random.choice(range(len(x_train)), size=batch_size, replace=False)
    acc = accuracy.eval(feed_dict={input_batch: x_train[sen_index],
                                   target_batch: y_train[sen_index],
                                   initial_state: initial_values})
    print('Epoch {}: Train acc = {}, Random subset acc = {}'.format(epoch, batch_acc / num_batch, acc))

# Generate poems
gen_outputs = []


def weighted_pick(weights):
    t = np.cumsum(weights)
    s = np.sum(weights)
    return np.array([[int(np.searchsorted(t, np.random.rand(1) * s))]])

for s in index_to_word[:20]:
    print(s)

with tf.variable_scope("lstm"):
    tf.get_variable_scope().reuse_variables()
    new_state = tf.zeros((1, lstm.state_size))
    seed_word = 'hoa'
    new_word = np.array([[word_to_index[seed_word]]])

    for j in range(sen_len):
        new_inp = tf.squeeze(tf.nn.embedding_lookup(embeddings, new_word), [1])
        new_cell_output, new_state = lstm(new_inp, new_state)
        new_probs = tf.nn.softmax(tf.matmul(new_cell_output, softmax_w) + softmax_b)

        new_probs_vals = sess.run(new_probs)
        new_word = weighted_pick(new_probs_vals)

        gen_outputs.append(index_to_word[new_word[0, 0]])

print(gen_outputs)
