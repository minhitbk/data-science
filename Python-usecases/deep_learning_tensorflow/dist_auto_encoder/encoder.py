'''
Created on Apr 14, 2016

@author: minhtran
'''
import numpy as np
from itertools import islice
import json
import tensorflow as tf

# Magic class  
class AutoEncoder(object):
    def __init__(self, config, embeddings=None):
        """ Initialization function.
        """
        # Set configuration        
        self._config = config
        
        # Build embeddings
        if embeddings is None: self._embeddings = self.build_embeddings()
        else: self._embeddings=embeddings

    def declare_placeholders(self):
        """ Generate placeholder variables to represent the input tensors. These 
        placeholders are used as inputs by the rest of the model building code.
        """
        # Get number of features
        num_feature = len(self._config.feature_desc)
        
        # Define input/output placeholder
        self._input_batch = tf.placeholder(tf.float32, [None, None, num_feature], 
                                           name='input_batch')
        self._output_batch = tf.placeholder(tf.float32, [None, None, num_feature],
                                            name='output_batch')
        # Set initial state placeholder
        self._initial_state = tf.placeholder(tf.float32, [None, 2*self._config.lstm_size],
                                             name='initial_state')
    
    def build_embeddings(self):
        """ Build an embedding object to embed categorical features.
        """
        # Initialize a dictionary for embeddings
        embeddings = dict()
        
        # Build the embeddings 
        for feature in range(len(self._config.feature_desc)):
            num_cat_value = self._config.feature_desc[feature]
            if num_cat_value == 1:
                pass
            elif num_cat_value <= self._config.vector_threshold:
                if not embeddings.has_key(num_cat_value):
                    embeddings[num_cat_value] = tf.Variable(np.identity(num_cat_value, 
                                                            dtype=np.float32), trainable=False, 
                                                            name='embeddings'+str(num_cat_value))
            else:
                if not embeddings.has_key(num_cat_value):
                    embeddings[num_cat_value] = tf.get_variable('embeddings'+str(num_cat_value), 
                                        shape=[num_cat_value, np.round(np.log2(num_cat_value))], 
                                        trainable=False)
        return embeddings
    
    def embed_feature_vector(self, batch):
        """ Embed a feature vector for an event.
        """
        for feature in range(len(self._config.feature_desc)):
            feature_val = batch[:, feature]            
            num_cat_value = self._config.feature_desc[feature]
            if feature == 0: 
                if num_cat_value == 1:
                    vector = tf.reshape(feature_val, [-1, 1])
                else:
                    vector = tf.nn.embedding_lookup(self._embeddings[num_cat_value],
                                                    tf.cast(feature_val, tf.int32))
            else:
                if num_cat_value == 1:
                    vector = tf.concat(1, [vector, tf.reshape(feature_val, [-1, 1])])
                else:
                    vector = tf.concat(1, [vector, tf.nn.embedding_lookup(
                                            self._embeddings[num_cat_value], 
                                            tf.cast(feature_val, tf.int32))])
        return vector
                    
    def inference(self, worker=0):
        """ Inference function for building the graph.
        """
        # Basic LSTM cell
        cell = tf.nn.rnn_cell.BasicLSTMCell(self._config.lstm_size, forget_bias=0.0)

        # Get initial state
        state = self._initial_state
            
        # Process for each event
        cell_outputs = []
        for event in range(self._config.num_event):
            # Create input vectors
            input_vec = self.embed_feature_vector(self._input_batch[ :, event, :])
                
            # Construct LSTMs             
            with tf.variable_scope('lstm', initializer=tf.random_normal_initializer(0, 
                                                            self._config.init_std)):
                if worker > 0 or event > 0: tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(input_vec, state)
                cell_outputs.append(cell_output)
        return cell_outputs, state
    
    def loss(self, cell_outputs):
        """ Loss function for building the graph.
        """
        # Size of the output softmax vector
        soft_size = int(np.sum([s if s <= self._config.vector_threshold 
                                else np.round(np.log2(s)) for s in self._config.feature_desc]))
        
        # 1 more hidden layer
        hid_matrix = tf.get_variable(initializer=tf.truncated_normal([self._config.lstm_size, soft_size], 
                                                     stddev=self._config.init_std, 
                                                     dtype=tf.float32), name='hid_matrix')
        hid_bias = tf.get_variable(initializer=tf.truncated_normal([soft_size], stddev=self._config.init_std,
                                                   dtype=tf.float32), name='hid_bias')
        
        # Process for each event
        outputs = []
        for event in range(self._config.num_event):     
            logits = tf.matmul(cell_outputs[event], hid_matrix) + hid_bias
            
            # Proceed for each feature
            index, feature_list = 0, []
            for feature in range(len(self._config.feature_desc)):
                num_cat_value = self._config.feature_desc[feature]
                if num_cat_value > self._config.vector_threshold:
                    num_cat_value = int(np.round(np.log2(num_cat_value)))
                feature_vec = logits[:, index:(index+num_cat_value)]
                predict_target = tf.nn.softmax(feature_vec)         
                feature_list.append(predict_target)
                index = index + num_cat_value

            # Store outputs
            outputs.append(tf.concat(1, feature_list))      

            # Get output from data
            output_vec = self.embed_feature_vector(self._output_batch[ :, event, :])
            if event == 0: output_batch = output_vec
            else: output_batch = tf.concat(1, [output_batch, output_vec])
       
        # Final output from the model
        output = tf.concat(1, outputs)

        # Define an objective function
        obj_func = tf.reduce_mean(tf.square(tf.sub(output, output_batch)))        
        return obj_func

    def train(self, obj_func):
        """ Set up the training Ops.
        """        
        tvars = tf.trainable_variables()
        grads = tf.gradients(obj_func, tvars) 
        norm_grads, _ = tf.clip_by_global_norm(grads, self._config.max_grad_norm)        

        # Define a training step
        optimizer = tf.train.AdamOptimizer(self._config.learning_rate)
        self._train_step = optimizer.apply_gradients(zip(norm_grads, tvars))
        self._grads = grads
        self._tvars = tvars

    def dist_train(self, obj_func, optimizer):
        """ Set up the distributed training Ops.
        """
        # Define a training step
        self._grads = optimizer.compute_gradients(obj_func)
        
    def build_model(self, worker=0):
        """ Build the total graph for training.
        """
        self.declare_placeholders()
        cell_outputs, _ = self.inference(worker)
        self._mean_square_error = self.loss(cell_outputs)
        self.train(self._mean_square_error)

    def build_dist_model(self, optimizer, worker=0):
        """ Build the total graph for training.
        """
        self.declare_placeholders()
        cell_outputs, _ = self.inference(worker)
        self._mean_square_error = self.loss(cell_outputs)
        self.dist_train(self._mean_square_error, optimizer)
         
    def build_encoder(self):
        """ Build the total graph for encoding.
        """
        self.declare_placeholders()
        _, state = self.inference()
        [vector_rep, _] = tf.split(1, 2, state)
        self._vector_rep = vector_rep
                
    def run_epoch(self, sess, data_file):
        """ Function for running 1 epoch.
        """        
        # Store mse so far
        mseList = []
        
        # Read data chunk by chunk
        while True:
            input_batch, output_batch, state_batch, _ = self.get_batch(data_file)
            if input_batch is None: break
            else:
                # Batch is not empty
                [_, mse] = sess.run([self.get_train_step(), self.get_mse()], 
                                    feed_dict={self.get_input_batch(): input_batch,
                                               self.get_output_batch(): output_batch,
                                               self.get_initial_state(): state_batch})
                mseList.append(mse)        
        return np.mean(mseList)

    def gen_rep(self, sess, data_file, rep_file):
        """ Generate representation for all users.
        """
        # Read data chunk by chunk
        while True:
            input_batch, _, state_batch, user_ids = self.get_batch(data_file)
            if input_batch is None: break
            else:
                # Batch is not empty
                [res] = sess.run([self._vector_rep], 
                                 feed_dict={self.get_input_batch(): input_batch, 
                                            self.get_initial_state(): state_batch})

                # Write results to file
                for user in range(len(user_ids)):    
                    rep_file.write(json.dumps({user_ids[user]: str(res[user]).replace('\n', '')}))
                    rep_file.write('\n')

    def get_vector_rep(self):
        """ Return the representation of a batch of users.
        """
        return self._vector_rep

    def get_train_step(self):
        """ Get the train_step ops.
        """
        return self._train_step
    
    def get_input_batch(self):
        """ Get the input_batch ops.
        """
        return self._input_batch
    
    def get_output_batch(self):
        """ Get the output_batch ops.
        """
        return self._output_batch
    
    def get_initial_state(self):
        """ Get the intial_state ops.
        """
        return self._initial_state
    
    def get_mse(self):
        """ Get the mse ops.
        """
        return self._mean_square_error
    
    def get_grads(self):
        """ Get the grads ops.
        """
        return self._grads

    def get_tvars(self):
        """ Get the tvars ops.
        """
        return self._tvars
    
    def get_embeddings(self):
        """ Get embeddings dict
        """
        return self._embeddings

    def get_batch(self, data_file):
        """ Read data as batch per time.
        """
        behave_batch, user_ids = [], []
        lines_gen = islice(data_file, self._config.batch_size)
        for line in lines_gen:
            user_behave = json.loads(line).values()[0]
                        
            # Remove those users who only have 1 event
            if (len(user_behave) <= self._config.num_min_event):
                continue
            
            if (len(user_behave) < self._config.num_event + 1):
                zero_pad = [[0 for _ in range(len(user_behave[0]))] 
                            for _ in range(self._config.num_event-len(user_behave)+1)]
                user_behave = user_behave + zero_pad
            elif (len(user_behave) > self._config.num_event + 1):
                user_behave = user_behave[(len(user_behave)-self._config.num_event-1) 
                                          : len(user_behave)]
            
            # Remove later
            update_user_behave = []
            for each_behave in user_behave:
                each_behave[5] = float(each_behave[5])/99
                update_user_behave.append(each_behave)
            user_behave = update_user_behave
                               
            behave_batch.append(np.array(user_behave)[:, list([1,2,3,7,8,9,10,11,12,13])])
            user_ids.append(json.loads(line).keys()[0])
        
        # Batch is empty then exit
        if not behave_batch:
            return None, None, None, None
    
        # Batch is not empty
        batch_to_arr = np.asarray(behave_batch)
        input_batch = batch_to_arr[:, :-1]
        output_batch = batch_to_arr[:, 1:]
        state_batch = np.zeros([len(behave_batch), 2*self._config.lstm_size])
        return input_batch, output_batch, state_batch, user_ids

            