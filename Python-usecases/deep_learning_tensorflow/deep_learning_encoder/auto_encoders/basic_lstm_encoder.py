'''
Created on Jun 13, 2016

@author: minhtran
'''
from ..cores.core_encoder import CoreEncoder
import tensorflow as tf
import numpy as np
import json

class BasicLSTMEncoder(CoreEncoder):
    '''
    This class implements a simple auto_encoder with 1 LSTM and with using the feed_dict
    mechanism to feed data into Tensorflow. 
    '''
    
    def __init__(self, data_reader, trainable_config, untrainable_config, embeddings=None, 
                 worker=0, optimizer=None):
        '''
        Constructor will build the total graph automatically.
        '''
        #super(BasicLSTMEncoder, self).__init__()
        CoreEncoder.__init__(self, data_reader, trainable_config, untrainable_config)
        
        # Build embeddings
        if embeddings is None: self._embeddings = self.build_embeddings()
        else: self._embeddings=embeddings
                
        # Build the total graph
        self.declare_input()
        cell_outputs, state = self.inference(worker)
        self._mean_square_error = self.loss(cell_outputs)
        [vector_rep, _] = tf.split(1, 2, state)
        self._vector_rep = vector_rep        
        self.train(self._mean_square_error, optimizer)

    def build_embeddings(self):
        '''
        Build an embedding object to embed categorical features.
        '''
        # Initialize a dictionary for embeddings
        embeddings = dict()
        
        # Build the embeddings 
        for feature in range(len(self._untrainable_config._feature_desc)):
            num_cat_value = self._untrainable_config._feature_desc[feature]
            if num_cat_value == 1:
                pass
            elif num_cat_value <= self._untrainable_config._vector_threshold:
                if not embeddings.has_key(num_cat_value):
                    embeddings[num_cat_value] = tf.Variable(np.identity(num_cat_value, 
                                                            dtype=np.float32), trainable=False, 
                                                            name="embeddings"+str(num_cat_value))
            else:
                if not embeddings.has_key(num_cat_value):
                    embeddings[num_cat_value] = tf.get_variable("embeddings"+str(num_cat_value), 
                                        shape=[num_cat_value, np.round(np.log2(num_cat_value))], 
                                        trainable=False)
        return embeddings
   
    def get_num_feature(self):
        '''
        Get number of features.
        '''                
        return len(self._untrainable_config._feature_desc)
            
    def declare_input(self):
        '''
        Generate placeholder variables to represent the input tensors. These placeholders 
        are used as inputs by the rest of the model building code.
        '''        
        if self._data_reader._type == "feeddict":
            # Define input/output placeholder
            self._input_batch = tf.placeholder(tf.float32, [None, None, self.get_num_feature()], 
                                               name="input_batch")
            self._output_batch = tf.placeholder(tf.float32, [None, None, self.get_num_feature()],
                                                name="output_batch")
    
            # Set initial state placeholder
            self._initial_state = tf.placeholder(tf.float32, [None, 
                                self._untrainable_config._lstm_size * 2], name="initial_state")
        elif self._data_reader._type == "pipeline":
            self._input_batch, self._output_batch, self._initial_state, _ = self._data_reader.get_batch()
        
    def embed_feature_vector(self, batch):
        '''
        Embed a feature vector for an event.
        '''
        for feature in range(len(self._untrainable_config._feature_desc)):
            feature_val = batch[:, feature]            
            num_cat_value = self._untrainable_config._feature_desc[feature]
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
        '''
        Inference function for building the graph.
        '''
        # Basic LSTM cell
        cell = tf.nn.rnn_cell.BasicLSTMCell(self._untrainable_config._lstm_size, forget_bias=0.0)

        # Get initial state
        state = self._initial_state
            
        # Process for each event
        cell_outputs = []
        for event in range(self._trainable_config._num_max_event):
            # Create input vectors
            input_vec = self.embed_feature_vector(self._input_batch[ :, event, :])
                
            # Construct LSTMs             
            with tf.variable_scope("lstm", initializer=tf.random_normal_initializer(0, 
                                                        self._untrainable_config._init_std)):
                if worker > 0 or event > 0: tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(input_vec, state)
                cell_outputs.append(cell_output)
        return cell_outputs, state    

    def loss(self, cell_outputs):
        '''
        Loss function for building the graph.
        '''
        # Size of the output softmax vector
        soft_size = int(np.sum([s if s <= self._untrainable_config._vector_threshold 
                                else np.round(np.log2(s)) for s in 
                                self._untrainable_config._feature_desc]))
        
        # 1 more hidden layer
        hid_matrix = tf.get_variable(initializer=tf.truncated_normal(
                                        [self._untrainable_config._lstm_size, soft_size], 
                                        stddev=self._untrainable_config._init_std, 
                                        dtype=tf.float32), name="hid_matrix")
        hid_bias = tf.get_variable(initializer=tf.truncated_normal([soft_size], 
                                                stddev=self._untrainable_config._init_std,
                                                dtype=tf.float32), name="hid_bias")
        
        # Process for each event
        outputs = []
        for event in range(self._trainable_config._num_max_event):     
            logits = tf.matmul(cell_outputs[event], hid_matrix) + hid_bias
            
            # Proceed for each feature
            index, feature_list = 0, []
            for feature in range(len(self._untrainable_config._feature_desc)):
                num_cat_value = self._untrainable_config._feature_desc[feature]
                if num_cat_value > self._untrainable_config._vector_threshold:
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
    
    def train(self, obj_func, optimizer=None):
        '''
        Set up the training Ops.
        '''        
        if optimizer is None:
            tvars = tf.trainable_variables()
            grads = tf.gradients(obj_func, tvars) 
            norm_grads, _ = tf.clip_by_global_norm(grads, self._trainable_config._max_grad_norm)        

            # Define a training step
            optimizer = tf.train.AdamOptimizer(self._trainable_config._learning_rate)
            self._train_step = optimizer.apply_gradients(zip(norm_grads, tvars))
        else:
            # Define a distributed gradients
            self._grads = optimizer.compute_gradients(obj_func)
        
    def run_epoch(self, sess, data_file):
        '''
        Function for running 1 epoch.
        '''
        # Store mse so far
        mseList = []
        
        # Read data chunk by chunk
        while True:
            input_batch, output_batch, state_batch, _ = self._data_reader.get_batch(data_file)
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
        '''
        Generate vector representations for all users.
        '''
        # Read data chunk by chunk
        while True:
            input_batch, _, state_batch, user_ids = self._data_reader.get_batch(data_file)
            if input_batch is None: break
            else:
                # Batch is not empty
                [res] = sess.run([self._vector_rep], 
                                 feed_dict={self.get_input_batch(): input_batch, 
                                            self.get_initial_state(): state_batch})

                # Write results to file
                for user in range(len(user_ids)):    
                    rep_file.write(json.dumps({user_ids[user]: str(res[user]).replace("\n", "")}))
                    rep_file.write("\n")
        
        