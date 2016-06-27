'''
Created on Jun 23, 2016

@author: minhtran
'''
import os

import tensorflow as tf
import pandas as pd
import numpy as np

from configuration import Config


# Set up configurations
config = Config()

def declare_input():
    '''
    This function is used to declare input batches.
    '''
    input_batch = tf.placeholder(tf.float32, [None, None])
    output_batch = tf.placeholder(tf.float32, [None, None])
    mask_batch = tf.placeholder(tf.float32, [None, None])
    return input_batch, output_batch, mask_batch

def inference(input_batch):
    '''
    This function is used to build the tensorflow graph.
    '''
    # First layer
    matrix_1 = tf.get_variable(initializer=tf.truncated_normal(
                                [config.num_item, config.num_hid_neuron], 
                                stddev=config.init_std, 
                                dtype=tf.float32), name="matrix_1")
    
    bias_1 = tf.get_variable(initializer=tf.truncated_normal(
                                [config.num_hid_neuron], 
                                stddev=config.init_std,
                                dtype=tf.float32), name="bias_1")
    
    # Second layer
    matrix_2 = tf.get_variable(initializer=tf.truncated_normal(
                                [config.num_hid_neuron, config.num_hid_neuron], 
                                stddev=config.init_std, 
                                dtype=tf.float32), name="matrix_2")
    
    bias_2 = tf.get_variable(initializer=tf.truncated_normal(
                                [config.num_hid_neuron], 
                                stddev=config.init_std,
                                dtype=tf.float32), name="bias_2")    

    # Third layer
    matrix_3 = tf.get_variable(initializer=tf.truncated_normal(
                                [config.num_hid_neuron, config.num_hid_neuron], 
                                stddev=config.init_std, 
                                dtype=tf.float32), name="matrix_3")
    
    bias_3 = tf.get_variable(initializer=tf.truncated_normal(
                                [config.num_hid_neuron], 
                                stddev=config.init_std,
                                dtype=tf.float32), name="bias_3")
    
    # Fourth layer
    matrix_4 = tf.get_variable(initializer=tf.truncated_normal(
                                [config.num_hid_neuron, config.num_hid_neuron], 
                                stddev=config.init_std, 
                                dtype=tf.float32), name="matrix_4")
     
    bias_4 = tf.get_variable(initializer=tf.truncated_normal(
                                [config.num_hid_neuron], 
                                stddev=config.init_std,
                                dtype=tf.float32), name="bias_4")
     
    # Fifth layer
    matrix_5 = tf.get_variable(initializer=tf.truncated_normal(
                                [config.num_hid_neuron, config.num_item], 
                                stddev=config.init_std, 
                                dtype=tf.float32), name="matrix_5")
     
    bias_5 = tf.get_variable(initializer=tf.truncated_normal(
                                [config.num_item], 
                                stddev=config.init_std,
                                dtype=tf.float32), name="bias_5")
    
    # Construct graph
    first_layer = tf.matmul(input_batch, matrix_1) + bias_1
    second_layer = tf.matmul(first_layer, matrix_2) + bias_2
    third_layer = tf.matmul(second_layer, matrix_3) + bias_3
    fourth_layer = tf.matmul(third_layer, matrix_4) + bias_4
    fifth_layer = tf.matmul(fourth_layer, matrix_5) + bias_5
    regularizers = tf.nn.l2_loss(matrix_1) + tf.nn.l2_loss(bias_1) + \
                    tf.nn.l2_loss(matrix_2) + tf.nn.l2_loss(bias_2) + \
                    tf.nn.l2_loss(matrix_3) + tf.nn.l2_loss(bias_3) + \
                    tf.nn.l2_loss(matrix_4) + tf.nn.l2_loss(bias_4) + \
                    tf.nn.l2_loss(matrix_5) + tf.nn.l2_loss(bias_5)
    return fifth_layer, regularizers

def loss(output_model, output_batch, mask_batch, regularizers):
    '''
    This function is used to build the loss.
    '''
    obj_func = tf.reduce_mean(tf.abs(tf.mul(tf.sub(output_model, 
                                    output_batch), mask_batch))) #+ regularizers
#    obj_func = tf.reduce_mean(tf.abs(tf.sub(output_model, 
#                                                output_batch))) + regularizers
    return obj_func

def train(obj_func):
    '''
    This function is used to define the train step.
    '''
    tvars = tf.trainable_variables()
    grads = tf.gradients(obj_func, tvars) 
    norm_grads, _ = tf.clip_by_global_norm(grads, config.max_grad_norm)
    
    # Define a training step
    optimizer = tf.train.AdamOptimizer(config.learning_rate)
    train_step = optimizer.apply_gradients(zip(norm_grads, tvars))
    return train_step

def run_epoch(sess, train_set, mask_set, train_step, obj_func, input_batch, 
              output_batch, mask_batch):
    ''' 
    Function for running 1 epoch.
    '''        
    # Store mse so far
    mseList = []
    
    num_batch = train_set.shape[0]/config.batch_size
    for batch in range(num_batch):
        start_batch = config.batch_size*batch
        end_batch = config.batch_size*(batch + 1)
        _, mse = sess.run([train_step, obj_func],
                  feed_dict={input_batch: train_set[start_batch:end_batch],
                             output_batch: train_set[start_batch:end_batch],
                             mask_batch: mask_set[start_batch:end_batch]})
        mseList.append(mse) 
    return np.mean(mseList)

def train_model(train_set, mask_set):
    '''
    This function is used to train the model.
    '''
    # Build the graph and run session
    with tf.Graph().as_default(), tf.Session() as sess:
        # Build the graph
        input_batch, output_batch, mask_batch = declare_input()
        output_model, regularizers = inference(input_batch)
        obj_func = loss(output_model, output_batch, mask_batch, regularizers)
        train_step = train(obj_func)
        
        # Start run session
        sess.run(tf.initialize_all_variables())
        for epoch in range(config.num_epoch):
            mse = run_epoch(sess, train_set, mask_set, train_step, obj_func, 
                            input_batch, output_batch, mask_batch)
            if epoch % 20 == 0:
                print ("Epoch %d has mean square error: %g" % (epoch, mse))
        
        # For saving checkpoint files and model
        saver = tf.train.Saver(tf.all_variables())
        saver.save(sess, config.save_path)
        print "Training done."

def scoring(train_set, test_set, mask_set):
    '''
    This function is used to score ratings.
    '''
    # Rebuild the graph and run session
    with tf.Graph().as_default(), tf.Session() as sess:
        # Rebuild the subgraph
        input_batch, _, _ = declare_input()
        output_model, _ = inference(input_batch)
        
        # Start run session
        sess.run(tf.initialize_all_variables())
        
        # Load model
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(config.save_path))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        
        # Scoring
        [output] = sess.run([output_model], feed_dict={input_batch: train_set})

#        print np.sum(np.abs(output - test_set)*(1 - mask_set))/np.sum(1 
#                                                                - mask_set)
        return output
        
def build_matrix():
    '''
    This function is used to split randomly the data set into train and test
    sets, by the way generates the masks.
    '''
    df = np.asarray(pd.read_csv(config.data_path))
    #df = np.asarray([[1,2,3,4,5], [2,3,4,5,6], [3,4,5,6,7], [4,5,6,7,8]])
    mask = np.random.rand(config.num_user, config.num_item)

    train = df.copy()
    test = df.copy()
    train[mask <= 0.2] = 0
    test[mask > 0.2] = 0
    mask[mask <= 0.2] = 0
    mask[mask > 0.2] = 1    
    return train, test, mask
           
def main():
    # Prepare the data for training
    train_set, test_set, mask_set = build_matrix()

    # Start training
    print "Start training..."
    train_model(train_set, mask_set)
    
    # Start scoring
    print "Start scoring..."
    est_set = scoring(train_set, test_set, mask_set)
    
    print est_set
    
if __name__ == '__main__':
    main()