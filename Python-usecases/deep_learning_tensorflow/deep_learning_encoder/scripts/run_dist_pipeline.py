'''
Created on Jun 15, 2016

@author: minhtran
'''
import tensorflow as tf
import numpy as np
from ..auto_encoders.basic_lstm_encoder import BasicLSTMEncoder
from ..data_readers.pipeline_lstm import PipelineReaderLSTM
from ..config.trainable_config import TrainableConfig
from ..config.untrainable_config import UnTrainableConfig
import time

# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS

def average_gradients(tower_grads, trainable_config):
    '''
    Calculate the average gradient for each shared variable.
    '''
    average_grads, tvars = [], []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)
        tvars.append(grad_and_vars[0][1])
        average_grads.append(grad)
         
    norm_grads, _ = tf.clip_by_global_norm(average_grads, trainable_config._max_grad_norm)
    return zip(norm_grads, tvars) 

# The main function                            
def main(_):
    # Setup configurations
    trainable_config = TrainableConfig(net_type="lstm")
    untrainable_config = UnTrainableConfig(net_type="lstm")
    
    # Setup a data reader
    data_reader = PipelineReaderLSTM(trainable_config, untrainable_config)
        
    # Get ps and worker servers
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    
    # Create and start a server for the local task
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    
    if FLAGS.job_name == "ps":
        # Initialize the autoencoders
        auto_encoders = [None for _ in range(len(worker_hosts))]
        
        # Build the graph and run
        with tf.Graph().as_default(), tf.Session(server.target) as sess:
            # Initialize grads, mse and embeddings
            total_grads, total_mse, embeddings = [], tf.constant(0.0), None
            
            # Build the optimizer
            optimizer = tf.train.AdamOptimizer(trainable_config._learning_rate)
            
            # Build sub-graphs for each worker
            for worker in range(len(worker_hosts)):
                with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" 
                                                              % worker, cluster=cluster)): 
                    if worker > 0: tf.get_variable_scope().reuse_variables()
                    
                    # Build a sub-graph
                    auto_encoders[worker] = BasicLSTMEncoder(data_reader, trainable_config, 
                                                    untrainable_config, embeddings=embeddings, 
                                                    worker=worker, optimizer=optimizer)
                    
                    # Accumulate grads and mse
                    total_grads.append(auto_encoders[worker].get_grads())
                    total_mse += auto_encoders[worker].get_mse() / len(worker_hosts)
                
                    # Get embeddings
                    embeddings = auto_encoders[worker].get_embeddings()

            # Define the norm_grads
            norm_grads = average_gradients(total_grads, trainable_config)   
        
            # Define a training step
            train_step = optimizer.apply_gradients(norm_grads)

            # Run initializations
            sess.run(tf.initialize_all_variables())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                start_time = time.time()
                step, mse_list = 0, []
                while not coord.should_stop():
                    [_, mse] = sess.run([train_step, total_mse])
                    mse_list.append(mse)
                    if (step % 20 == 0) and (step != 0):
                        print ("Step %d has mean square error: %g, average mse: %g" % (step, mse, 
                                                                                np.mean(mse_list)))
                        mse_list = []
                    step += 1
            except tf.errors.OutOfRangeError:
                duration = time.time() - start_time
                print("Total time training: %g" % duration)
                saver = tf.train.Saver(tf.all_variables())
                saver.save(sess, untrainable_config._save_path)            
                print("Done training, saved model.")
            finally:
                coord.request_stop()

            # Wait for threads to finish
            coord.join(threads)        
    elif FLAGS.job_name == "worker":
        server.join()

# The main method
if __name__ == "__main__":
    tf.app.run()
    

    