'''
Created on Jun 17, 2016

@author: minhtran
'''
from ..data_readers.pipeline_lstm import PipelineReaderLSTM
from ..config.trainable_config import TrainableConfig
from ..config.untrainable_config import UnTrainableConfig
import tensorflow as tf

trainable_config = TrainableConfig(net_type="lstm")
untrainable_config = UnTrainableConfig(net_type="lstm")
reader = PipelineReaderLSTM(trainable_config, untrainable_config)

# Convert data format
reader.convert_to_tfrecord("input")

# Get the stuffs for testing
input_batch, output_batch, state_batch, user_id_batch = reader.get_batch()

# The op for initializing the variables
init_op = tf.initialize_all_variables()

# Create a session for running operations in the Graph
sess = tf.Session()

# Initialize the variables (the trained variables and the epoch counter)
sess.run(init_op)

# Start input enqueue threads
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

try:
    step = 0
    while not coord.should_stop():
        # res1, res2, res3, res4 = sess.run([input_batch, output_batch, state_batch, user_id_batch])
        res = sess.run([input_batch])
        print (res)        
except tf.errors.OutOfRangeError:
    print("Done.")
finally:
    coord.request_stop()

# Wait for threads to finish.
coord.join(threads)
sess.close()