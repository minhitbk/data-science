'''
Created on Jun 17, 2016

@author: minhtran
'''
import tensorflow as tf
from ..auto_encoders.basic_lstm_encoder import BasicLSTMEncoder
from ..data_readers.pipeline_lstm import PipelineReaderLSTM
from ..config.trainable_config import TrainableConfig
from ..config.untrainable_config import UnTrainableConfig
import numpy as np
import time

# The main function                            
def main(_):
    # Setup configurations
    trainable_config = TrainableConfig(net_type="lstm")
    untrainable_config = UnTrainableConfig(net_type="lstm")

    # Setup a data reader
    data_reader = PipelineReaderLSTM(trainable_config, untrainable_config)

    # Build the graph and run session
    with tf.Graph().as_default(), tf.Session() as sess:
        auto_encoder = BasicLSTMEncoder(data_reader, trainable_config, untrainable_config)
        sess.run(tf.initialize_all_variables())
       
        # Start input enqueue threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            start_time = time.time()
            step, mse_list = 0, []
            while not coord.should_stop():
                [_, mse] = sess.run([auto_encoder.get_train_step(), auto_encoder.get_mse()])
                mse_list.append(mse)
                if (step % 40 == 0) and (step != 0):
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

if __name__ == "__main__":
    tf.app.run()