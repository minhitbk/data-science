'''
Created on Jun 14, 2016

@author: minhtran
'''
import tensorflow as tf
from ..auto_encoders.basic_lstm_encoder import BasicLSTMEncoder
from ..data_readers.feed_dict_lstm import FeedingReaderLSTM
from ..config.trainable_config import TrainableConfig
from ..config.untrainable_config import UnTrainableConfig
import time

# The main function                            
def main(_):
    # Setup configurations
    trainable_config = TrainableConfig(net_type="lstm")
    untrainable_config = UnTrainableConfig(net_type="lstm")

    # Setup a data reader
    data_reader = FeedingReaderLSTM(trainable_config, untrainable_config)

    # Build the graph and run session
    with tf.Graph().as_default(), tf.Session() as sess:
        auto_encoder = BasicLSTMEncoder(data_reader, trainable_config, untrainable_config)
        sess.run(tf.initialize_all_variables())
        start_time = time.time()
        for epoch in range(untrainable_config._num_epoch):
            with open(untrainable_config._data_path, "r") as data_file:
                mse = auto_encoder.run_epoch(sess, data_file)
                if epoch % 1 == 0:
                    print ("Epoch %d has mean square error: %g" % (epoch, mse))
        duration = time.time() - start_time
        print("Total time training: %g" % duration)
        saver = tf.train.Saver(tf.all_variables())
        saver.save(sess, untrainable_config._save_path)
        print("Done training, saved model.")
        
if __name__ == "__main__":
    tf.app.run()