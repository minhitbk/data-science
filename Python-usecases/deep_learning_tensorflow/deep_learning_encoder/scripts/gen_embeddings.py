'''
Created on Jun 14, 2016

@author: minhtran
'''
import tensorflow as tf
from ..auto_encoders.basic_lstm_encoder import BasicLSTMEncoder
from ..data_readers.feed_dict_lstm import FeedingReaderLSTM
from ..config.trainable_config import TrainableConfig
from ..config.untrainable_config import UnTrainableConfig
import os

# The main function                            
def main(_):
    # Setup configurations
    trainable_config = TrainableConfig(net_type="lstm")
    untrainable_config = UnTrainableConfig(net_type="lstm")

    # Setup a data reader
    data_reader = FeedingReaderLSTM(trainable_config, untrainable_config)
        
    # Rebuild the graph and run session
    with tf.Graph().as_default(), tf.Session() as sess:
        auto_encoder = BasicLSTMEncoder(data_reader, trainable_config, untrainable_config)
        sess.run(tf.initialize_all_variables())
    
        # Load the auto encoding model
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(untrainable_config._save_path))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else: 
            print "There are no model available, please train one..."
            return
    
            # Generate representations for users
        with open(untrainable_config._data_path, "r") as data_file, open(
                                    untrainable_config._rep_path, "w") as rep_file:
            auto_encoder.gen_rep(sess, data_file, rep_file)

if __name__ == "__main__":
    tf.app.run()