'''
Created on Apr 20, 2016

@author: minhtran
'''
import tensorflow as tf
from encoder import AutoEncoder
from configuration import Config

# The main function                            
def main(_):
    # Load configuration
    config = Config()
    
    # Load feature description
    #feature_desc = [4, 13, 6, 6, 57, 1, 2, 2, 2, 2, 2, 2, 2, 2] 
    #feature_desc = [13, 6, 6, 2, 2, 2, 2, 2, 2, 2]
     
    # Rebuild the graph
    with tf.Graph().as_default(), tf.Session() as sess:
        auto_encoder = AutoEncoder(config)
        auto_encoder.build_encoder()
        sess.run(tf.initialize_all_variables())
    
        # Load the auto encoding model
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state('save')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
    
        # Generate representations for users
        with open(config.data_path, 'r') as data_file, open(config.rep_path, 'w') as rep_file:
            auto_encoder.gen_rep(sess, data_file, rep_file)

if __name__ == "__main__":
    tf.app.run()