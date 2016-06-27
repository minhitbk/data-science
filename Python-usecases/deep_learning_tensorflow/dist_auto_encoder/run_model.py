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
    
    # Build the graph and run session
    with tf.Graph().as_default(), tf.Session() as sess:
        autoEncoder = AutoEncoder(config)
        autoEncoder.build_model()
        sess.run(tf.initialize_all_variables())
        for epoch in range(config.num_epoch):
            with open(config.data_path, 'r') as data_file:
                mse = autoEncoder.run_epoch(sess, data_file)
            if epoch % 1 == 0:
                print ('Epoch %d has mean square error: %g' % (epoch, mse))

        saver = tf.train.Saver(tf.all_variables())
        saver.save(sess, config.save_path)
        
if __name__ == "__main__":
    tf.app.run()