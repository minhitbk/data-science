'''
Created on Jun 17, 2016

@author: minhtran
'''
from ..cores.core_data_reader import CoreDataReader
import tensorflow as tf
import json
import os
from itertools import islice
import numpy as np

class PipelineReaderLSTM(CoreDataReader):
    '''
    This class contains implementations of a data reader that will feed data to a TensorFlow
    by using the data pipeline mechanism.
    '''
    
    def __init__(self, trainable_config, untrainable_config):
        '''
        Constructor.
        '''
        super(PipelineReaderLSTM, self).__init__(trainable_config, untrainable_config)
        self._type = "pipeline"      
        self.convert_to_tfrecord("input")

    def _int64_feature(self, value):
        '''
        Encode a feature to int64.
        '''
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        '''
        Encode a feature to bytes.
        '''
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def convert_to_tfrecord(self, file_name):
        '''
        This method is used to convert the input data into Tensorflow input.
        '''
        # Open a tfrecord writer
        tfrecord_writer = tf.python_io.TFRecordWriter(os.path.join(os.path.dirname(
                                                        self._untrainable_config._data_path), 
                                                                   file_name + ".tfrecords"))        
        with open(self._untrainable_config._data_path) as data_file:
            while True:
                should_exit = True
                lines_gen = islice(data_file, 10000)
                                
                # Process for each line
                for line in lines_gen:
                    user_id = int(json.loads(line).keys()[0])
                    user_behave = np.asarray(self.padding_or_cutting(json.loads(line).values()[0]), 
                                             dtype=np.float32).tostring()
                    example = tf.train.Example(features=tf.train.Features(feature={
                                                "user_id": self._int64_feature(user_id),
                                                "user_behave": self._bytes_feature(user_behave)}))
                    tfrecord_writer.write(example.SerializeToString())
                    should_exit = False
            
                # End of file
                if should_exit: break                
        tfrecord_writer.close()

    def read_and_decode(self, filename_queue):
        '''
        Decodes the data.
        '''
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features={
                                                "user_id": tf.FixedLenFeature([], tf.int64),
                                                "user_behave": tf.VarLenFeature(tf.string),})
        # Get user_id
        user_id = features["user_id"]
        
        # Convert from a scalar string tensor to a float tensor and reshape
        user_behave = tf.reshape(tf.decode_raw(features["user_behave"].values, tf.float32), 
                                                [self._trainable_config._num_max_event + 1, 
                                                 self._untrainable_config._num_feature])
        return user_id, user_behave
    
    def inputs(self, file_name):
        '''
        Reads input data num_epochs times.
        '''
        filename = os.path.join(os.path.dirname(self._untrainable_config._data_path), 
                                                                    file_name + ".tfrecords")
        with tf.name_scope("input"):
            filename_queue = tf.train.string_input_producer([filename], 
                                            num_epochs=self._untrainable_config._num_epoch)
    
            # Even when reading in multiple threads, share the filename queue
            user_id, user_behave = self.read_and_decode(filename_queue)
    
            # Shuffle the examples and collect them into batch_size batches
            # (Internally uses a RandomShuffleQueue.)
            # We run this in two threads to avoid being a bottleneck
            user_id_batch, user_behave_batch = tf.train.shuffle_batch([user_id, user_behave], 
                                            batch_size=self._untrainable_config._batch_size, 
                                            num_threads=3,
                                            capacity=1000+3*self._untrainable_config._batch_size,
                                            min_after_dequeue=1000)
        return user_id_batch, user_behave_batch
    
    def get_batch(self):
        '''
        This function implements the abstract method of the super class and is used to read 
        data as batch per time.
        '''
        user_id_batch, user_behave_batch = self.inputs("input")
        input_batch = user_behave_batch[:, 0:self._trainable_config._num_max_event, :]
        output_batch = user_behave_batch[:, 1:self._trainable_config._num_max_event + 1, :]
        state_batch = np.zeros([user_id_batch.get_shape().as_list()[0], 
                                2*self._untrainable_config._lstm_size], dtype=np.float32)
        return input_batch, output_batch, state_batch, user_id_batch
