'''
Created on Jun 10, 2016

@author: minhtran
'''
from ..cores.core_data_reader import CoreDataReader
from itertools import islice
import json
import numpy as np

class FeedingReaderLSTM(CoreDataReader):
    '''
    This class contains implementations of a data reader that will feed data to a TensorFlow
    by using the feeding mechanism.
    '''
    
    def __init__(self, trainable_config, untrainable_config):
        '''
        Constructor.
        '''
        super(FeedingReaderLSTM, self).__init__(trainable_config, untrainable_config)
        self._type = "feeddict"
        
    def get_batch(self, data_file):
        '''
        This function implements the abstract method of the super class and is used to read 
        data as batch per time.
        '''
        behave_batch, user_ids = [], []
        lines_gen = islice(data_file, self._untrainable_config._batch_size)
        for line in lines_gen:
            behave_batch.append(self.padding_or_cutting(json.loads(line).values()[0]))
            user_ids.append(json.loads(line).keys()[0])
        
        # Batch is empty then exit
        if not behave_batch: return (None,)*4
    
        # Batch is not empty
        input_batch = np.asarray(behave_batch)[:, :-1]
        output_batch = np.asarray(behave_batch)[:, 1:]
        state_batch = np.zeros([len(behave_batch), 2*self._untrainable_config._lstm_size])
        return input_batch, output_batch, state_batch, user_ids
