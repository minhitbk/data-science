'''
Created on Jun 10, 2016

@author: minhtran
'''
import abc
import numpy as np

class CoreDataReader(object):
    '''
    This class contains implementations of all basic functions of a data reader. It is 
    necessary to extend this class in order to build a complete reader for feeding a 
    TensorFlow program.
    '''

    def __init__(self, trainable_config, untrainable_config):
        '''
        Constructor.
        '''
        self._trainable_config = trainable_config
        self._untrainable_config = untrainable_config
        
#     def __enter__(self):
#         self._data_file = open(self._untrainable_config._data_path, "r")
#         return self
#         
#     def __exit__(self, exc_type, exc_value, traceback):
#         self._data_file.close()
        
    @abc.abstractmethod
    def get_batch(self, param):
        '''
        This function is used to read data as batch per time.
        '''
        return
    
    def get_data_file(self):
        return self._data_file
    
    def padding_or_cutting(self, user_behave):
        '''
        This function is used to pad zeros into or cut a sequence according to the limitation
        of the number of time series events.
        '''
        if (len(user_behave) < self._trainable_config._num_max_event + 1):
            user_behave += [[0 for _ in range(len(user_behave[0]))] for _ in 
                            range(self._trainable_config._num_max_event - len(user_behave) + 1)]
        else:        
            user_behave = user_behave[(len(user_behave) - self._trainable_config._num_max_event 
                                       - 1) : len(user_behave)]
        user_behave = np.asarray(user_behave)
        user_behave[:, 5] = user_behave[:, 5] / 99
        return user_behave
    
    
    