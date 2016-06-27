'''
Created on Jun 9, 2016

@author: minhtran
'''
import abc

class CoreEncoder(object):
    '''
    This class contains implementations of all basic functions of an auto-encoder. It is 
    necessary to extend this class in order to build a complete RNN/CNN auto-encoder.
    '''
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, data_reader, trainable_config, untrainable_config):
        '''
        Constructor used for initialization.
        '''
        # Set configuration        
        self._trainable_config = trainable_config
        self._untrainable_config = untrainable_config
        
        # Save data_reader
        self._data_reader = data_reader
    
    @abc.abstractmethod
    def declare_input(self):
        '''
        This method is used to declare the inputs of a TensorFlow graph of an encoder.
        '''        
        return
    
    @abc.abstractmethod
    def inference(self, worker=0):
        '''
        This method is used to build the inference part of a TensorFlow graph of an encoder.
        '''
        return
    
    @abc.abstractmethod
    def loss(self, encoder_outputs):
        '''
        This method is used to build the loss part of a TensorFlow graph of an encoder.
        '''        
        return
    
    @abc.abstractmethod
    def train(self, obj_func, optimizer=None):
        '''
        This method is used to build the train part of a TensorFlow graph of an encoder.
        '''
        return
    
    @abc.abstractmethod
    def run_epoch(self, sess):
        '''
        This method is used to run 1 epoch.
        '''
        return
    
    @abc.abstractmethod
    def gen_rep(self, sess, data_file, rep_file):
        '''
        This method is used to generate embeddings and save results in rep_file.
        '''
        return
    
    def get_vector_rep(self):
        '''
        Return the representation of a batch of users.
        '''
        return self._vector_rep

    def get_train_step(self):
        ''' 
        Get the train_step ops.
        '''
        return self._train_step
    
    def get_input_batch(self):
        '''
        Get the input_batch ops.
        '''
        return self._input_batch
    
    def get_output_batch(self):
        '''
        Get the output_batch ops.
        '''
        return self._output_batch
    
    def get_initial_state(self):
        '''
        Get the intial_state ops.
        '''
        return self._initial_state
    
    def get_mse(self):
        '''
        Get the mse ops.
        '''
        return self._mean_square_error
    
    def get_grads(self):
        '''
        Get the grads ops.
        '''
        return self._grads

    def get_tvars(self):
        '''
        Get the tvars ops.
        '''
        return self._tvars
    
    def get_embeddings(self):
        '''
        Get embeddings dict.
        '''
        return self._embeddings
    
    def get_data_reader(self):
        '''
        Get data reader.
        '''
        return self._data_reader
    
    
    