'''
Created on Jun 9, 2016

@author: minhtran
'''
from config_parser import config_parser 

class TrainableConfig(object):
    '''
    This class is going to contain all trainable parameters.
    '''
    def __init__(self, net_type):
        '''
        Constructor.
        '''
        # Common configurations
        self._learning_rate = float(config_parser.get("trainable", "learning_rate"))
        self._dropout_ratio = float(config_parser.get("trainable", "dropout_ratio"))
        self._max_grad_norm = float(config_parser.get("trainable", "max_grad_norm"))
        
        # Specific configurations
        if net_type == "lstm":
            self._num_max_event = int(config_parser.get("trainable", "num_max_event"))
            self._num_layer = int(config_parser.get("trainable", "num_layer"))
        elif net_type == "cnn":
            pass
        else:
            pass
