'''
Created on Jun 10, 2016

@author: minhtran
'''
from config_parser import config_parser
import json

class UnTrainableConfig(object):
    '''
    This class is going to contain all untrainable parameters.
    '''
    def __init__(self, net_type):
        '''
        Constructor.
        '''
        # Common configurations
        self._data_path = config_parser.get("untrainable", "data_path")
        self._save_path = config_parser.get("untrainable", "save_path")
        self._batch_size = int(config_parser.get("untrainable", "batch_size"))
        self._num_epoch = int(config_parser.get("untrainable", "num_epoch"))
        self._init_std = float(config_parser.get("untrainable", "init_std"))
        #self._data_list = json.loads(config_parser.get("untrainable", "data_list"))
        
        # Specific configurations
        if net_type == "lstm":
            self._rep_path = config_parser.get("untrainable", "rep_path")
            self._vector_threshold = int(config_parser.get("untrainable", "vector_threshold"))
            self._num_top_feature = int(config_parser.get("untrainable", "num_top_feature"))
            self._feature_desc = json.loads(config_parser.get("untrainable", "feature_desc"))
            self._lstm_size = int(config_parser.get("untrainable", "lstm_size"))
            self._num_feature = int(config_parser.get("untrainable", "num_feature"))
        elif net_type == "cnn":
            pass
        else:
            pass 
    