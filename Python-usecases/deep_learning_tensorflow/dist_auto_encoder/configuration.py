'''
Created on Apr 21, 2016

@author: minhtran
'''
# Define configuration
class Config(object):
    data_path = 'data/sample_json.txt'
    save_path = 'save/model.ckpt'
    rep_path = 'rep/representation.txt'
    batch_size = 1000
    num_epoch = 30
    lstm_size = 32
    init_std = 0.01
    num_event = 20 
    max_grad_norm = 10
    vector_threshold = 64  
    learning_rate = 0.05
    num_top_feature = 3
    num_min_event = 10
    #feature_desc = [4, 13, 6, 6, 57, 1, 2, 2, 2, 2, 2, 2, 2, 2] 
    feature_desc = [13, 6, 6, 2, 2, 2, 2, 2, 2, 2]
