'''
Created on Jun 23, 2016

@author: minhtran
'''

class Config(object):
    '''
    This class contains all configurations for the recommendation program.
    '''
    data_path = "data/input.csv"
    save_path = "save/model.ckpt"
    rep_path = "rep/output.csv"
    batch_size = 100
    num_epoch = 100
    max_grad_norm = 3
    learning_rate = 0.01
    num_item = 100
    num_user = 1000
    num_hid_neuron = 20
    init_std = 0.01
    
    