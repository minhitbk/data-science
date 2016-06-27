'''
Created on Jun 26, 2016

@author: minhtran
'''

class Config(object):
    '''
    This class contains all configurations for the recommendation program.
    '''
    data_path = "data/input.csv"
    save_path = "save/output.csv"
    num_item = 100
    num_user = 1000
    num_dim = 10
    min_bound = -10
    max_bound = 10
    init_std = 0.01
    alpha = 2
    n_samples = 100
    burn_in = 100
    
    
    