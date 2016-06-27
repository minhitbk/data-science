'''
Created on Apr 26, 2016

@author: minhtran
'''
from configuration import Config
import json
import numpy as np
import pickle

# Function to save an object
def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

# Function to load an object           
def load_object(filename):
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
    return obj

def get_length_dist():
    # Load configuration
    config = Config()
    
    # Get distribution of num_events
    result = []
    with open(config.data_path, 'r') as data_file:
        lines = data_file.readlines()
        for line in lines:
            user_behave = json.loads(line).values()[0]
            result.append(len(user_behave))

    return result

def main():
    dist = get_length_dist()
    print np.percentile(dist, [0, 25, 50, 75, 90, 95, 98])

if __name__ == '__main__':
    main()