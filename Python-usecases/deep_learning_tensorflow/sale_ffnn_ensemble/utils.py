import os

import pickle


def to_pickle(filename, obj):
    dir = os.path.dirname(filename)
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def from_pickle(filename):
    file = open(filename,'rb')
    return pickle.load(file)