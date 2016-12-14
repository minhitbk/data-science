""" 
 Copyright (C) Arimo, Inc - All Rights Reserved.
 Unauthorized copying of this file, via any medium is strictly prohibited.
 Proprietary and confidential.
 Written by Minh Tran <minhtran@arimo.com>, Dec 2016.
"""
import tensorflow as tf

from ...utils.misc import xavier_init
from ...config.cf_container import Config


def _act_func():
    if Config.act_func == "relu":
        return tf.nn.relu
    elif Config.act_func == "sigmoid":
        return tf.nn.sigmoid
    elif Config.act_func == "tanh":
        return tf.nn.tanh
    elif Config.act_func == "linear":
        return lambda x: x
    else:
        raise ValueError("Activation function not supported!")


def predict(input):
    """
    Build a feedforward network.
    :param input:
    :return:
    """
    # Check if the list of dropouts is the same as the list of layers
    if len(Config.keep_drop) != len(Config.layers):
        raise ValueError("Hidden layers and keep_drop list should be "
                         "corresponding!")

    hid_matrix = [None]*len(Config.layers)
    hid_bias = [None]*len(Config.layers)
    output = [None]*(len(Config.layers)+1)

    act_func = _act_func()

    input_dim = input.get_shape().as_list()[1]
    output[0] = input
    for (i, l) in enumerate(Config.layers):
        hid_matrix[i] = tf.Variable(xavier_init(input_dim, l),
                                    name="predict_matrix_{}".format(i))
        hid_bias[i] = tf.Variable(tf.truncated_normal(
            [l], stddev=0.1, dtype=tf.float32),
            name="predict_bias_{}".format(i))
        output[i+1] = act_func(tf.nn.dropout(tf.matmul(
            output[i], hid_matrix[i]) + hid_bias[i], Config.keep_drop[i]))
        input_dim = l

    return output[-1]
