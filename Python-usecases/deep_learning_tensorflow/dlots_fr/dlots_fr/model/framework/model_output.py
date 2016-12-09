""" 
 Copyright (C) Arimo, Inc - All Rights Reserved.
 Unauthorized copying of this file, via any medium is strictly prohibited.
 Proprietary and confidential.
 Written by Minh Tran <minhtran@arimo.com>, Dec 2016.
"""
import tensorflow as tf

from ...utils.misc import xavier_init
from predict import _act_func
from ...config.cf_container import Config


def _model_func():
    if Config.model_type == "classification":
        return _classifier
    elif Config.model_type == "regression":
        return _regressor
    else:
        raise ValueError("Model type %s is not supported yet!"
                         % Config.model_type)


def _classifier(input, output_dim):
    input_dim = input.get_shape()[1]

    hid_matrix = tf.Variable(xavier_init(input_dim, output_dim))
    hid_bias = tf.Variable(tf.truncated_normal([output_dim], stddev=0.1,
                                               dtype=tf.float32))
    output = tf.nn.softmax(tf.matmul(input, hid_matrix) + hid_bias)


def _regressor(input, output_dim):
    input_dim = input.get_shape()[1]
    act_func = _act_func()

    hid_matrix = tf.Variable(xavier_init(input_dim, output_dim))
    hid_bias = tf.Variable(tf.truncated_normal([output_dim], stddev=0.1,
                                               dtype=tf.float32))
    output = act_func(tf.matmul(input, hid_matrix) + hid_bias)

    return output


def model_output(input, output_dim):
    """
    Generate model output.
    :param output_dim:
    :return:
    """
    model_func = _model_func()
    result = model_func(input, output_dim)

    return result
