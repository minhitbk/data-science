""" 
 Copyright (C) Arimo, Inc - All Rights Reserved.
 Unauthorized copying of this file, via any medium is strictly prohibited.
 Proprietary and confidential.
 Written by Minh Tran <minhtran@arimo.com>, Nov 2016.
"""
import tensorflow as tf

from ...arimo_tf.mod_rnn import bidirectional_dynamic_rnn
from ...arimo_tf.mod_rnn import dynamic_rnn
from ...config.cf_container import Config


def _create_cell():
    """
    Define a function for creating a cell.
    :return: A callable function
    """
    if Config.cell_type == "LSTMCell":
        cell_func = tf.nn.rnn_cell.LSTMCell
    elif Config.cell_type == "GRUCell":
        cell_func = tf.nn.rnn_cell.GRUCell
    elif Config.cell_type == "BasicRNNCell":
        cell_func = tf.nn.rnn_cell.BasicRNNCell
    else:
        raise NotImplementedError("Cell type %s is not supported yet!"
                                  % Config.cell_type)

    return cell_func


def encode(embedded_input, x_length):
    """
    Define a function for encoding.
    :param embedded_input: X input
    :param x_length: Length of examples
    :return: Input representations
    """
    cell_func = _create_cell()
    cell = cell_func(num_units=Config.cell_size)

    def encode_bd(embedded_input, x_length):
        """
        Encode with a bidirectional rnn.
        :param embedded_input:
        :param x_length:
        :return:
        """
        _, states = bidirectional_dynamic_rnn(
            cell_fw=cell, cell_bw=cell, dtype=tf.float32,
            sequence_length=x_length, inputs=embedded_input)
        states_fw, states_bw = states

        return tf.concat(2, [states_fw, states_bw])

    def encode_sd(embedded_input, x_length):
        """
        Encode with a single rnn.
        :param embedded_input:
        :param x_length:
        :return:
        """
        _, states = dynamic_rnn(cell=cell, dtype=tf.float32,
                                      sequence_length=x_length,
                                      inputs=embedded_input)

        if Config.cell_type == "LSTMCell":
            result = states.c
        else:
            result = states

        return result

    def encode_multiple_rnn(embedded_input, x_length):
        """
        Encode with a chain of multiple rnns
        :param embedded_input:
        :param x_length:
        :return:
        """
        multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell]*Config.num_cell)
        _, states = dynamic_rnn(cell=multi_cell, dtype=tf.float32,
                                      sequence_length=x_length,
                                      inputs=embedded_input)

        if Config.cell_type == "LSTMCell":
            result = states[-1].c
        else:
            result = states[-1]

        return result

    if Config.encode == "bidirectional_rnn":
        rep = encode_bd(embedded_input, x_length)
    elif Config.encode == "single_rnn":
        rep = encode_sd(embedded_input, x_length)
    elif Config.encode == "multiple_rnn":
        rep = encode_multiple_rnn(embedded_input, x_length)
    else:
        raise NotImplementedError("Encoding type %s is not supported yet!"
                                  % Config.encode)

    return rep
