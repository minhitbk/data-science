""" 
 Copyright (C) Arimo, Inc - All Rights Reserved.
 Unauthorized copying of this file, via any medium is strictly prohibited.
 Proprietary and confidential.
 Written by Minh Tran <minhtran@arimo.com>, Nov 2016.
"""
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape

from ...config.cf_container import Config
from ...utils.misc import xavier_init
from ...utils.misc import weighted_pick
from ...model.framework.embed import get_time_steps


def attend():
    if Config.attention == "soft":
        return _soft_attention
    elif Config.attention == "hard":
        return _hard_attention
    else:
        raise ValueError("Only soft and hard attentions are supported!")


def _common_attention(state, inputs):
    """
    Implement the common part of attention.
    :param state:
    :param inputs:
    :return:
    """
    def time_step_loop(time, ms):
        cy = tf.concat(1, [state, inputs[:, time, :]])
        w = tf.get_variable("w", shape=[cy_size, Config.att_size])
        p = tf.get_variable("p", shape=[Config.att_size, 1])
        ms = tf.concat(1, [ms, tf.matmul(tf.tanh(tf.matmul(cy, w)), p)])

        return time + 1, ms

    time_steps = get_time_steps(inputs)
    time = array_ops.constant(1, dtype=dtypes.int32)
    cy = tf.concat(1, [state, inputs[:, 0, :]])
    cy_size = cy.get_shape().as_list()[1]
    w = tf.Variable(xavier_init(cy_size, Config.att_size), name="w")
    p = tf.Variable(tf.truncated_normal([Config.att_size, 1], stddev=0.01,
                                        dtype=tf.float32), name="p")
    ms = tf.matmul(tf.tanh(tf.matmul(cy, w)), p)

    _, ms = control_flow_ops.while_loop(
        cond=lambda time, *_: time < time_steps,
        body=time_step_loop,
        loop_vars=[time, ms],
        shape_invariants=[time.get_shape(),
                          tensor_shape.TensorShape([None, None])])

    return tf.nn.softmax(ms)


def _soft_attention(state, inputs, sess):
    """
    Implement the soft part of attention.
    :param state:
    :param inputs:
    :param sess:
    :return:
    """
    def time_step_loop(time, result):
        s = tf.reshape(ca[:, time], [-1, 1])
        input = inputs[:, time, :]
        cell_size = 2*Config.cell_size if Config.encode == "bidirectional_rnn" \
            else Config.cell_size

        for _ in range(1, cell_size):
            s = tf.concat(1, [s, tf.reshape(ca[:, time], [-1, 1])])

        result = tf.add(result, tf.mul(s, input))

        return time + 1, result

    time_steps = get_time_steps(inputs)
    time = array_ops.constant(0, dtype=dtypes.int32)
    ca = _common_attention(state, inputs)
    result = tf.zeros_like(inputs[:, 0, :])

    _, result = control_flow_ops.while_loop(
        cond=lambda time, *_: time < time_steps,
        body=time_step_loop,
        loop_vars=[time, result])

    return result


def _hard_attention(state, inputs, sess):
    """
    Implement the hard part of attention.
    :param state:
    :param inputs:
    :param sess:
    :return:
    """
    # TODO: NOT IMPLEMENTED YET
    # new_inputs = [inputs[:, i, :] for i in range(inputs.get_shape()[1])]
    # ca = _common_attention(state, new_inputs)
    # ss = sess.run([ca])
    #
    # result = new_inputs[weighted_pick(ss)]
    result = None
    return result
