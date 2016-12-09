""" 
 Copyright (C) Arimo, Inc - All Rights Reserved.
 Unauthorized copying of this file, via any medium is strictly prohibited.
 Proprietary and confidential.
 Written by Minh Tran <minhtran@arimo.com>, Nov 2016.
"""
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops

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
    cys = [tf.concat(1, [state, input]) for input in inputs]
#    cys = [tf.concat(1, [state, inputs[:, i, :]])
#           for i in range(inputs.get_shape()[1])]
    cy_size = cys[0].get_shape()[1]

    ms = []
    for cy in cys:
        w = tf.Variable(xavier_init(cy_size, Config.att_size))
        p = tf.Variable(tf.truncated_normal([Config.att_size, 1],
                                            stddev=0.01,
                                            dtype=tf.float32))
        ms.append(tf.matmul(tf.tanh(tf.matmul(cy, w)), p))

    return tf.nn.softmax(tf.concat(1, ms))


def _soft_attention(state, inputs, sess):
    """
    Implement the soft part of attention.
    :param state:
    :param inputs:
    :param sess:
    :return:
    """
    time_steps = get_time_steps(inputs)
    time = array_ops.constant(1, dtype=dtypes.int32)

    _, embedded_input = control_flow_ops.while_loop(
        cond=lambda time, *_: time < time_steps,
        body=time_step_loop,
        loop_vars=[time, embedded_input],
        shape_invariants=[time.get_shape(),
                          tensor_shape.TensorShape([None, None, f_size])])

    new_inputs = [inputs[:, i, :] for i in range(inputs.get_shape()[1])]
    ca = _common_attention(state, new_inputs)
    ca_len = ca.get_shape()[1]
    ss = tf.split(1, ca_len, ca)

    result = tf.zeros_like(inputs[:, 0, :])
    for (new_input, s) in zip(new_inputs, ss):
        result = tf.add(result, tf.mul(s, new_input))

    return result


def _hard_attention(state, inputs, sess):
    """
    Implement the hard part of attention.
    :param state:
    :param inputs:
    :param sess:
    :return:
    """
    new_inputs = [inputs[:, i, :] for i in range(inputs.get_shape()[1])]
    ca = _common_attention(state, new_inputs)
    ss = sess.run([ca])

    result = new_inputs[weighted_pick(ss)]

    return result
