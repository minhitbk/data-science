""" 
Written by Minh Tran <minhitbk@gmail.com>, Jan 2017.
"""
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops

from ...config.cf_container import Config


def get_time_steps(org_input):
    """
    Get the time dimension.
    :param org_input:
    :return:
    """
    flat_input = nest.flatten(org_input)
    flat_input = tuple(array_ops.transpose(input_, [1, 0, 2])
                       for input_ in flat_input)
    input_shape = tuple(array_ops.shape(input_) for input_ in flat_input)
    time_steps = input_shape[0][0]

    return time_steps


def embed():
    # Initialize an embedding dict to store embedding weights
    embed_dict = {}

    def embed_features(batch, f_size):
        """
        Embed a feature vector for an event.
        """
        for f in range(Config.num_feature):
            feature_val = batch[:, f]
            num_cat_value = Config.schema[f]

            if f == 0:
                if num_cat_value == 1:
                    vector = tf.reshape(feature_val, [-1, 1])
                else:
                    vector = tf.nn.embedding_lookup(embed_dict[f], tf.cast(
                        feature_val, tf.int32))
            else:
                if num_cat_value == 1:
                    vector = tf.concat(1, [vector, tf.reshape(feature_val,
                                                              [-1, 1])])
                else:
                    vector = tf.concat(1, [vector, tf.nn.embedding_lookup(
                        embed_dict[f], tf.cast(feature_val, tf.int32))])

        result = tf.reshape(vector, [-1, 1, f_size])

        return result

    def embed_events(org_input, f_size):
        """
        Embed all events.
        :param org_input: Original inputs
        :param num_event:
        
        :return: Embedded inputs
        """
        def time_step_loop(time, embedded_input):
            embedded_input = tf.concat(1, [embedded_input, embed_features(
                org_input[:, time, :], f_size)])

            return time + 1, embedded_input

        # Create embedded inputs
        embedded_input = embed_features(org_input[:, 0, :], f_size)
        time_steps = get_time_steps(org_input)
        time = array_ops.constant(1, dtype=dtypes.int32)

        _, embedded_input = control_flow_ops.while_loop(
            cond=lambda time, *_: time < time_steps,
            body=time_step_loop,
            loop_vars=[time, embedded_input],
            shape_invariants=[time.get_shape(),
                              tensor_shape.TensorShape([None, None, f_size])])

        return embedded_input

    def embedding(org_input):
        """
        Embed categorical variables with embeddings.
        :param org_input: Original inputs
        :param num_event:
        :return: Embedded inputs
        """
        # Create the embedding list
        for f in range(Config.num_feature):
            num_cat_value = Config.schema[f]

            if num_cat_value == 1:
                pass
            elif num_cat_value > 1:
                embed_dict[f] = tf.get_variable(
                    name="embed_" + str(f),
                    shape=[num_cat_value, Config.embed_size[f]],
                    trainable=True)
            else:
                raise ValueError("Schema values should be positive integers!")

        # Create embedded inputs
        f_size = np.sum(Config.embed_size)
        embedded_input = embed_events(org_input, f_size)

        return embedded_input

    def one_hot(org_input):
        """
        Embed categorical variables with onehot.
        :param org_input: Original inputs
        :param num_event:
        :return: Embedded inputs
        """
        # Create the embedding list
        for f in range(Config.num_feature):
            num_cat_value = Config.schema[f]

            if num_cat_value == 1:
                pass
            elif num_cat_value > 1:
                embed_dict[f] = tf.Variable(np.identity(
                    num_cat_value, dtype=np.float32),
                    trainable=False,
                    name="embed_"+str(f))
            else:
                raise ValueError("Schema values should be positive integers!")

        # Create embedded inputs
        f_size = np.sum(Config.schema)
        embedded_input = embed_events(org_input, f_size)

        return embedded_input

    # Select an embedding method
    if Config.embed == "embedding":
        embedded_func = embedding

    elif Config.embed == "onehot":
        embedded_func = one_hot

    else:
        raise NotImplementedError("The option %s is not supported, please "
                                  "set embed to 'embedding' or 'onehot'!"
                                  % Config.embed)

    return embedded_func
