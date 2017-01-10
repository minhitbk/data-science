""" 
Written by Minh Tran <minhitbk@gmail.com>, Jan 2017.
"""
import tensorflow as tf

from reader.data_reader import DataReader
from ..model.framework.embed import embed
from ..model.framework.encode import encode
from ..model.framework.attend import attend
from ..model.framework.predict import predict
from ..model.framework.model_output import model_output
from ..config.cf_container import Config


class GraphBuilder(object):

    def __init__(self, sess, train=True):
        self._sess = sess
        if train:
            self._train_step, self._obj_func = self._build_model_for_train()
        else:
            self._pred_output, self._obj_func = self._build_model_for_test()

    @staticmethod
    def _declare_inputs():
        """
        Declare inputs of a tensorflow graph.
        """
        dr = DataReader()
        with dr():
            x_batch, y_batch, x_length, context = dr.get_batch()

        return x_batch, y_batch, x_length, context

    def _inference(self, x_batch, y_batch, x_length, context):
        """
        Build graph.
        :param x_batch:
        :param y_batch:
        :param x_length:
        :return:
        """
        # First stage: embedding
        embedded_func = embed()
        embedded_input = embedded_func(x_batch)

        # Second stage: encoding
        encoded_input = encode(embedded_input, x_length)

        # Third stage: attention
        attended_func = attend()
        attended_input = attended_func(context, encoded_input, self._sess)

        # Fourth stage: prediction
        predicted_input = predict(attended_input)

        # Final stage
        final_output = model_output(predicted_input)

        return final_output

    @staticmethod
    def _loss(pred_output, true_output):
        """
        Define an objective function.
        """
        if Config.model_type == "classification":
            obj_func = -tf.reduce_sum(pred_output * tf.log(true_output))
        elif Config.model_type == "regression":
            obj_func = tf.reduce_mean(
                tf.square(tf.sub(pred_output, true_output)))
        else:
            raise ValueError("Model type %s is not supported yet!"
                             % Config.model_type)

        return obj_func

    @staticmethod
    def _train(obj_func):
        # Create gradient steps
        tvars = tf.trainable_variables()
        grads = tf.gradients(obj_func, tvars)
        norm_grads, _ = tf.clip_by_global_norm(grads, Config.max_grad_norm)

        # Define a train step
        optimizer = tf.train.AdamOptimizer(Config.learning_rate)
        train_step = optimizer.apply_gradients(zip(norm_grads, tvars))

        return train_step

    def _build_model_for_train(self):
        """
        Build the total graph for training.
        """
        x_batch, y_batch, x_length, context = self._declare_inputs()
        pred_output = self._inference(x_batch, y_batch, x_length, context)
        obj_func = self._loss(pred_output, y_batch)
        train_step = self._train(obj_func)

        return train_step, obj_func

    def _build_model_for_test(self):
        """
        Rebuild the model for test.
        """
        x_batch, y_batch, x_length, context = self._declare_inputs()
        pred_output = self._inference(x_batch, y_batch, x_length, context)
        obj_func = self._loss(pred_output, y_batch)

        return pred_output, obj_func

    @property
    def get_pred_output(self):
        return self._pred_output

    @property
    def get_error(self):
        return self._obj_func

    @property
    def get_train_step(self):
        return self._train_step
