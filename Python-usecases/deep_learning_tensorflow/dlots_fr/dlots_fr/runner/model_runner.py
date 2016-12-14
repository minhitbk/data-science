""" 
 Copyright (C) Arimo, Inc - All Rights Reserved.
 Unauthorized copying of this file, via any medium is strictly prohibited.
 Proprietary and confidential.
 Written by Minh Tran <minhtran@arimo.com>, Nov 2016.
"""
import os

import tensorflow as tf
import numpy as np

from ..model.graph_builder import GraphBuilder
from ..model.reader.data_reader import DataReader
from ..config.cf_container import Config
from ..utils.misc import clear_model_dir


class ModelRunner(object):
    """
    Main class to train/test models.
    """
    def __init__(self):
        pass

    @staticmethod
    def _run_train_step(sess, train_step, obj_func):
        """
        Train a batch of data.
        :param sess:
        :param train_step:
        :param obj_func:
        :return: error
        """
        [_, err] = sess.run([train_step, obj_func])
        return err

    @staticmethod
    def _run_test_step(sess, model_output, obj_func):
        """
        Test a batch of data.
        :param sess:
        :param model_output:
        :param obj_func:
        :return: model output and error
        """
        [m_output, mse] = sess.run([model_output, obj_func])
        return m_output, mse

    def train_model(self, start=True):
        """
        Main method for training.
        :param start: new training or continue with current training
        """
        dr = DataReader()
        with tf.Graph().as_default(), tf.Session() as sess, dr():
            gp = GraphBuilder(sess=sess, train=True)
            sess.run(tf.group(tf.global_variables_initializer(),
                              tf.local_variables_initializer()))

            if not start:
                # Load the model
                saver = tf.train.Saver(tf.all_variables())
                ckpt = tf.train.get_checkpoint_state(os.path.dirname(
                    Config.save_model))
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    print "Model not exist! Start to train a new model."

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            try:
                step, err_list = 0, []
                while not coord.should_stop():
                    err = self._run_train_step(sess, gp.get_train_step,
                                               gp.get_error)
                    err_list.append(err)

                    step += 1
                    if step % 10 == 0:
                        print "Step %d has error: %g, average error: %g" % (
                            step, err, np.mean(err_list))

            except tf.errors.OutOfRangeError:
                clear_model_dir(os.path.dirname(Config.save_model))
                saver = tf.train.Saver(tf.global_variables())
                saver.save(sess, Config.save_model)

            finally:
                coord.request_stop()

            coord.join(threads)
            sess.close()

    def test_model(self):
        dr = DataReader()
        with tf.Graph().as_default(), tf.Session() as sess, dr():
            gp = GraphBuilder(sess=sess, train=False)
            sess.run(tf.group(tf.global_variables_initializer(),
                              tf.local_variables_initializer()))

            # Load the model
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(
                Config.save_model))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise IOError("Model not exist!")

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            out_list, err_list = [], []
            try:
                while not coord.should_stop():
                    m_out, err = self._run_test_step(sess, gp.get_pred_output,
                                                     gp.get_error)
                    out_list.append(m_out)
                    err_list.append(err)

            except tf.errors.OutOfRangeError:
                print "Test error is: %g" % np.mean(err_list)
                print "Output: "
                print out_list

            finally:
                coord.request_stop()

            coord.join(threads)
            sess.close()

            return out_list
