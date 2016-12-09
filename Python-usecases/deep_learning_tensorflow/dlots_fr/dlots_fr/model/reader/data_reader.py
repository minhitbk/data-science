""" 
 Copyright (C) Arimo, Inc - All Rights Reserved.
 Unauthorized copying of this file, via any medium is strictly prohibited.
 Proprietary and confidential.
 Written by Minh Tran <minhtran@arimo.com>, Nov 2016.
"""
import os

import tensorflow as tf
from itertools import islice
import numpy as np
import json

from base_reader import BaseReader
from ...config.cf_container import Config


class DataReader(BaseReader):
    """
    This class contains implementations of a data reader that will feed data
    to TensorFlow by using the data pipeline mechanism.
    """

    def __init__(self):

        super(DataReader, self).__init__()

    def __enter__(self):

        super(DataReader, self).__enter__()

        try:
            self._convert_to_tfrecord("input")

        except IOError:
            print "File %s is not found!" % self._data_path

        except TypeError:
            print "File %s is not opened yet, please put it " \
                  "under a context!" % self._data_path

    def __call__(self):

        return self

    @staticmethod
    def _int64_feature(value):
        """
        Encode a feature to int64.
        """
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _bytes_feature(value):
        """
        Encode a feature to bytes.
        """
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _convert_to_tfrecord(self, file_name):
        """
        This method is used to convert the input data into Tensorflow input.
        """
        # Open a tfrecord writer
        tfrecord_writer = tf.python_io.TFRecordWriter(os.path.join(
            os.path.dirname(self._data_path), file_name + ".tfrecords"))

        while True:
            should_exit = True
            lines_gen = islice(self._data_file, 10000)

            # Process for each line
            for line in lines_gen:
                x = np.asarray(json.loads(line).get("x"),
                               dtype=np.float32).tostring()
                y = np.asarray(json.loads(line).get("y"),
                               dtype=np.float32).tostring()
                seq_len = np.asarray(json.loads(line).get("x"),
                                     dtype=np.float32).shape[0]

                example = tf.train.Example(
                    features=tf.train.Features(feature={
                        "x": self._bytes_feature(x),
                        "y": self._bytes_feature(y),
                        "seq_len": self._int64_feature(seq_len)}))

                tfrecord_writer.write(example.SerializeToString())
                should_exit = False

            # End of file
            if should_exit: break

        tfrecord_writer.close()

    def _read_and_decode(self, filename_queue):
        """
        Decode the data.
        """
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features={
            "x": tf.VarLenFeature(tf.string),
            "y": tf.VarLenFeature(tf.string),
            "seq_len": tf.FixedLenFeature([], tf.int64)})

        # Convert from a scalar string tensor to a float tensor and reshape
        x = tf.reshape(tf.decode_raw(features["x"].values, tf.float32),
                       [-1, Config.num_feature])
        y = tf.reshape(tf.decode_raw(features["y"].values, tf.float32), [-1])
        seq_len = features["seq_len"]

        return x, y, seq_len

    def _read_input(self, file_name):
        """
        Reads input data num_epoch times.
        """
        filename = os.path.join(os.path.dirname(self._data_path),
                                file_name + ".tfrecords")

        with tf.name_scope("input"):
            filename_queue = tf.train.string_input_producer(
                [filename], num_epochs=Config.num_epoch)

            # Even when reading in multiple threads, share the filename queue
            x, y, seq_len = self._read_and_decode(filename_queue)

            # Collect examples into batch_size batches
            # We run this in three threads to avoid being a bottleneck
            x_batch, y_batch, seq_len_batch = tf.train.batch(
                [x, y, seq_len], batch_size=Config.batch_size, num_threads=3,
                capacity=100 + 3 * Config.batch_size, dynamic_pad=True,
                allow_smaller_final_batch=True)

        return x_batch, y_batch, seq_len_batch

    def get_batch(self):
        """
        This function implements the abstract method of the super class and
        is used to read data as batch per time.
        """
        x_batch, y_batch, seq_len_batch = self._read_input("input")

        return x_batch, y_batch, seq_len_batch

