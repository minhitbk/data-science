import os

import tensorflow as tf
import numpy as np

from dlots_fr.model.reader.data_reader import DataReader
from dlots_fr.config.cf_container import Config
from dlots_fr.runner.model_runner import ModelRunner
from dlots_fr.model.framework.embed import embed
from dlots_fr.model.framework.encode import encode


def test_data_reader():
    dr = DataReader()
    with tf.Graph().as_default(), tf.Session() as sess, dr():
        x_batch, y_batch, seq_len_batch, context_batch = dr.get_batch()

        sess.run(tf.group(tf.global_variables_initializer(),
                          tf.local_variables_initializer()))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            while not coord.should_stop():
                print sess.run([x_batch, y_batch, seq_len_batch, context_batch])

        except tf.errors.OutOfRangeError:
            print "erorrrr"

        finally:
            coord.request_stop()

        coord.join(threads)
        sess.close()


def test_tfrecords():
    filename = os.path.join(os.path.dirname(Config.data_path),
                            "input.tfrecords")

    for serialized_example in tf.python_io.tf_record_iterator(filename):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
        features = tf.parse_single_example(serialized_example, features={
            "x": tf.VarLenFeature(tf.string),
            "y": tf.VarLenFeature(tf.string),
            "seq_len": tf.FixedLenFeature([], tf.int64)})

        x = np.fromstring(example.features.feature["x"].bytes_list.value[0],
                          dtype=np.float32)
        y = np.fromstring(example.features.feature["y"].bytes_list.value[0],
                          dtype=np.float32)
        seq_len = example.features.feature["seq_len"].int64_list.value[0]
        print x, y, seq_len


def test_graph():
    dr = DataReader()
    with tf.Graph().as_default(), tf.Session() as sess, dr():
        x_batch, y_batch, x_length = dr.get_batch()
        embedded_func = embed()
        embedded_input = embedded_func(x_batch)
        encoded_input = encode(embedded_input, x_length)

        sess.run(tf.group(tf.global_variables_initializer(),
                          tf.local_variables_initializer()))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            while not coord.should_stop():
                print sess.run([x_batch, y_batch, x_length,
                                encoded_input])

        except tf.errors.OutOfRangeError:
            print "errorrrr"

        finally:
            coord.request_stop()

        coord.join(threads)
        sess.close()


def test_train():
    mr = ModelRunner()
    mr.train_model(start=True)


def test_test():
    mr = ModelRunner()
    mr.test_model()


def main():
    test_test()
    #test_train()
    #test_data_reader()
    #test_tfrecords()
    #test_graph()

if __name__ == "__main__":
    main()
