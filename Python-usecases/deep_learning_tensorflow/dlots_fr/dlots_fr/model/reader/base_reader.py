""" 
Written by Minh Tran <minhitbk@gmail.com>, Jan 2017.
"""
import abc

from ...config.cf_container import Config


class BaseReader(object):
    """
    This class contains implementations of all basic functions of a data reader.
    It is necessary to extend this class in order to build a complete reader
    for feeding a TensorFlow program.
    """

    def __init__(self):

        self._data_file = None
        self._data_path = Config.data_path

    def __enter__(self):

        try:
            self._data_file = open(self._data_path, "r")

        except IOError:
            print "File %s is not found!" % self._data_path

    def __exit__(self, exc_type, exc_value, traceback):

        self._data_file.close()

    @abc.abstractmethod
    def get_batch(self):
        """
        This function is used to read data as batch per time.
        """
        return

    @property
    def get_data_file(self):

        return self._data_file

