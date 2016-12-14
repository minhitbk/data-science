""" 
 Copyright (C) Arimo, Inc - All Rights Reserved.
 Unauthorized copying of this file, via any medium is strictly prohibited.
 Proprietary and confidential.
 Written by Minh Tran <minhtran@arimo.com>, Nov 2016.
"""
import os

import shutil
import numpy as np
import tensorflow as tf


def clear_model_dir(mdir):
    if os.path.isdir(mdir):
        shutil.rmtree(mdir)

    os.mkdir(mdir)


def xavier_init(fan_in, fan_out, const=1):
    low = -const * np.sqrt(3.0 / (fan_in + fan_out))
    high = const * np.sqrt(3.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high)


def weighted_pick(weights):
    t = np.cumsum(weights)
    s = np.sum(weights)

    return int(np.searchsorted(t, np.random.rand(1) * s))
