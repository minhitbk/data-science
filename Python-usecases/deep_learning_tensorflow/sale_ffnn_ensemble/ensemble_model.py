import os

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

from configuration import Config
import utils


def form_whole_dataset(num_models, sid):
    # Load all necessary outputs from all models
    models = dict()
    for model_idx in range(num_models):
        model_output = pd.read_csv(Config.output_path + "{0}_{1}.csv".format(
            model_idx, sid), header=None)[0]
        models[str(model_idx)] = model_output

    # Create input data
    x = pd.DataFrame(models)

    # Create target data
    y_train = pd.read_csv(Config.output_path + "train_{}".format(sid),
                          header=None)[1]
    y_test = pd.read_csv(Config.output_path + "test_{}".format(sid),
                         header=None)[1]
    y = y_train.append(y_test).reset_index().drop(["index"], axis=1)

    # Remove the first day in target
    y = y[1:]

    return x, y


# Split train vs. test
def split_train_test(x, y):
    # Split input
    x_train = x[:-Config.test_size]
    x_test = x[-Config.test_size:]

    # Split target
    y_train = y[:-Config.test_size]
    y_test = y[-Config.test_size:]

    return x_train, x_test, y_train, y_test


def run_lr(num_models):
    # Load store_ids
    store_ids = utils.from_pickle(Config.save_dir + "store_id.pkl")

    # Compute mse, mae and mape
    mse, mae, mape = 0.0, 0.0, 0.0
    for sid in store_ids:
        if not os.path.exists(
                        Config.output_path + "{0}_{1}.csv".format(0, sid)):
            continue

        # Prepare train/test
        x, y = form_whole_dataset(num_models, sid)
        x_train, x_test, y_train, y_test = split_train_test(x, y)

        # Build linear regression model and do prediction
        lrm = LinearRegression()
        lrm.fit(x_train, y_train)
        pred = lrm.predict(x_test)

        # Compute mse, mae and mape
        mse += metrics.mean_squared_error(y_test, pred)
        mae += metrics.mean_absolute_error(y_test, pred)
        mape += np.mean(abs(y_test - pred) / y_test)

    mse /= len(store_ids)
    mae /= len(store_ids)
    mape /= len(store_ids)

    return mse, mae, mape
