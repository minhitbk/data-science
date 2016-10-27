import os

from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder
import pandas as pd
import numpy as np

import utils
from configuration import Config


def _preprocess_data():
    # Load store sale with state
    all_stores = pd.read_pickle(Config.store_data_path)

    # Add more columns of days of week and days of month
    trn_sls_dte = pd.to_datetime(all_stores.trn_sls_dte)
    all_stores["dayofweek"] = trn_sls_dte.dt.dayofweek
    all_stores["dayofmonth"] = trn_sls_dte.dt.day

    # Load weather data
    weather = pd.read_csv(Config.weather_data_path)

    # Join store sales and weather data
    store_weather = pd.merge(all_stores, weather,
                             left_on=["store_id", "trn_sls_dte"],
                             right_on=["location", "date"])
    store_weather.drop(['date', 'location', 'max_temp', 'min_temp'],
                       axis=1, inplace=True)

    # Sort the dataframe with date_time
    store_weather.sort_values(["store_id", "trn_sls_dte"], inplace=True)

    # Remove store with smaller than 180 days
    store_ids = store_weather.store_id.unique()
    for sid in store_ids:
        c_store = store_weather[store_weather.store_id == sid]
        if c_store.shape[0] < 180:
            store_weather = store_weather[store_weather.store_id != sid]

    # Shift the response variables to the past 1 day
    store_weather["p_total_revenue"] = store_weather.total_revenue.shift(1)
    store_weather["p_total_volume"] = store_weather.total_volume.shift(1)

    # Drop the first day of each store
    store_size = store_weather.groupby(["store_id"]).size()
    row_drop_idx = np.cumsum(store_size) - store_size
    store_weather.drop(store_weather.index[row_drop_idx], axis=0, inplace=True)

    # Backup store_ids
    store_weather["store_id_bk"] = store_weather.store_id

    # Factorize categorical features
    cat_cols = ["dayofweek", "dayofmonth", "state", "isholiday", "store_id"]
    for col in cat_cols:
        store_weather[col] = pd.factorize(store_weather[col])[0]

    # Drop the time column
    store_weather.drop(["trn_sls_dte"], axis=1, inplace=True)

    # Save the preprocessed dataframe to pickle object
    utils.to_pickle(Config.save_dir + "store_weather.pkl", store_weather)

    return store_weather


def _train_test_split():
    # Build the store_weather dataframe
    store_weather_filename = Config.save_dir + "store_weather.pkl"
    if os.path.exists(store_weather_filename):
        store_weather = utils.from_pickle(store_weather_filename)
    else:
        store_weather = _preprocess_data()

    # Split train test for each store
    train = pd.DataFrame({})
    test = pd.DataFrame({})
    store_ids = store_weather.store_id_bk.unique()
    for sid in store_ids:
        c_store = store_weather[store_weather.store_id_bk == sid]
        s_train = c_store[:-Config.test_size]
        s_test = c_store[-Config.test_size:]
        train = train.append(s_train).reset_index().drop(["index"], axis=1)
        test = test.append(s_test).reset_index().drop(["index"], axis=1)

    # Scale numeric columns
    num_cols = ["p_total_revenue", "p_total_volume", "mean_temp",
                "total_precipitation", "total_snow"]
    scaler = MaxAbsScaler().fit(train.loc[:, num_cols])
    train.loc[:, num_cols] = scaler.transform(train.loc[:, num_cols])
    test.loc[:, num_cols] = scaler.transform(test.loc[:, num_cols])

    # Scale 2 output columns
    revenue_scale = MaxAbsScaler().fit(train.loc[:, ["total_revenue"]])
    volume_scale = MaxAbsScaler().fit(train.loc[:, ["total_volume"]])
    train.loc[:, ["total_revenue"]] = revenue_scale.transform(
        train.loc[:, ["total_revenue"]])
    test.loc[:, ["total_revenue"]] = revenue_scale.transform(
        test.loc[:, ["total_revenue"]])
    train.loc[:, ["total_volume"]] = volume_scale.transform(
        train.loc[:, ["total_volume"]])
    test.loc[:, ["total_volume"]] = volume_scale.transform(
        test.loc[:, ["total_volume"]])

    # Save the train/test dataframes to pickle objects
    utils.to_pickle(Config.save_dir + "train_set.pkl", train)
    utils.to_pickle(Config.save_dir + "test_set.pkl", test)

    # Save the 2 scaler for later use
    utils.to_pickle(Config.save_dir + "revenue_scale", revenue_scale)
    utils.to_pickle(Config.save_dir + "volume_scale", volume_scale)

    # Save store_ids
    utils.to_pickle(Config.save_dir + "store_id.pkl", store_ids)

    return train, test


def final_data(embedding=True, num_var2keeps=None, cat_var2keeps=None):
    # Build train/test dataframes
    train_filename = Config.save_dir + "train_set.pkl"
    test_filename = Config.save_dir + "test_set.pkl"
    if os.path.exists(train_filename) and os.path.exists(test_filename):
        train = utils.from_pickle(train_filename)
        test = utils.from_pickle(test_filename)
    else:
        train, test = _train_test_split()

    train = train[num_var2keeps + cat_var2keeps +
                  ["store_id_bk", "total_revenue", "total_volume"]]
    test = test[num_var2keeps + cat_var2keeps +
                ["store_id_bk", "total_revenue", "total_volume"]]

    if embedding:
        return _data2embedding(train, test)
    else:
        return _data2onehot(train, test, cat_var2keeps)


def _split_input_target(train, test):
    output_cols = ["total_revenue", "total_volume"]
    y_train = train.loc[:, output_cols + ["store_id_bk"]]
    x_train = train.drop(output_cols, axis=1)
    y_test = test.loc[:, output_cols + ["store_id_bk"]]
    x_test = test.drop(output_cols, axis=1)

    return x_train, y_train, x_test, y_test


def _data2embedding(train, test):
    return _split_input_target(train, test)


def _col2dummies(data, cat_var2keeps):
    for col in cat_var2keeps:
        col_to_dummies = pd.get_dummies(data[col], prefix=col)
        data = pd.concat([data.drop([col], axis=1), col_to_dummies], axis=1)

    return data


def _data2onehot(train, test, cat_var2keeps):
#    train = _col2dummies(train, cat_var2keeps)
#    test = _col2dummies(test, cat_var2keeps)

    onehot_encoder = OneHotEncoder()
    train_dummies = onehot_encoder.fit_transform(train[cat_var2keeps])
    test_dummies = onehot_encoder.transform(test[cat_var2keeps])

    train = pd.concat([train.drop(cat_var2keeps, axis=1),
                       pd.DataFrame(train_dummies.toarray())], axis=1)
    test = pd.concat([test.drop(cat_var2keeps, axis=1),
                      pd.DataFrame(test_dummies.toarray())], axis=1)

    return _split_input_target(train, test)

