import os

import numpy as np
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Activation, Dropout, Merge, Reshape

from configuration import Config
from data_processing import final_data
import utils


def split_features(X):
    X_list = [X[..., [i]] for i in range(X.shape[1])]
    return X_list


def _get_train_func(embedding=True, num_var2keeps=None, cat_var2keeps=None):
    # List of all cols in use
    cols = num_var2keeps + cat_var2keeps

    # For training with onehot representation
    def _train_embedding(x_train, y_train, input_dim, hidden_layers,
                         batch_size, nb_epoch):
        models = []
        for feature in cols:
            if Config.feature_sizes[feature] == 1:
                model = Sequential()
                model.add(Dense(1, input_dim=1))
                models.append(model)
            else:
                model = Sequential()
                model.add(Embedding(Config.feature_sizes[feature],
                                    Config.embedding_sizes[feature],
                                    input_length=1))
                model.add(Reshape(target_shape=(
                    Config.embedding_sizes[feature],)))
                models.append(model)

        model = Sequential()
        model.add(Merge(models, mode='concat'))

        for l in range(len(hidden_layers)):
            model.add(Dense(output_dim=hidden_layers[l], input_dim=input_dim))
            model.add(Dropout(Config.dropout))
            model.add(Activation(Config.activation))
            input_dim = hidden_layers[l]

        model.compile(loss=Config.loss_func, optimizer=Config.optimizer)
        model.fit(split_features(np.array(x_train)), np.array(y_train),
                  batch_size=batch_size, nb_epoch=nb_epoch)

        return model

    def _train_onehot(x_train, y_train, input_dim, hidden_layers, batch_size,
                      nb_epoch):
        model = Sequential()
        for l in range(len(hidden_layers)):
            model.add(Dense(output_dim=hidden_layers[l], input_dim=input_dim))
            model.add(Dropout(Config.dropout))
            model.add(Activation(Config.activation))
            input_dim = hidden_layers[l]

        model.compile(loss=Config.loss_func, optimizer=Config.optimizer)
        model.fit(np.array(x_train), np.array(y_train), batch_size=batch_size,
                  nb_epoch=nb_epoch)

        return model

    if embedding:
        train_func = _train_embedding
    else:
        train_func = _train_onehot

    return train_func


def _get_input_dim(embedding=True, num_var2keeps=None, cat_var2keeps=None):
    # List of cols in use
    cols = num_var2keeps + cat_var2keeps

    # Use suitable dict for computing input_dim
    if embedding:
        dict2use = Config.embedding_sizes
    else:
        dict2use = Config.feature_sizes

    # Compute input dim
    input_dim = np.array([dict2use[col] for col in cols]).sum()

    return input_dim


def run_separate_model(model_idx=None, embedding=True, num_var2keeps=None,
                       cat_var2keeps=None):
    # Form data for training and testing separate models
    x_train, y_train, x_test, y_test = final_data(embedding=embedding,
                                                  num_var2keeps=num_var2keeps,
                                                  cat_var2keeps=cat_var2keeps)

    # Compute the dimension of input
    input_dim = _get_input_dim(embedding=embedding, num_var2keeps=num_var2keeps,
                              cat_var2keeps=cat_var2keeps)

    # Get the corresponding training model
    train_func = _get_train_func(embedding=embedding,
                                 num_var2keeps=num_var2keeps,
                                 cat_var2keeps=cat_var2keeps)

    # Train separate models
    model = train_func(x_train.drop(["store_id_bk"], axis=1),
                       y_train.drop(["store_id_bk"], axis=1),
                       input_dim, Config.hidden_layers,
                       Config.batch_size, Config.nb_epoch)
    fit_and_forecast(model_idx=model_idx, model=model, embedding=embedding,
                     x_train=x_train, x_test=x_test)

    return model


def fit_and_forecast(model_idx, model, embedding, x_train, x_test):
    revenue_scale = utils.from_pickle(Config.save_dir + "revenue_scale")
    volume_scale = utils.from_pickle(Config.save_dir + "volume_scale")
    store_ids = x_test.store_id_bk.unique()

    for sid in store_ids:
        s_train = x_train[x_train.store_id_bk == sid]
        s_test = x_test[x_test.store_id_bk == sid]
        s_x = s_train.append(s_test).reset_index().drop(["index",
                                                         "store_id_bk"], axis=1)

        if embedding:
            s_fit_forecast = model.predict(split_features(np.array(s_x)))
        else:
            s_fit_forecast = model.predict(np.array(s_x))

        s_fit_forecast[:, 0] = revenue_scale.inverse_transform(
            s_fit_forecast[:, 0].reshape(-1, 1))[:, 0]
#        s_fit_forecast[:, 1] = volume_scale.inverse_transform(
#            s_fit_forecast[:, 1])

        if not os.path.exists(Config.output_path):
            os.makedirs(Config.output_path)

        # Store fit and forecast to file
        np.savetxt(Config.output_path + "{0}_{1}.csv".format(model_idx, sid),
                   s_fit_forecast[:, 0])

