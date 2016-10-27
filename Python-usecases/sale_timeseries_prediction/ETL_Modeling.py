
# coding: utf-8

# In[ ]:

get_ipython().magic(u'matplotlib inline')

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:

########################################################################
# Do ETL                                                               #
########################################################################

# Load data from csv file
df = pd.read_csv("data/training.csv", header=0)

# Configurations
NUM_PRODUCTS = 2
NUM_PAST_DAYS = 15
NUM_FUTURE_DAYS = 15

# Form initial dataframe with separate columns for product series
dic = {"date": df.TSDate.unique().tolist()}
for i in range(1, NUM_PRODUCTS + 1):
    dic["series{}".format(i)] = df[df.serieNames=="serie_{}".format(i)]             .sales.reset_index().drop(["index"], axis=1).sales
etl_df = pd.DataFrame(dic)

# Scale numeric columns
numeric_cols = ["series{}".format(i+1) for i in range(NUM_PRODUCTS)]
scaler = MinMaxScaler().fit(etl_df.loc[:, numeric_cols])
etl_df.loc[:, numeric_cols] = scaler.transform(etl_df.loc[:, numeric_cols])

# Add more columns of days of week and days of month
tf_date = pd.to_datetime(etl_df.date)
etl_df["dayofweek"] = tf_date.dt.dayofweek
#etl_df["dayofmonth"] = tf_date.dt.day

# Factorize categorical features
cat_cols = ["dayofweek"]#, "dayofmonth"]
for col in cat_cols:
    etl_df[col] = pd.factorize(etl_df[col])[0]

# Fit onehot for categorical features
onehot_encoder = OneHotEncoder()
dummies = onehot_encoder.fit_transform(etl_df[cat_cols])
etl_df = pd.concat([etl_df.drop(cat_cols, axis=1),
                   pd.DataFrame(dummies.toarray())], axis=1)

# Shift past and future days for all products
for i in range(1, NUM_PRODUCTS + 1):
    # Shift past days
    for p_day in range(1, NUM_PAST_DAYS + 1):
        etl_df["series{0}_p{1}".format(i, p_day)] = etl_df.series1.shift(p_day)

    # Shift future days
    for f_day in range(1, NUM_FUTURE_DAYS + 1):
        etl_df["series{0}_f{1}".format(i, f_day)] = etl_df.series1.shift(-f_day)

# Drop the date column
etl_df.drop(["date"], axis=1, inplace=True)

# Keep the last day for future prediction
last_day = etl_df.iloc[-1]

# Drop NaN values
etl_df.dropna(inplace=True)


# In[ ]:

########################################################################
# Develope machine learning model with cross-validation on time series #
########################################################################

def train_model(x_train, y_train, alpha=1e-3, hid_layers=[512], max_iter=100):
    """
    Train model on training data.
    :param x_train: training examples
    :param y_train: target variables
    :param alpha: L2 regularization coefficient
    :param hid_layers: hidden layer sizes
    :param max_iter: maximum number of iterations in L-BFGS optimization
    :return a model trained with neuron network
    """
    nn_model = MLPRegressor(solver='lbgfs', hidden_layer_sizes=hid_layers, 
                            alpha=alpha, max_iter=max_iter, 
                            activation="relu", random_state=1)
    nn_model.fit(x_train, y_train)
    
    return nn_model


def timeseries_cv(x, y, kfolds=10, alpha=1e-3, hid_layers=[512], max_iter=100):
    """
    Implement a cross-validation procedure for time series data.
    :param x: training and validation examples
    :param y: target variables
    :param kfolds: number of folds for cross-validation
    :param alpha: L2 regularization coefficient
    :param hid_layers: hidden layer sizes
    :param max_iter: maximum number of iterations in L-BFGS optimization
    :return cross-validation training and validation loss
    """
    # Number of examples in each fold
    k = int(np.floor(float(x.shape[0]) / kfolds))
    
    num_loops, acc_train_loss, acc_val_loss = 0, 0, 0

    # Loop from the first 2 folds
    for i in range(2, kfolds + 1):
        # Get current training and validation data, slide through time
        x_ = x[:(k*i)]
        y_ = y[:(k*i)]

        # Define a split point for training and validation
        split = float(i-1)/i
        index = int(np.floor(x_.shape[0] * split))
        
        # Training folds
        x_train = x_[:index]        
        y_train = y_[:index]
        
        # Validation folds
        x_val = x_[index:]
        y_val = y_[index:]
        
        # Train model with current sliding data
        model = train_model(x_train, y_train, alpha=alpha, 
                            hid_layers=hid_layers, max_iter=max_iter)

        # Compute train and validation loss with current sliding data
        train_loss = np.mean(np.mean((model.predict(x_train) - y_train)**2))
        val_loss = np.mean(np.mean((model.predict(x_val) - y_val)**2))
        
        # Accumulate train and validation loss
        acc_train_loss += train_loss
        acc_val_loss += val_loss
        num_loops += 1

    return acc_train_loss/num_loops, acc_val_loss/num_loops


# In[ ]:

########################################################################
# Model tuning, currently the code enables to tune 2 hyper parameters, #
# namely hid_layers and alpha. The tuning can be measured by using     # 
# cross-validation evaluation. More hyper parameters can also be       #
# extended easily.                                                     #
########################################################################

# Hyper parameters for tuning
hid_layers = (50,)
alpha = 1

# Get target columns from the etl_df dataframe
y_cols = ["series{0}_f{1}".format(i+1, j+1) for j in range(
        NUM_FUTURE_DAYS) for i in range(NUM_PRODUCTS)]

# Get feature columns from the etl_df dataframe
x_cols = [col for col in etl_df.columns.tolist() if col not in y_cols]

# Form x and y sets
x = etl_df[x_cols]
y = etl_df[y_cols]

# Do cross-validation and draw training versus validation loss
max_iters = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
kfolds = 10

train_loss_cv = np.zeros(len(max_iters))
val_loss_cv = np.zeros(len(max_iters)) 
for i, max_iter in enumerate(max_iters):
    train_loss_cv[i], val_loss_cv[i] = timeseries_cv(x, y, kfolds=kfolds, 
                                                     alpha=alpha, 
                                                     hid_layers=hid_layers,
                                                     max_iter=max_iter)


# In[ ]:

# Plot train versus validation loss
plt.figure(figsize=(5, 3))
plt.xlim([0, 100])
plt.ylim([0, 0.02])
plt.xlabel("Number of iterations")
plt.ylabel("Loss")
plt.plot(max_iters, train_loss_cv, label="Training cross validation loss")
plt.plot(max_iters, val_loss_cv, label="Validation cross validation loss")
plt.legend(shadow=True)


# In[ ]:

########################################################################
# Answer questions                                                     #
########################################################################

# First, train model with best hyper-parameters found after tuning
model = train_model(x, y, alpha=alpha, hid_layers=hid_layers, 
                    max_iter=100)


# Predict for the next 15 days from 2015/11/16 to 2015/11/30
y_pred = model.predict(last_day[x_cols])

# Unscale the predicted values to the original ranges
for i in range(NUM_FUTURE_DAYS):
    y_pred[:, (2*i):(2*(i+1))] = np.round(scaler.inverse_transform(y_pred[:, (2*i):(2*(i+1))]))

print "Predict of Product 1: {}".format(y_pred[0, 0::2])
print "Predict of Product 2: {}".format(y_pred[0, 1::2])


# In[ ]:

# Graphical output with the model fitting (2013/06/21 to 2015/11/15)
y_fit = model.predict(x)
y_tru = np.array(y)

# Unscale the predicted values to the original ranges
for i in range(15):
    y_fit[:, (2*i):(2*(i+1))] = scaler.inverse_transform(y_fit[:, (2*i):(2*(i+1))])
    y_tru[:, (2*i):(2*(i+1))] = scaler.inverse_transform(y_tru[:, (2*i):(2*(i+1))])

# Draw fitting
for j in range(NUM_FUTURE_DAYS):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 3))
    axes[0].set_title("Series1, future day {}".format(j+1))
    axes[0].set_xlabel("Day")
    axes[0].set_ylabel("Sale")
    axes[0].plot(range(y_tru.shape[0])[-120:], y_tru[:, 2*j][-120:], label="Data")
    axes[0].plot(range(y_fit.shape[0])[-120:], y_fit[:, 2*j][-120:], label="Fitted")
    axes[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, shadow=True)
   
    axes[1].set_title("Series2, future day {}".format(j+1))
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("Sale")
    axes[1].plot(range(y_tru.shape[0])[-120:], y_tru[:, 2*j + 1][-120:], label="Data")
    axes[1].plot(range(y_fit.shape[0])[-120:], y_fit[:, 2*j + 1][-120:], label="Fitted")
    axes[1].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, shadow=True)


# In[ ]:

# Implement grid search for hyper parameter tuning based on 
# cross-validation. Currently it only supports for 2 hyper 
# parameters, namely hid_layers and alpha.
def grid_search(x, y):
    # Define a grid of hyper parameters
    hid_layers_list = [(40,), (45,), (50,), (20, 20,), (25, 20,), (25, 25)]
    alpha_list = [0.001, 0.01, 0.1, 1, 1.1, 1.2, 1.3]
    
    best_hid_layers, best_alpha, best_loss = None, None, 1e9
    for hid_layers in hid_layers_list:
        for alpha in alpha_list:
            _, val_loss_cv = timeseries_cv(x, y, kfolds=10, 
                                           alpha=alpha, 
                                           hid_layers=hid_layers,
                                           max_iter=100)
            # Update best hyper parameters
            if val_loss_cv < best_loss:
                best_hid_layers = hid_layers
                best_alpha = alpha
                best_loss = val_loss_cv
    
    return best_hid_layers, best_alpha

