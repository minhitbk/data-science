'''
Created on Jun 23, 2016

@author: minhtran
'''
import os

import numpy as np
import pandas as pd
import pymc3 as pm
from scipy import optimize
from theano import tensor
import logging

from configuration import Config


config = Config()

def build_matrix():
    '''
    This function is used to split randomly the data set into train and test
    sets.
    '''
    df = np.asarray(pd.read_csv(config.data_path))
    mask = np.random.rand(config.num_user, config.num_item)
    train = df.copy()
    test = df.copy()
    train[mask <= 0.2] = np.nan
    test[mask > 0.2] = np.nan
    return train, test

def build_model(data):
    '''
    This function is used to build a Bayesian inference model for estimating
    missing ratings.
    '''
    # Perform mean value imputation
    nan_mask = np.isnan(data)
    data[nan_mask] = data[~nan_mask].mean()
    
    # Set to the mean variance across users and items
    alpha_u = 1/data.var(axis=1).mean()
    alpha_v = 1/data.var(axis=0).mean()
    
    # Build the model
    with pm.Model() as pmf:
        # Prior distribution for the latent U matrix representing users
        U = pm.MvNormal("U", mu=0, tau=alpha_u*np.eye(config.num_dim), 
                        shape=(config.num_user, config.num_dim), 
                        testval=np.random.randn(config.num_user, 
                                            config.num_dim)*config.init_std)
        
        # Prior distribution for the latent V matrix representing items
        V = pm.MvNormal("V", mu=0, tau=alpha_v*np.eye(config.num_dim),
                        shape=(config.num_item, config.num_dim), 
                        testval=np.random.randn(config.num_item, 
                                            config.num_dim)*config.init_std)
        
        # The likelihood function for computing ratings
        _ = pm.Normal("R", mu=tensor.dot(U, V.T), 
                      tau=config.alpha*np.ones((config.num_user, 
                                               config.num_item)), 
                      observed=data)
    return pmf

def train_model(model):
    '''
    This function is used to train a Bayesian model.
    '''
    with model:
        # Find a MAP starting value
        print "...Start finding map..."
        map_start = pm.find_MAP(fmin=optimize.fmin_powell, disp=True)

        # Use the MAP value as a good start for fast converging and use
        # the NUTS strategy for sampling
        step = pm.NUTS(scaling=map_start)
        backend = pm.backends.Text(os.path.dirname(config.save_path))
        logging.info('Backing up trace to directory: %s' % 
                     os.path.dirname(config.save_path))
        
        # Sampling with the NUTS strategy
        print "...Starting sampling..."
        trace = pm.sample(config.n_samples, step, start=map_start, 
                          njobs=2, trace=backend)
    return trace, map_start

def predict(self, U, V):
    '''
    This function is used to estimate ratings from the given values of U 
    and V matrixes.
    '''
    R = np.dot(U, V.T)
    sample_R = np.array([
                         [np.random.normal(R[i,j], np.sqrt(1.0 / config.alpha)) 
                          for j in xrange(config.num_item)]
                        for i in xrange(config.num_user)
                        ])

    # Bound ratings
    sample_R[sample_R < config.min_bound] = config.min_bound
    sample_R[sample_R > config.max_bound] = config.max_bound
    return sample_R

def rmse(test_data, predicted):
    '''
    Calculate root mean squared error.
    '''
    # Indicator for missing values
    indicator = ~np.isnan(test_data)
    
    # Number of non-missing values
    num_value = indicator.sum()
    
    # Compute MSE
    sq_error = abs(test_data - predicted)**2
    mse = sq_error[indicator].sum()/num_value
    return np.sqrt(mse)     

def eval_with_MAP(map_start, train, test):
    U = map_start.map["U"]
    V = map_start.map["V"]

    # Make predictions and calculate RMSE on train & test with MAP estimate
    predictions = predict(U, V)
    train_rmse = rmse(train, predictions)
    test_rmse = rmse(test, predictions)
    overfit = test_rmse - train_rmse

    # Print report
    print "MAP training RMSE: %.5f" % train_rmse
    print "MAP testing RMSE:  %.5f" % test_rmse
    print "Train/test difference: %.5f" % overfit
    return test_rmse

def eval_with_trace(trace, test_data):
    '''
    Calculate RMSE for each step of the trace to monitor convergence.
    '''
    R = np.zeros(test_data.shape)
    for cnt, sample in enumerate(trace[config.burn_in:]):
        sample_R = predict(sample["U"], sample["V"])
        R += sample_R
        running_R = R/(cnt + 1)

    # Return the final predictions and the RMSE calculations
    test_rmse = rmse(test_data, running_R)
    return running_R, test_rmse

def main():
    # Set up train and test data
    print "Start building train and test sets..."
    train, test = build_matrix()
    
    # Build Bayesian model
    print "Start building model..."
    model = build_model(train)
    
    # Fit Bayesian model
    print "Start fitting model..."
    trace, map_start = train_model(model)
    
    # Evaluate the fitting with point estimate and posterior predictive 
    # distributions
    test_rmse_map = eval_with_MAP(map_start, train, test)
    print "Test RMSE with MAP: %g" % test_rmse_map
    
    final_ratings, final_test_rmse = eval_with_trace(trace, test)
    print "Test RMSE with Trace: %g" % final_test_rmse
    
    print np.asarray(final_ratings)
        
if __name__ == '__main__':
    main()
    
    
