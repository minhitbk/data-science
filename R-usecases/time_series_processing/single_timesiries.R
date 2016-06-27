###############################################################################
# (c) Copyright 2016, Arimo, Inc. All Rights Reserved.
# @author: minhtran
#
# This file contains all implementations with respect to the forecasting of 
# single time series.

library(tseries)
library(GeneCycle)
library(forecast)

# This section defines utility functions used for the single-timeseries module.

# Calculate the norm of a vector using the inner product defined on the 
# p-periodic subspaces: ||x|| = sqrt(<x,x>) = norm(x)/sqrt(length(x)).
internalPeriodNorm <- function(x) {
    norm(x, type="2") / sqrt(length(x))
}

# Find the closest p-periodic vector to the vector s by projecting s onto the 
# periodic subspace P_p.
internalProjectP <- function(s, p) {
    n <- length(s)
    nones <- rep(1, n)
    baselem <- 0 * nones
    for (shift in (1:p)) {
        ind <- seq(from=shift, to=n, by=p)
        baselem[ind] <- nones[ind] * sum(s[ind]) / length(ind)
    }
    return (baselem)
}

# Find all factors of an integer n.
internalFactorP <- function(n) {
    allFactors <- c(1)
    for (i in (2:round(n/2))) {
        if (n/i == round(n/i)) allFactors <- c(allFactors, i)
    }
    return (allFactors)
}

# Calculate the Periodicity Transform using the "M-Best" Algorithm.
internalMBest <- function(s, m=1, n=30) {
    sNorm <- internalPeriodNorm(s)
    listP <- rep(0, m)
    listNorm <- rep(0, m)
    listBas <- matrix(0, m, length(s))
    listChange <- ifelse(m > 1, 1, 0)
    # Build initial list of m best periodicities
    for (i in (1:m)) {
        maxNorm <- 0
        maxP <- 0
        maxBas <- rep(0, length(s))
        for (p in (1:n)) {
            bas <- internalProjectP(s, p)
            if (internalPeriodNorm(bas) > maxNorm) {
                maxP <- p
                maxNorm <- internalPeriodNorm(bas)
                maxBas <- bas
            }
        }
        listP[i] <- maxP
        listNorm[i] <- maxNorm
        listBas[i, ] <- maxBas
        s <- s - maxBas
    }

    # Step 2: decompose basis elements and residuals
    while (listChange == 1) {
        i <- 1
        while (i < m) {
            listChange <- 0
            maxNorm <- 0
            fact <- internalFactorP(listP[i])
            for (nf in (1:length(fact))) {
                p <- fact[nf]
                bas <- internalProjectP(listBas[i, ], p)
                if (internalPeriodNorm(bas) >= maxNorm) {
                    maxP <- p 
                    maxNorm <- internalPeriodNorm(bas)
                    maxBas <- bas
                }
            }
            xbigq <- internalProjectP(listBas[i, ], maxP)
            xsmallq <- listBas[i, ] - xbigq
            nbigq <- internalPeriodNorm(xbigq)
            nsmallq <- internalPeriodNorm(xsmallq)
            minq <- min(listNorm) 
            ptwice <- sum(listP==maxP)
            if (((nsmallq + nbigq) > (listNorm[m] + listNorm[i])) & 
                        (nsmallq > minq) & (nbigq > minq) & (ptwice == 0)) {
                listChange <- 1
                listNorm <- c(listNorm[c(1:(i-1))], nbigq, nsmallq, 
                              listNorm[c((i+1):(m-1))])
                listP <- c(listP[c(1:(i-1))], maxp, listp[c(i:(m-1))])
                if (i > 1) {
                    listBas <- rbind(listBas[c(1:(i-1)), ], maxBas, xsmallq, 
                                     listBas[c((i+1):(m-1)), ])
                } else {
                    listBas <- rbind(maxBas, xsmallq, listBas[c(2:(m-1)), ])
                }
            } else {
                i <- i + 1;
            }
        }
    }
    results <- list('periods'=listP, 'powers'=listNorm/sNorm, 'basis'=listBas)
    return (results)
}

# Function for computing error metric.
internalComputeMetric <- function(x, y) {
    return (mean(abs(x - y)))
}

# Function for forecasting
internalForecast <- function(model, forecastList) {
    forecaster <- model$forecaster
    predictValue <- forecaster[forecastList, 1] + forecaster[forecastList, 2]
    lower <- forecaster[forecastList, 2] + forecaster[forecastList, 4]
    upper <- forecaster[forecastList, 2] + forecaster[forecastList, 5]    
    result <- list('predictValue'=predictValue, 'lower'=lower, 'upper'=upper)
    return (result)
}

# Function for training a split.
internalTrain <- function(data, level, damped, initial, exponential, 
                                                   res, bas, per) {
    # Number of data
    dataLength <- length(data)

    # Model resigduals
    residualModel <- holt(res, h=dataLength, level=level, damped=damped, 
                               initial=initial, exponential=exponential)
    resModel <- residualModel$mean
    
    # Model periodic component
    if (dataLength %% per > 0)
        basModel <- c(bas[(dataLength %% per + 1):dataLength], 
                                   bas[1:(dataLength %% per)])
    else
        basModel <- bas

    # Fitted values
    fitted <- residualModel$fitted + bas

    # Prediction interval
    lower <- residualModel$lower 
    upper <- residualModel$upper

    # Final model
    forecaster <- data.frame(resModel, basModel, fitted, lower, upper)
    model <- list('forecaster'=forecaster, 'level'=level)
    return (model)
}

# Function for modeling with ptholt.
internalModelPtholt <- function(data, level, res, bas, per) {
    # Number of data
    dataLength <- length(data)
    
    # Number of folds for cross-validation
    k <- 10
    # Fold size
    foldSize <- dataLength %/% k

    # Fold starts
    starts <- array(c(1:k))
    for (i in (1:k)) {
        starts[i] <- (i - 1) * foldSize + 1
    }

    # End fold for train and train base data
    kTrain <- k %/% 2
    dataTrain <- data[c(1:(starts[kTrain] - 1))]
    basTrain <- bas[c(1:(starts[kTrain] - 1))]
    resTrain <- res[c(1:(starts[kTrain] - 1))]
    
    # Set parameters for cross-validation
    dampeds <- c(TRUE, FALSE)
    initials <- c('optimal')
    exponentials <- c(FALSE)
    
    # Grid search for cross-validation
    bestErr <- .Machine$double.xmax
    bestDamped <- NULL
    bestInitial <- NULL
    bestExponential <- NULL
    for (d in dampeds)
        for (i in initials)
            for (e in exponentials) {
                # Initialize the test error
                testErr <- 0
                for (j in ((kTrain + 1):k)) {
                    # Split train/test
                    dataTrain <- c(dataTrain, 
                         data[c(starts[j - 1]:(starts[j - 1] + foldSize - 1))])
                    basTrain <- c(basTrain, 
                          bas[c(starts[j - 1]:(starts[j - 1] + foldSize - 1))])
                    resTrain <- c(resTrain, 
                          res[c(starts[j - 1]:(starts[j - 1] + foldSize - 1))])
                    dataTest <- data[c(starts[j]:(starts[j] + foldSize - 1))]
                    # Train on the train set
                    modelTrain <- internalTrain(dataTrain, level, d, i, e, 
                                                  resTrain, basTrain, per)
                    # Forecast on the test set
                    forecastTest <- internalForecast(modelTrain, 
                                          c(1:length(dataTest)))
                    # Accumulate the test errors
                    testErr <- testErr + internalComputeMetric(dataTest, 
                                              forecastTest$predictValue)
                }
                
                # Update best model
                if (testErr < bestErr) {
                    bestErr <- testErr
                    bestDamped <- d
                    bestInitial <- i
                    bestExponential <- e
                }
            }
    
    # Train last time with best parameters
    model <- internalTrain(data, level, bestDamped, bestInitial, 
                                 bestExponential, res, bas, per)
    attr(model, 'modelType') <- 'ptholt'
    return (model)
}

# This function is to model with ptarima
internalModelPtarima <- function(data, level, res, bas, per, season=FALSE) {
    # Number of data
    dataLength = length(data)
    
    # Model resigduals
    resArima <- auto.arima(res, max.p=10, max.q=10, max.d=2, max.order=20,
               start.p=1, start.q=1, max.P=3, max.Q=3, max.D=2, start.P=2, 
                              start.Q=2, seasonal=season, stepwise=FALSE)
    resModel_ <- forecast(resArima, dataLength, level=level)
    resModel <- resModel_$mean
    
    # Model periodic component
    if (dataLength %% per > 0)
        basModel <- c(bas[(dataLength %% per + 1):dataLength], 
                                   bas[1:(dataLength %% per)])
    else
        basModel <- bas
    
    # Fitted values
    fitted <- resModel_$fitted + bas

    # Prediction interval
    lower <- resModel_$lower 
    upper <- resModel_$upper
    
    # Final model
    forecaster <- data.frame(resModel, basModel, fitted, lower, upper)
    model <- list("forecaster"=forecaster, "level"=level)

    if (season == TRUE) attr(model, 'modelType') <- 'ptarima-season'
    else attr(model, 'modelType') <- 'ptarima'
    return (model)
}

# This function is to automatically generate the best arima model
internalModelArima <- function(data, level, season=FALSE) {
    # Number of data
    dataLength <- length(data)
    
    # Train for the best arima model
    modelArima_ <- auto.arima(data, max.p=10, max.q=10, max.d=2, max.order=20,
                       start.p=1, start.q=1, seasonal=season, stepwise=FALSE)
    
    # Create model in accordance to a consistent format, so 2 pseudo lists 
    # are created
    modelArima <- forecast(modelArima_, dataLength, level=level)
    resModel <- modelArima$mean
    basModel <- list(rep(0, dataLength))
    fitted <- modelArima$fitted
    
    # Prediction interval
    lower <- modelArima$lower 
    upper <- modelArima$upper
    
    # Final model
    forecaster <- data.frame(resModel, basModel, fitted, lower, upper)
    model <- list("forecaster"=forecaster, "level"=level)

    if (season == TRUE) attr(model, 'modelType') <- 'arima-season'
    else attr(model, 'modelType') <- 'arima'
    return (model)
}

#' This function is to allow users to select automatically a proper time series 
#' forecasting model, based on the auto-detection of time series patterns.
#'
#' @param data A time series 
#' @param level The confidence values associated with the prediction intervals
#' @param maxPer The maximum period that the algorithm looks for periodicity
#' @return A forecasting model for the time series data
#' @examples
#' model <- arimo.sts.metaAlg(data)
#' model <- arimo.sts.metaAlg(data, maxPer=10)
#' @export
arimo.sts.metaAlg <- function(data, level=95, maxPer=30) {
    # Statistical test for periodicity
    pValue <- fisher.g.test(data)
    
    # First there exists periodicity in data
    if (pValue < 0.01) {
        # Decompose the data
        perTransform = internalMBest(data, m=1, n=maxPer)
        per <- perTransform$periods
        bas <- perTransform$basis[1,]
        res <- data - bas

        # Check the property of level-stationary of the residuals
        levelSta <- kpss.test(res, null=c('Level'))
        # Check the property of trend-stationary of the residuals
        trendSta <- kpss.test(res, null=c('Trend'))
        # Case of level-stationary
        if (levelSta$statistic < 0.146) {
            print('ptholt model is recommended for forecasting.')
            model <- internalModelPtholt(data, level, res, bas, per)
        }
        # Case of non level-stationary
        else {
            # Case of trend-stationary
            if (trendSta$statistic < 0.146) {
                print('ptarima model is recommended for forecasting.')
                model <- internalModelPtarima(data, level, res, bas, per)
            }
            # Case of non trend-stationary
            else {
                print('ptarima with season is recommended for forecasting.')
                model <- internalModelPtarima(data, level, res, bas, per, 
                                                             season=TRUE)
            }
        }
    }
    # Second if periodicity is not present
    else {
        # Check the property of level-stationary
        levelSta_ <- kpss.test(data, null=c('Level'))
        # Check the property of trend-stationary
        trendSta_ <- kpss.test(data, null=c('Trend'))
        # Case of stationary
        if (levelSta_$statistic < 0.146 || trendSta_$statistic < 0.146 ) {
            print('arima model is recommended for forecasting.')
            model <- internalModelArima(data, level)
        }
        # Case of non stationary
        else {
            #print("A multivariate model should be used for forecasting.")
            #model <- arimo.sts.model_mla(data, k, m )
            
            # Temporarily use ptarima
            print('ptarima model is recommended for forecasting.')
            model <- internalModelPtarima(data, level, res, bas, per)
        }
    }
    return (model)
}

#' This function is to allow users to build a forecasting model with the arima
#' algorithm.
#'
#' @param data A time series 
#' @param level The confidence values associated with the prediction intervals
#' @param season A boolean variable to specify whether to search for seasonality
#' @return A forecasting model for the time series data
#' @examples
#' model <- arimo.sts.model_arima(data)
#' @export
arimo.sts.model_arima <- function(data, level=95, season=FALSE) {
    model <- internalModelArima(data, level, season=season)
    return (model)    
}

#' This function is to allow users to build a forecasting model with the ptarima
#' algorithm.
#'
#' @param data A time series 
#' @param level The confidence values associated with the prediction intervals
#' @param maxPer The maximum period that the algorithm looks for periodicity
#' @param season A boolean variable to specify whether to search for seasonality
#' @return A forecasting model for the time series data
#' @examples
#' model <- arimo.sts.model_ptarima(data)
#' model <- arimo.sts.model_ptarima(data, maxPer=10)
#' @export
arimo.sts.model_ptarima <- function(data, level=95, maxPer=30, season=FALSE) {
    # decompose the data
    perTransform = internalMBest(data, m=1, n=maxPer)
    per <- perTransform$periods
    bas <- perTransform$basis[1,]
    res <- data - bas
    model <- internalModelPtarima(data, level, res, bas, per, season=season)
    return (model)
}

#' This function is to allow users to build a forecasting model with the ptholt
#' algorithm.
#'
#' @param data A time series 
#' @param level The confidence values associated with the prediction intervals
#' @param maxPer The maximum period that the algorithm looks for periodicity
#' @return A forecasting model for the time series data
#' @examples
#' model <- arimo.sts.model_ptholt(data)
#' model <- arimo.sts.model_ptholt(data, maxPer=10)
#' @export
arimo.sts.model_ptholt <- function(data, level=95, maxPer=30) {
    # decompose the data
    perTransform = internalMBest(data, m=1, n=maxPer)
    per <- perTransform$periods
    bas <- perTransform$basis[1,]
    res <- data - bas
    model <- internalModelPtholt(data, level, res, bas, per)
    return (model)
}

# This function is used for forecasting a list of future values.
#'
#' @param model A forecasting model returned by an algorithm such as ptarima
#' @param forecastList The list of values to be forecasted. The maximal number 
#' of predicted values is equal to data length
#' @return A list of forecasted values and their prediction intervals
#' @examples
#' model <- arimo.sts.metaAlg(data)
#' result <- arimo.sts.forecast(model, c(1, 2, 3))
#' @export
arimo.sts.forecast <- function(model, forecastList) {
    result <- internalForecast(model, forecastList)
    return (result)
}
