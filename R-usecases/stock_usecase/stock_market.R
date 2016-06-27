###############################################################################
# (c) Copyright 2016, Arimo, Inc. All Rights Reserved.
# @author: minhtran
#
# This file contains scripts for processing the usecase of stock market with
# pipeline.

# Function to add past/future values
addPastFutureValues <- function(formula, df, k=10, m=10) {
  
  # Number of observations
  dataLength <- nrow(df)
  featureName <- attr(terms(formula), 'term.labels')
  data <- df[, featureName]
  
  # Generate the data frame
  # First, add the past
  df <- df[c((m+1):dataLength), ]
  colNames <- colnames(df)
  if (m > 0) {
    for (i in (1:m)) {
      df <- data.frame(df, data[((m-i+1):(dataLength-i))])
      colNames <- c(colNames, paste0(featureName, 'Past', i))
    }
  }
  
  # Second, add the future
  df <- df[c(1:(dataLength-m-k)), ]
  if (k > 0) {
    for (i in (1:k)) {
      df <- data.frame(df, data[((m+i+1):(dataLength-k+i))])
      colNames <- c(colNames, paste0(featureName, 'Future', i))    
    }
  }
  colnames(df) <- colNames
  return (df)
}

# Main processing
df <- read.csv('data/sp500.csv', sep=' ')
df$Index <- as.Date(df$Index)
df <- df[order(df$Index), ]

df$AverPrice <- (df$High + df$Close + df$Low) / 3
averPriceDF <- addPastFutureValues(formula('~ AverPrice'), df, k=10, m=0)
averPriceListCols <- c('AverPriceFuture1', 'AverPriceFuture2', 'AverPriceFuture3', 
                       'AverPriceFuture4', 'AverPriceFuture5', 'AverPriceFuture6', 
                       'AverPriceFuture7', 'AverPriceFuture8', 'AverPriceFuture9', 
                       'AverPriceFuture10')

for (i in averPriceListCols) {
  averPriceDF[i] <- (averPriceDF[i] - averPriceDF['Close']) / 
                                      averPriceDF['Close']
}

p <- 0.025
averPriceDF$T.ind <- apply(averPriceDF[averPriceListCols], 1, 
                           function(x) sum(x[x > p | x < -p]))

pastCloseDF <- addPastFutureValues(formula('~ Close'), averPriceDF, k=0, m=10)

closeListCols <- c('ClosePast1', 'ClosePast2', 'ClosePast3', 'ClosePast4', 
                   'ClosePast5', 'ClosePast6', 'ClosePast7', 'ClosePast8', 
                   'ClosePast9', 'ClosePast10')

for (i in closeListCols) {
  pastCloseDF[i] <- (pastCloseDF['Close'] - pastCloseDF[i]) / pastCloseDF[i]
}

pastCloseDF$ATR <- ATR(pastCloseDF[c('High', 'Low', 'Close')])[,'atr']
pastCloseDF$SMI <- SMI(pastCloseDF[c('High', 'Low', 'Close')])[,'SMI']
pastCloseDF$ADX <- ADX(pastCloseDF[c('High', 'Low', 'Close')])[,'ADX']
pastCloseDF$Aroon <- aroon(pastCloseDF[c('High', 'Low')])[ , 'oscillator']
pastCloseDF$BB <- BBands(pastCloseDF[c('High', 'Low', 'Close')])[, 'pctB']
pastCloseDF$ChaikinVol <- chaikinVolatility(pastCloseDF[c('High', 'Low')])
pastCloseDF$CLV <- EMA(CLV(pastCloseDF[c('High', 'Low', 'Close')]))
pastCloseDF$EMV <- EMV(pastCloseDF[c('High', 'Low')], 
                       pastCloseDF['Volume'])[, 2]
pastCloseDF$MACD <- MACD(pastCloseDF['Close'])[, 2]
pastCloseDF$MFI <- MFI(pastCloseDF[c('High', 'Low', 'Close')], 
                       pastCloseDF['Volume'])
pastCloseDF$SAR <- SAR(pastCloseDF[c('High', 'Close')])
pastCloseDF$Volat <- volatility(pastCloseDF[c('Open', 'High', 'Low', 'Close')], 
                                calc='garman')

pastCloseDF$Decision[pastCloseDF$T.ind > 0.1] <- 1 
pastCloseDF$Decision[pastCloseDF$T.ind < -0.1] <- -1 
pastCloseDF$Decision[pastCloseDF$T.ind <= 0.1 & pastCloseDF$T.ind >= -0.1] <- 0
pastCloseDF$Decision <- as.factor(pastCloseDF$Decision)

finalDF <- pastCloseDF[complete.cases(pastCloseDF), c('Open', 'High', 'Low',
                    'Close', 'Volume','AdjClose', 'ClosePast1', 'ClosePast2', 
                    'ClosePast3', 'ClosePast4', 'ClosePast5', 'ClosePast6', 
                    'ClosePast7', 'ClosePast8', 'ClosePast9', 'ClosePast10',
                    'ATR', 'SMI', 'ADX', 'Aroon', 'BB', 'EMV', 'ChaikinVol',
                    'MFI', 'SAR', 'Volat','MACD', 'Decision')]

trainDF <- finalDF[(1:8000), ]

xtrain <- trainDF[c('MACD', 'ADX', 'Volat', 'Open', 'Close', 'MFI', 'SAR', 
                    'Volume', 'SMI', 'Aroon', 'Low', 'ChaikinVol', 'High',
                    'EMV', 'AdjClose')]
ytrain <- trainDF[ , c('Decision')]

testDF <- finalDF[(8001:9962), ]
xtest <- testDF[c('MACD', 'ADX', 'Volat', 'Open', 'Close', 'MFI', 'SAR',
                  'Volume', 'SMI', 'Aroon', 'Low', 'ChaikinVol', 'High',
                  'EMV', 'AdjClose')]
ytest <- testDF[ , c('Decision')]

set.seed(1234)
rf <- randomForest(x=xtrain, y=ytrain, xtest=xtest, ytest=ytest, 
                   ntree=50, mtry=3, importance=TRUE)
rf$test$confusion
