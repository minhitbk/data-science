###############################################################################
# (c) Copyright 2016, Arimo, Inc. All Rights Reserved.
# @author: minhtran
#
# This file contains scripts for processing the usecase of stock market with
# pipeline.

library(arimo)

# Connect to server
arimo.connect('dev.adatao.com', username='guest', password='welcome1')

# Create pipelines
firstPipeline <- arimo.createPipeline()
secondPipeline <- arimo.createPipeline()

# Get DDF data
ddfSP500 <- arimo.getDDF('ddf://adatao/stockmarket_sp500')

# Set DDF data as Input Stage
arimo.setDDFAsStageInput(firstPipeline, ddfSP500, inputName='ddfSP500')

# Transform Index to Date
# THIS IS NOT SUPPORTED YET IN PIPELINE
transformIndexToDate <- arimo.createTransformStage('Index=as.Date(Index)') 

# Order the ddf according to Index
# THIS IS NOT SUPPORTED IN PIPELINE YET. IN GENERAL, TIME SERIES IS ALSO NOT
# SUPPORTED BY DISTRIBUTED DATAFRAME, THIS SHOULD BE CHANGED TO DEAL WITH TIME 
# SERIES DATA
orderDdfSP500 <- arimo.createOrderStage('ordered Index')

# Transform to calculate average HLC
transformAverPrice <- arimo.createTransformStage('AverPrice=mean(High, Low, 
                                                                 Close)')

averPriceCols <- c('AverPriceFuture1', 'AverPriceFuture2', 'AverPriceFuture3', 
                   'AverPriceFuture4', 'AverPriceFuture5', 'AverPriceFuture6', 
                   'AverPriceFuture7', 'AverPriceFuture8', 'AverPriceFuture9', 
                   'AverPriceFuture10')

# Transform to add future values of average price
# THIS IS NOT SUPPORTED YET WHILE THIS TRANSFORMATION IS IMPORTANT FOR 
# PROCESSING TIME SERIES DATA
transformFutureAverPrice <- arimo.createTransformStage(formula='~ AverPrice',
                         colnames=averPriceCols, past=0, future=10, drop=TRUE)

# Update future values of average price
# THIS IS NOT SUPPORTED YET, IT COULD BE DONE IN PARALLEL
updateFutureAverPrice <- arimo.createTransformStage(colnames=averPriceCols,
                                                    'col=(col-Close)/Close')

# Transform to create the evaluation Indicator
# IT IS NOT SURE IF THIS KIND OF TRANSFORMATION IS SUPPORTED IN PIPELINE
transformIndicator <- arimo.createTransformStage('T.ind=apply(averPriceCols,
                              1, function(x) sum(x[x > 0.025 | x < -0.025]))')

closeCols <- c('ClosePast1', 'ClosePast2', 'ClosePast3', 'ClosePast4', 
               'ClosePast5', 'ClosePast6', 'ClosePast7',  
               'ClosePast8', 'ClosePast9', 'ClosePast10')

# Transform to add past values of the Close column
transformPastClose <- arimo.createTransformStage(formula='~ Close',
                             colnames=closeCols, past=10, future=0, drop=TRUE)

# Update past values of the Close column
updatePastClose <- arimo.createTransformStage(colnames=closeCols,
                                              'col=(Close-col)/col')


# Transform to create a new feature ATR. It is similarly used for creating 
# other new features such as SMI, ADX, etc.
# IT IS NOT SURE IF THIS KIND OF TRANSFORMATION IS SUPPORTED IN PIPELINE
transformAddATR <- arimo.createTransformStage("ATR=ATR(ddf[c('High', 'Low', 
                                                      'Close')])[, 'atr']") 

# Transform to create Decision target
transformAddDecision <- arimo.createTransformStage('Decision=ifelse(T.ind < 
                                       -0.1, -1, ifelse(T.ind <= 0.1, 0, 1))')

# Transform to factorize the Decision target
transformFactorDecision <- arimo.createSetAsFactorStage('Decision')

# Filter to create train/test data set
trainDataSet <- arimo.createFilterStage("Index < '2000-12-31'")
testDataSet <- arimo.createFilterStage("Index >= '2000-12-31'")

keepList = c('Open', 'High', 'Low', 'Close', 'Volume', 'AdjClose', 'ClosePast1', 
        'ClosePast2', 'ClosePast3', 'ClosePast4', 'ClosePast5', 'ClosePast6', 
        'ClosePast7', 'ClosePast8', 'ClosePast9', 'ClosePast10', 'ATR', 'BB',
        'SMI', 'ADX', 'Aroon', 'EMV', 'ChaikinVol', 'MFI', 'SAR', 'Volat',
        'MACD', 'Decision')

# Drop NA rows and filter a number of columns
# IT IS NOT SURE IF THIS KIND OF FILTER IS SUPPORTED IN PIPELINE
filterRowsColumnsTrain <- arimo.createFilterStage(rows='complete.cases',
                                                  varsToKeep=keepList)
filterRowsColumnsTest <- arimo.createFilterStage(rows='complete.cases',
                                                 varsToKeep = keepList)

# Create random forest stage
randomForest <- arimo.createRandomForestStage(algo='classification',
                                            featureColumns=keepList[-c(28)], 
                                            labelColumn='Decision', 
                                            minInfoGain=0.01, 
                                            operationMode='Train')

# Create a chain for training
arimo.chainStages(firstPipeline, transformIndexToDate, orderDdfSP500,
                         transformAverPrice, transformFutureAverPrice, 
                         updateFutureAverPrice, transformIndicator, 
                         transformPastClose, updatePastClose, transformAddATR,
                         transformAddDecision, transformFactorDecision,
                         trainDataSet, filterRowsColumnsTrain, randomForest)

# Start train
arimo.runPipeline(firstPipeline)

# Scoring
arimo.setStageParam(randomForest, randomForest@operationMode, 'Score')

# Get output of the transformFactorDecision stage
tmpDDF <- arimo.getStageOutput(transformFactorDecision)

# Set tmpDDF as input stage
arimo.setDDFAsStageInput(secondPipeline, tmpDDF, inputName='tmpDDF')

# Create a chain for testing
arimo.chainStages(secondPipeline, testDataSet, filterRowsColumnsTest, 
                                               randomForest)

# Start test
arimo.runPipeline(secondPipeline)

# Get evaluation results
evalDDF <- arimo.getStageOutput(randomForest)

