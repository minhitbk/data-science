###############################################################################
# (c) Copyright 2016, Arimo, Inc. All Rights Reserved.
# @author: minhtran
#
# This file contains scripts for processing the usecase of predictive 
# maintenance of harddrive failures, which is applied to the blogpost 
# entitled 'Predictive Maintenance with Big Apps'.

library(arimo)
source('compute_metric.R')

# Function for training
trainBestModel <- function(ddfData, trainFormula) {
    
    # Number of folds for cross-validation
    kFold <- 5
    
    # Permutate data to change the random order
    randomData <- arimo.sample2ddf(ddfData, fraction=1, replace=FALSE)
    
    # Generate a list of kFold train/test tuples
    listTrainTest <- arimo.cv.kfold(randomData, nFolds=kFold)
    
    # Set parameters for cross-validation / grid search
    learningRates <- c(0.01, 0.02, 0.03)
    numIterations <- c(50, 100, 200)
    lambdas <- c(0.01, 0.02, 0.03)
    thresholds <- c(0.8, 0.9, 0.99, 0.999, 0.9999)

    # Initialize best parameters
    bestMetric <- 99999999999
    bestLearningRate <- NULL
    bestNumIteration <- NULL
    bestLambda <- NULL
    bestThreshold <- NULL
    
    # Cross-validate
    for (learningRate in learningRates)
        for (numIteration in numIterations)
            for (lambda in lambdas)
                for (threshold in thresholds) {
                    print("Start a new loop...")
                    # Initialize the test metric
                    validationMetric <- 0
                    for (i in (1:kFold)) {
                        # Get train/validation sets
                        train <- listTrainTest[[i]][['train']]
                        validation <- listTrainTest[[i]][['test']]
                        trainModel <- arimo.glm.gd(formula=trainFormula, 
                                                   data=train, 
                                                   learningRate=learningRate, 
                                                   numIterations=numIteration,
                                                   regularized='ridge', 
                                                   lambda=lambda, 
                                                   initialWeights=NULL,
                                                   ref.levels=NULL)                      
                        
                        # Accumulate the validation metric
                        validationMetric <- validationMetric + 
                                            computeMetric(trainModel, 
                                                          validation, 
                                                          threshold) / kFold
                    }
                    
                    # Update best model
                    if (validationMetric < bestMetric) {
                        bestMetric <- validationMetric
                        bestLearningRate <- learningRate
                        bestNumIteration <- numIteration
                        bestLambda <- lambda
                        bestThreshold <- threshold
                        cat(bestMetric, bestLearningRate, bestNumIteration, 
                            bestLambda, bestThreshold, '\n')
                    }
                }
    
    # Train last time with best parameters
    bestModel <- arimo.glm.gd(formula=trainFormula, data=ddfData, 
                              learningRate=bestLearningRate, 
                              numIterations=bestNumIteration,
                              regularized='ridge', lambda=bestLambda, 
                              initialWeights=NULL, ref.levels=NULL)    
    
    attr(bestModel, 'glm.threshold') <- bestThreshold
    return (bestModel)
}
