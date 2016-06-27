###############################################################################
# (c) Copyright 2016, Arimo, Inc. All Rights Reserved.
# @author: minhtran
#
# This file contains scripts for processing the usecase of predictive 
# maintenance of harddrive failures, which is applied to the blogpost 
# entitled 'Predictive Maintenance with Big Apps'.

library(arimo)

# Function for computing the metric
computeMetric <- function(model, data, threshold) {
    # Predict data with the model
    result <- arimo.predict(model, data)
    
    # Compute the predicted labels
    hiveStr <- paste0('yPredictLabel = if(yPredict > ', threshold, ', 1, 0)')
    finalResult <- eval(call('arimo.transform', result, 'Hive', hiveStr))
    
    conMatrix <- adatao.xtabs(~ ytrue + yPredictLabel, finalResult)
    
    fpr <- conMatrix[1,2] / (conMatrix[1,2] + conMatrix[1,1])
    fnr <- conMatrix[2,1] / (conMatrix[2,1] + conMatrix[2,2])
    
    return (1.25 * fnr * fpr / (0.25 * fnr + fpr))
}
