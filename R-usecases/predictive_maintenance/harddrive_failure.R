###############################################################################
# (c) Copyright 2016, Arimo, Inc. All Rights Reserved.
# @author: minhtran
#
# This file contains scripts for processing the usecase of predictive 
# maintenance of harddrive failures, which is applied to the blogpost 
# entitled 'Predictive Maintenance with Big Apps'.

library(arimo)
source('feature_engineer.R')
source('train_model.R')

# Connect to arimo server
arimo.connect('172.28.10.200', 16000, 'minhtran@arimo.com', 'abcde')

# Load the original data set
hddDdf <- arimo.getDDF('ddf://adatao/hdd')

# Filter necessary columns/rows
filterDdf <- hddDdf[hddDdf$column2 == 'Hitachi HDS5C3030ALA630', c('column0', 
                        'column1', 'column2', 'column3', 'column4', 'column6',
                        'column14', 'column20', 'column50','column56')]
colnames(filterDdf) <- c('datetime', 'serial', 'modelname', 'capacity', 
                         'failure', 'error', 'reallocate', 'power',
                         'temperature', 'pending')

# Create the final transformed DDF
finalDdf <- featureEngineering(filterDdf)

# Store the finalDdf on PE
arimo.setDDFName(finalDdf, 'finalDdf')

# Create the final train/test data sets
trainDdf <- arimo.sql2ddf("select * from finalDdf where r_datetime < '2014-09-01'")
testDdf <- arimo.sql2ddf("select * from finalDdf where r_datetime >= '2014-09-01'")

# Up-sample the trainDdf to balance the classes
localTrainDdf <- head(trainDdf[trainDdf$r_r_failure == 1, ], 
                      nrow(trainDdf[trainDdf$r_r_failure == 1, ]))

upTime <- round(log2(nrow(trainDdf) / nrow(localTrainDdf))/1.3)
for (i in (1:upTime)) {
    localTrainDdf <- rbind(localTrainDdf, localTrainDdf)
}

tmpTrainDdf <- arimo.df2ddf(localTrainDdf)
tmpTrainDdf <- arimo.sql2ddf("select * from {1} where r_reallocate + r_pending > 12", 
                             ddfList = list(tmpTrainDdf))
finalTrainDdf <- arimo.rbind(trainDdf, tmpTrainDdf)

# Train the model
trainFormula <- formula('r_r_failure ~ r_temperature + r_reallocate + r_pending')
model <- trainBestModel(finalTrainDdf, trainFormula)

# Test the model
testResult <- arimo.predict(model, testDdf)
hiveStr <- paste0('if(yPredict > ', attr(model, 'glm.threshold'), ', 1, 0)')
finalTestResult <- eval(call('arimo.transform', testResult, 'SQL', 
                             yPredictLabel=hiveStr))
print(arimo.xtabs(~ ytrue + yPredictLabel, finalTestResult))
