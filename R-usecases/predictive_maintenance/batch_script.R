library(arimo)

# Connect to arimo server
arimo.connect('172.28.10.xxx', 16000, 'minhtran@arimo.com', 'abcde')

# Load the original data set
hddDdf <- arimo.getDDF('ddf://adatao/hdd')

# Filter necessary columns/rows
filterDdf <- hddDdf[hddDdf$column2 == 'Hitachi HDS5C3030ALA630', 
                    c('column0', 'column1', 'column2', 'column3', 'column4', 
                      'column6', 'column14', 'column20', 'column50','column56')]
colnames(filterDdf) <- c('datetime', 'serial', 'modelname', 'capacity', 
                         'failure', 'error', 'reallocate', 'power',
                         'temparature', 'pending')

# Create the final transformed DDF
addDayToDdfData <- arimo.transform(filterDdf, method='R',
                                   'datetime0 = as.Date(datetime) + 0',
                                   'datetime1 = as.Date(datetime) + 1',
                                   'datetime_1 = as.Date(datetime) - 1')

joinYesterdayToday <- arimo.join(addDayToDdfData, addDayToDdfData,
                                 by.x=c('serial', 'datetime1'), 
                                 by.y=c('serial', 'datetime0'), 
                                 type='inner')

colnames(addDayToDdfData)[c(1:length(colnames(addDayToDdfData)))] <- paste('r', 
                                            colnames(addDayToDdfData), sep='_')

finalDdf <- arimo.join(joinYesterdayToday, addDayToDdfData, 
                                         by.x=c('serial', 'datetime1'), 
                                         by.y=c('r_serial', 'r_datetime_1'), 
                                         type='inner')

arimo.setDDFName(finalDdf, 'finalDdf')

# Create the final train/test data sets
trainDdf <- arimo.sql2ddf("select * from finalDdf where r_datetime < '2014-09-01'")
testDdf <- arimo.sql2ddf("select * from finalDdf where r_datetime >= '2014-09-01'")

upSampleTrainDdf <- trainDdf[trainDdf$r_r_failure == 1, ]
upSampleTrainDdf <- arimo.sql2ddf("select * from {1} where r_reallocate + r_pending > 12", 
                                  ddfList = list(upSampleTrainDdf))
upSampleTrainDdf <- arimo.sample2ddf(upSampleTrainDdf, 40000, replace=TRUE)

finalTrainDdf <- arimo.rbind(trainDdf, upSampleTrainDdf)

# Train the model
trainFormula <- formula('r_r_failure ~ r_temparature + r_reallocate + r_pending')
model <- arimo.glm.gd(formula=trainFormula, data=finalTrainDdf, 
                       learningRate=0.001, numIterations=85,
                       regularized="ridge", lambda=0.02789616, 
                       initialWeights=NULL, ref.levels=NULL)

# Test the model
testResult <- arimo.predict(model, testDdf)

# This command is for printing the probability
head(testResult[testResult$ytrue == 1, ], 20)

# This command is for printing the confusion matrix
hiveStr <- paste0("if(yPredict > ", 0.999, ", 1, 0)")
finalTestResult <- eval(call("arimo.transform", testResult, 'SQL', 
                             yPredictLabel = hiveStr))
print(arimo.xtabs(~ ytrue + yPredictLabel, finalTestResult))

# Get the finalDdf that was already pre-processed
finalDdf <- arimo.getDDF('ddf://adatao/finalDdf')

# Create the final train/test data sets
trainDdf <- arimo.sql2ddf("select * from finalDdf where r_datetime < '2014-09-01'")
testDdf <- arimo.sql2ddf("select * from finalDdf where r_datetime >= '2014-09-01'")

upSampleTrainDdf <- trainDdf[trainDdf$r_r_failure == 1, ]
upSampleTrainDdf <- arimo.sql2ddf("select * from {1} where r_reallocate + r_pending > 12", 
                                  ddfList = list(upSampleTrainDdf))
upSampleTrainDdf <- arimo.sample2ddf(upSampleTrainDdf, 40000, replace=TRUE)

finalTrainDdf <- arimo.rbind(trainDdf, upSampleTrainDdf)

# Train the model
trainFormula <- formula('r_r_failure ~ r_temparature + r_reallocate + r_pending')
model <- arimo.glm.gd(formula=trainFormula, data=finalTrainDdf, 
                      learningRate=0.001, numIterations=85,
                      regularized="ridge", lambda=0.02789616, 
                      initialWeights=NULL, ref.levels=NULL)

# Test the model
testResult <- arimo.predict(model, testDdf)

# This command is for printing the probability
print(head(testResult[testResult$ytrue == 1, ], 20))

# This command is for printing the confusion matrix
hiveStr <- paste0("if(yPredict > ", 0.99, ", 1, 0)")
finalTestResult <- eval(call("arimo.transform", testResult, 'SQL', 
                             yPredictLabel = hiveStr))
print(arimo.xtabs(~ ytrue + yPredictLabel, finalTestResult))

