###############################################################################
# (c) Copyright 2016, Arimo, Inc. All Rights Reserved.
# @author: minhtran
#
# This file contains scripts for processing the usecase of predictive 
# maintenance of harddrive failures, which is applied to the blogpost 
# entitled 'Predictive Maintenance with Big Apps'.

library(arimo)

# Function to transform data with new past-future columns 
featureEngineering <- function(ddfData) {
    addDayToDdfData <- arimo.transform(ddfData, method='R',
                                       'datetime0 = as.Date(datetime) + 0',
                                       'datetime1 = as.Date(datetime) + 1',
                                       'datetime_1 = as.Date(datetime) - 1')
    
    joinYesterdayToday <- arimo.join(addDayToDdfData, addDayToDdfData,
                                     by.x=c('serial', 'datetime1'), 
                                     by.y=c('serial', 'datetime0'), 
                                     type='inner')
    
    colnames(addDayToDdfData)[c(1:length(colnames(addDayToDdfData)))] <- 
                            paste('r', colnames(addDayToDdfData), sep='_')
    
    joinYesterdayTodayTomorrow <- arimo.join(joinYesterdayToday, 
                                             addDayToDdfData, 
                                             by.x=c('serial', 'datetime1'), 
                                             by.y=c('r_serial', 'r_datetime_1'), 
                                             type='inner')
    return (joinYesterdayTodayTomorrow)
}
