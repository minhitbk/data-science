"""
@author: Tran Ngoc Minh
"""
from base_scheduler import BaseScheduler, logger
from scheduler.database.db_class import session, AppConfigSchedule
import uuid, json

class FifoScheduler(BaseScheduler):
    """
    Implement a fifo scheduler
    """

    def __init__(self):
        super(FifoScheduler, self).__init__()
        
    def doScheduling(self, appConfig):
        """
        Parse the application configuration to get requested resources
        """
        machines, storages, routers = self.parseAppConfig(appConfig)
        
        """
        Invoke the IaaS service to get available resources that
        satisfy the application configuration
        """

        """
        Process machine node
        """
        machineIDList = []
        machineDesList = []        
        machinePrice = 0
        for i in range(len(machines)):
            machineDescriptor = {"machine":machines[i]}
            try:               
                print "........The fifo scheduler gets available" \
                        " machines for machine description %d from" \
                        " IaaS and does scheduling" %(i+1)                    
                machineIDs = self.getAvailableNodeList(machineDescriptor, i+1)                
                machineID = machineIDs.pop(0)                 
                while machineID in machineIDList:                    
                    machineID = machineIDs.pop(0)                
            except IndexError:                
                result = json.dumps({"result":None})                
                logger.error("Cannot find available machines"
                             " satisfying requirements")                
                return result
                
            machinePrice = machinePrice + self.getNodePrice(machineDescriptor,
                                              machineID, 1)
            machineDesList.append({machineID:machines[i]})
            machineIDList.append(machineID)

        # for storing to database
        machineInfo = {"machine":machineDesList}

        """
        Process storage node
        """
        storageIDList = []
        storageDesList = []               
        storagePrice = 0
        
        for i in range(len(storages)):        
            storageDescriptor = {"storage":storages[i]}            
            try:
                print "........The fifo scheduler gets available" \
                        " storages for storage description %d from" \
                        " IaaS and does scheduling" %(i+1)                                        
                storageIDs = self.getAvailableNodeList(storageDescriptor, i+1)
                storageID = storageIDs.pop(0)                 
                while storageID in storageIDList:                    
                    storageID = storageIDs.pop(0)                            
            except IndexError:                
                result = json.dumps({"result":None})
                logger.error("Cannot find available storages"
                             " satisfying requirements")                                
                return result                
            
            storagePrice = storagePrice + self.getNodePrice(storageDescriptor,
                                              storageID, 1)            
            storageDesList.append({storageID:storages[i]})
            storageIDList.append(storageID)

        # for storing to database
        storageInfo = {"storage":storageDesList}           
        
        """
        Process router node
        """
        routerIDList = []
        routerDesList = []                     
        routerPrice = 0
        
        for i in range(len(routers)):        
            routerDescriptor = {"router":routers[i]}            
            try:
                print "........The fifo scheduler gets available" \
                        " routers for router description %d from" \
                        " IaaS and does scheduling" %(i+1)                                        
                routerIDs = self.getAvailableNodeList(routerDescriptor, i+1)                            
                routerID = routerIDs.pop(0)                 
                while routerID in routerIDList:                    
                    routerID = routerIDs.pop(0)                    
            except IndexError:                
                result = json.dumps({"result":None})
                logger.error("Cannot find available routers"
                             " satisfying requirements")                                                
                return result                                            
            
            routerPrice = routerPrice + self.getNodePrice(routerDescriptor,
                                              routerID, 1)            
            routerDesList.append({routerID:routers[i]})
            routerIDList.append(routerID)

        # for storing to database
        routerInfo = {"router":routerDesList}            

        """
        Store the scheduling result to database
        """
        configID = str(uuid.uuid4())

        # currently we do not process network scheduling
        networkInfo = ""        
        acs = AppConfigSchedule(configID,
                                json.dumps(storageInfo),
                                json.dumps(machineInfo),
                                json.dumps(networkInfo),
                                json.dumps(routerInfo))        
        session.add(acs)
        session.commit()

        """
        Return scheduling result to users to inform them the price
        """        
        totalPrice = machinePrice + storagePrice + routerPrice
        result = json.dumps({"result":
                                    {"schedule":[machineInfo,
                                                 storageInfo,
                                                 routerInfo],
                                     "price":totalPrice,
                                     "configID":configID
                                     }
                             })        
        return result
        
        