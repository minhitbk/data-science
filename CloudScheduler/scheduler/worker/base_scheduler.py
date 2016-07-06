"""
@author: Tran Ngoc Minh
"""

from scheduler.common.wrappers import Connection
from scheduler.common.loaders import logHandler, config, method
import logging, json, sys
from httplib import NotConnected

logger = logging.getLogger()
logger.addHandler(logHandler)

"""
This class will be inherited by scheduler classes.
This is where scheduling algorithms will be implemented
and this forms the second research question. Therefore, inheritance 
of this class is open.
"""
class BaseScheduler(object):
    """
    The base class for a scheduler
    """

    def __init__(self):
        hostname = config.get("IaaS", "HOSTNAME")
        port = int(config.get("IaaS", "PORT"))
        self.conn = Connection(hostname, port)
        
    def __del__(self):
        self.conn.close()
        
    def doScheduling(self, appConfig):
        return None
   
    """
    This function will parse the requested configuration of an 
    application into separate components like machines, storages, 
    routers.
    
    This is where we need to re-implement when the format of an 
    application manifest is finished and when we have an agreement 
    on the application configuration. This will form the third 
    research question.
    
    Currently it is assumed that the application configuration 
    resulted by the performance model has the same format that is
    described in the Harness API document.
    """
    def parseAppConfig(self, appConfig):
        machines = appConfig["machine"]
        storages = appConfig["storage"]
        routers  = appConfig["router"]
        return (machines,storages,routers) 
            
    def getAvailableNodeList(self, nodeDescriptor, numNode):
        params = {"nodeDescriptor":nodeDescriptor,"numNode":numNode}

        """
        Catch exception when cannot make connection to the IaaS service"
        """                      
        try:
            nodes = self.conn.request(method, "/getAvailableNodeList",
                                       json.dumps(params))            
            result = json.loads(nodes)["result"]            
        except NotConnected:            
            logger.error("Cannot make http request to"
                                 " the IaaS service")            
            result = []        
        return result
    
    def getNodePrice(self, nodeDescriptor, nodeID, duration):
        params = {"nodeDescriptor":nodeDescriptor,
                  "nodeID":nodeID,
                  "duration":duration}

        """
        Catch exception when cannot make connection to the IaaS service"
        """                      
        try:
            price = self.conn.request(method, "/getNodePrice",
                                      json.dumps(params))
            result = float(json.loads(price)["result"])
        except NotConnected:            
            logger.error("Cannot make http request to"
                                 " the IaaS service")            
            result = sys.maxint          
        return result        
    