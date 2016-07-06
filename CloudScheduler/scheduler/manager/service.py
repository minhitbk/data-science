"""
@author: Tran Ngoc Minh
"""

"""This module implements the communication API between 
application management layer and resource management layer"""

from flask import Flask, json, request
from scheduler.common.loaders import logHandler, config, method
from scheduler.database.db_class import session, AppConfigSchedule
from httplib import NotConnected
from scheduler.manager.common import loadBalancer, workerConn, iaasConn

"""
The communication api web service
"""
app = Flask(__name__)

"""
Add logger to the web service
"""
app.logger.addHandler(logHandler)

"""==================================================="""
"""
Implement APIs
"""

@app.before_request
def beforeRequest():
    if request.headers["Content-Type"] != "application/json":
        app.logger.error("Unsupported media type, use application/json")
        return "Unsupported media type, use application/json\n"

######################################################################
"""
    { 
        "Devices" : [{"Num" : 2, "Attributes" : {"Type" : "CPU", "Cores" : 2}}, 
                        {"Num" : 1, "Attributes" : {"Type" : "CPU", "Cores" : 1}}],
        "Connections" : [[0,1,200], [1,0,100]] 
    }
    
    |||
    
    { 
        "storage":[],
    
        "machine":[{"Cores": 3},{"Cores":4,"RAM":1000}],
                   
        "router":[]             
    }
"""
######################################################################

@app.route("/prepareReservation", methods=[method])
def prepareReservation():
        print "....The scheduling manager received request for scheduling"   
        data = request.get_json()
        machine_list = []
        
        for mset in data["Devices"]:
            for m in range(int(mset["Num"])):
                new_dict = {}
                for key in  mset["Attributes"]:
                    if key !="Type":
                        new_dict[key] = mset["Attributes"][key]
                machine_list.append(new_dict)

        data = { 
                    "storage":[],
                    "machine": machine_list,                   
                    "router":[]             
                }       
        
        result = None
        
        while result == None:        
            print "....The scheduling manager asks the configured load balancer for the internal" \
                    " scheduler to serve the request"
                    
            schedulerIndex = loadBalancer.doLoadBalancing()
            
            print "........The load balancer selects the internal scheduler %d" \
                " to serve the request" % (schedulerIndex+1) 
            
            """
            Catch exceptions when schedulerIndex is out of range or the 
            corresponding scheduler died, or the connection lost,
            then go to another scheduler
            """
            try:                
                print "....The scheduling manager invokes the selected internal scheduler for" \
                        " scheduling "
                
                result = workerConn[schedulerIndex].request(
                                            method,"/doScheduling",
                                            json.dumps(data))
            except IndexError:
                app.logger.error("Scheduler index is out of range")
            except NotConnected:
                app.logger.error("Cannot make http request to the"
                                 " selected scheduler")
        return result
    
@app.route("/getStaticResourceInfo", methods=[method])
def getStaticResourceInfo():
    """
    Get identities of compute and storage nodes 
    like machine, router, storage
    """
    
    nodeList = resultIDList = []
    checkList = ["machine","storage","router"]
    nodeList.append("/")
    
    while len(list) > 0:
        nodeID = nodeList.pop(1)
        nodeType = nodeID.split(":")[0].split("/")[-1]
        if nodeType in checkList:
            resultIDList.append(nodeID)
        else:
            """
            Catch exception when cannot make connection to the IaaS service"
            """     
            try:
                childNodeList = iaasConn.request(method,"/getNodeList",
                                        json.dumps({"nodeID":nodeID}))            
                nodeList.extend(childNodeList["result"])                            
            except NotConnected:                
                app.logger.error("Cannot make http request to"
                                 " the IaaS service")                  
                result = json.dumps({"result":None})                
                return result           
    
    """
    Get static information of these nodes
    """
    resultInfoList = []
    for nodeID in resultIDList: 
        """
        Catch exception when cannot make connection to the IaaS service"
        """     
        try:
            resultInfo = iaasConn.request(method,"/getStaticNodeInfo",
                                    json.dumps({"nodeID":nodeID}))            
            resultInfoList.append(resultInfo)            
        except NotConnected:
            app.logger.error("Cannot make http request to"
                                 " the IaaS service")                              
            result = json.dumps({"result":None})            
            return result                    
    result = json.dumps({"result":resultInfoList})
    return result

@app.route("/getMonitoringInfo", methods=[method])
def getMonitoringInfo():
    """
    Catch exception when cannot make connection to the IaaS service"
    """    
    try:
        result = iaasConn.request(method,"/getNodeMonitoringInfo",
                                  json.dumps(request.json))
    except NotConnected:
        app.logger.error("Cannot make http request to the IaaS service")        
        result = json.dumps({"result":None})
    return result    
    
@app.route("/createReservation", methods=[method])
def createReservation():
    """
    Catch exception when input request format is wrong
    """
    try:
        configID = request.json["configID"]
    except ValueError:
        app.logger.error("Input request format is wrong, "
                         "no configID found to create reservation")        
        result = json.dumps({"result":None})
        return result
    
    """
    Access database to get reservation description
    """
    reservInfo = session.query(AppConfigSchedule).get(configID)

    """
    If there is not a reservation available yet
    """
    if not reservInfo:
        result = json.dumps({"result":None})
        return result
    else:        
        storageInfo = json.loads(reservInfo.storageInfo)["storage"]
        machineInfo = json.loads(reservInfo.machineInfo)["machine"]
        networkInfo = ""
        routerInfo = json.loads(reservInfo.routerInfo)["router"]                        
        
        reservDescriptor = {"reservDescriptor":{"storage":storageInfo,
                                                "machine":machineInfo,
                                                "network":networkInfo,
                                                "router":routerInfo}}

    """
    Connect to the IaaS layer for creating reservation on resources,
    catch exception when cannot make connection to the IaaS service
    """          
    try:
        result = iaasConn.request(method,"/createReservation",
                                  json.dumps(reservDescriptor))        
    except NotConnected:        
        app.logger.error("Cannot make http request to the IaaS service")            
        result = json.dumps({"result":None})
    return result    

@app.route("/shutdown", methods=[method])
def shutdown():    
    func = request.environ.get("werkzeug.server.shutdown")
    if func is None:    
        raise RuntimeError("Not running with the Werkzeug Server")
    
    func()
    return "scheduling manager shutting down...\n"
    
"""==================================================="""                        
"""
Start the communication api web service
"""
if __name__=="__main__": 
    hostname = config.get("Scheduler", "HOSTNAME")
    port = int(config.get("Scheduler", "PORT")) 
    app.run(host=hostname, port=port, debug=False)

