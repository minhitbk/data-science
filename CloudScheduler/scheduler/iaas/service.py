"""
@author: Tran Ngoc Minh
"""

"""
This module implements a faked IaaS API
"""

from flask import Flask, json, request
from scheduler.common.loaders import logHandler, config, method
import cloud_infra, uuid
from random import randrange
from scheduler.database.db_class import StorageType, StorageResource
from scheduler.database.db_class import RouterType, RouterResource
from scheduler.database.db_class import MachineType, MachineResource
from scheduler.database.db_class import session, Base, dbEngine

"""
The faked api web service
"""
app = Flask(__name__)

"""
Add logger to the web service
"""
app.logger.addHandler(logHandler)

"""
Remove drop_all function when not using for testing
"""
Base.metadata.drop_all(dbEngine)
Base.metadata.create_all(dbEngine)

"""==================================================="""
"""
Implement node APIs
"""

"""
Load faked cloud information and store on database, this would be
replaced soon, just created for demo
"""
@app.before_first_request
def beforeFirstRequest():

    session.add(MachineType("MACHINE_TYPE_1",
                               cloud_infra.MACHINE_TYPE_1["numcpu"],
                               cloud_infra.MACHINE_TYPE_1["numcore"],
                               cloud_infra.MACHINE_TYPE_1["frequency"],
                               cloud_infra.MACHINE_TYPE_1["numfpga"],
                               cloud_infra.MACHINE_TYPE_1["memfpga"],
                               cloud_infra.MACHINE_TYPE_1["numgpu"],
                               cloud_infra.MACHINE_TYPE_1["memory"],
                               cloud_infra.MACHINE_TYPE_1["disk"],
                               cloud_infra.MACHINE_TYPE_1["typegpu"],
                               cloud_infra.MACHINE_TYPE_1["typefpga"]))

    for i in range(cloud_infra.MACHINE_TYPE_1["quantity"]):
        session.add(MachineResource("/dc1/rack1/machine%s/machine:" %i,
                                          "MACHINE_TYPE_1"))
        
    session.add(MachineType("MACHINE_TYPE_2",
                               cloud_infra.MACHINE_TYPE_2["numcpu"],
                               cloud_infra.MACHINE_TYPE_2["numcore"],
                               cloud_infra.MACHINE_TYPE_2["frequency"],
                               cloud_infra.MACHINE_TYPE_2["numfpga"],
                               cloud_infra.MACHINE_TYPE_2["memfpga"],
                               cloud_infra.MACHINE_TYPE_2["numgpu"],
                               cloud_infra.MACHINE_TYPE_2["memory"],
                               cloud_infra.MACHINE_TYPE_2["disk"],
                               cloud_infra.MACHINE_TYPE_2["typegpu"],
                               cloud_infra.MACHINE_TYPE_2["typefpga"]))

    for i in range(cloud_infra.MACHINE_TYPE_2["quantity"]):
        session.add(MachineResource("/dc2/rack1/machine%s/machine:" %i,
                                          "MACHINE_TYPE_2"))
        
    session.add(MachineType("MACHINE_TYPE_3",
                               cloud_infra.MACHINE_TYPE_3["numcpu"],
                               cloud_infra.MACHINE_TYPE_3["numcore"],
                               cloud_infra.MACHINE_TYPE_3["frequency"],
                               cloud_infra.MACHINE_TYPE_3["numfpga"],
                               cloud_infra.MACHINE_TYPE_3["memfpga"],
                               cloud_infra.MACHINE_TYPE_3["numgpu"],
                               cloud_infra.MACHINE_TYPE_3["memory"],
                               cloud_infra.MACHINE_TYPE_3["disk"],
                               cloud_infra.MACHINE_TYPE_3["typegpu"],
                               cloud_infra.MACHINE_TYPE_3["typefpga"]))    
 
    for i in range(cloud_infra.MACHINE_TYPE_3["quantity"]):
        session.add(MachineResource("/dc2/rack2/machine%s/machine:" %i,
                                          "MACHINE_TYPE_3"))    

    session.add(RouterType("ROUTER_TYPE_1",
                            cloud_infra.ROUTER_TYPE_1["inbw"],
                            cloud_infra.ROUTER_TYPE_1["outbw"],
                            cloud_infra.ROUTER_TYPE_1["buffer"],
                            cloud_infra.ROUTER_TYPE_1["capacity"],
                            cloud_infra.ROUTER_TYPE_1["connector"],
                            cloud_infra.ROUTER_TYPE_1["version"]))
    
    for i in range(cloud_infra.ROUTER_TYPE_1["quantity"]):
        session.add(RouterResource("/dc1/rack1/router%s/router:" %i,
                                          "ROUTER_TYPE_1"))
        
    session.add(RouterType("ROUTER_TYPE_2",
                            cloud_infra.ROUTER_TYPE_2["inbw"],
                            cloud_infra.ROUTER_TYPE_2["outbw"],
                            cloud_infra.ROUTER_TYPE_2["buffer"],
                            cloud_infra.ROUTER_TYPE_2["capacity"],
                            cloud_infra.ROUTER_TYPE_2["connector"],
                            cloud_infra.ROUTER_TYPE_2["version"]))
    
    for i in range(cloud_infra.ROUTER_TYPE_2["quantity"]):
        session.add(RouterResource("/dc2/rack1/router%s/router:" %i,
                                          "ROUTER_TYPE_2"))

    session.add(StorageType("STORAGE_TYPE_1",
                            cloud_infra.STORAGE_TYPE_1["capacity"],
                            cloud_infra.STORAGE_TYPE_1["ranbw"],
                            cloud_infra.STORAGE_TYPE_1["seqbw"]))
    
    for i in range(cloud_infra.STORAGE_TYPE_1["quantity"]):
        session.add(StorageResource("/dc1/rack1/storage%s/storage:" %i,
                                          "STORAGE_TYPE_1"))

    session.add(StorageType("STORAGE_TYPE_2",
                            cloud_infra.STORAGE_TYPE_2["capacity"],
                            cloud_infra.STORAGE_TYPE_2["ranbw"],
                            cloud_infra.STORAGE_TYPE_2["seqbw"]))
    
    for i in range(cloud_infra.STORAGE_TYPE_2["quantity"]):
        session.add(StorageResource("/dc2/rack1/storage%s/storage:" %i,
                                          "STORAGE_TYPE_2"))
  
    session.commit()
    
@app.before_request
def beforeRequest():
    
    if request.headers["Content-Type"] != "application/json":
    
        app.logger.error("Unsupported media type, use application/json")

        return "Unsupported media type, use application/json\n" 

@app.route("/getStaticNodeInfo",methods=[method])
def getStaticNodeInfo():

    nodeID = request.json["nodeID"]
    nodeType = nodeID.split(":")[0].split("/")[-1]
        
    if nodeType == "machine":
        
        try:
            typeID = session.query(MachineResource).get(nodeID).typeID
            
            nodeInfo = session.query(MachineType).get(typeID)
            
            result = {"nodeID":nodeID,
                      "numcpu":nodeInfo.numcpu,
                      "numcore":nodeInfo.numcore,
                      "frequency":nodeInfo.frequency,
                      "numfpga":nodeInfo.numfpga,
                      "memfpga":nodeInfo.memfpga,
                      "typefpga":nodeInfo.typefpga,
                      "numgpu":nodeInfo.numgpu,
                      "typegpu":nodeInfo.typegpu,
                      "memory":nodeInfo.memory,
                      "disk":nodeInfo.disk}
        
        except AttributeError:
            
            app.logger.error("Machine %s not found" %nodeID)
            
            result = None
            
    elif nodeType == "router":

        try:
            typeID = session.query(RouterResource).get(nodeID).typeID
        
            nodeInfo = session.query(RouterType).get(typeID)
        
            result = {"nodeID":nodeID,
                      "inbw":nodeInfo.inbw,
                      "outbw":nodeInfo.outbw,
                      "buffer":nodeInfo.buffer,
                      "capacity":nodeInfo.capacity,
                      "connector":nodeInfo.connector,
                      "version":nodeInfo.version}
        except AttributeError:
            
            app.logger.error("Router %s not found" %nodeID)
            
            result = None
                        
    elif nodeType == "storage":

        try:
            typeID = session.query(StorageResource).get(nodeID).typeID
            
            nodeInfo = session.query(StorageType).get(typeID)
            
            result = {"nodeID":nodeID,
                      "capacity":nodeInfo.capacity,
                      "ranbw":nodeInfo.ranbw,
                      "seqbw":nodeInfo.seqbw}
        
        except AttributeError:
            
            app.logger.error("Storage %s not found" %nodeID)
            
            result = None            
    
    else:
        
        result = None 
    
    result = json.dumps({"result":result})
            
    return result
 
@app.route("/getDynamicNodeInfo",methods=[method])
def getDynamicNodeInfo():

    nodeID = request.json["nodeID"]
    nodeType = nodeID.split(":")[0].split("/")[-1]
        
    if nodeType == "machine":
        
        try:
            
            status = session.query(MachineResource).get(nodeID).status
                   
            result = {"nodeID":nodeID, "status":status}
            
        except AttributeError:
            
            app.logger.error("Machine %s not found" %nodeID)
            
            result = None
        
    elif nodeType == "router":
        
        try:
            
            status = session.query(RouterResource).get(nodeID).status
                   
            result = {"nodeID":nodeID, "status":status}
            
        except AttributeError:
            
            app.logger.error("Router %s not found" %nodeID)
            
            result = None
        
    elif nodeType == "storage":

        try:
            
            status = session.query(StorageResource).get(nodeID).status
                   
            result = {"nodeID":nodeID, "status":status}
            
        except AttributeError:
            
            app.logger.error("Storage %s not found" %nodeID)
            
            result = None
            
    else:
        
        result = None 
    
    result = json.dumps({"result":result})
            
    return result
 
@app.route("/isNodeRunning",methods=[method])
def isNodeRunning():

    nodeID = request.json["nodeID"]
    nodeType = nodeID.split(":")[0].split("/")[-1]
    
    result = json.dumps({"result":(nodeType in ["machine", "router", "storage"])})    
        
    return result
 
@app.route("/getAvailableNodeList",methods=[method])
def getAvailableNodeList():

    nodeDescriptor = request.json["nodeDescriptor"]
    numNode = request.json["numNode"]
    nodeType = json.loads(json.dumps(nodeDescriptor)).keys()[0]

    nodeIDs = []
    
    if nodeType == "machine":
        
        try:
            
            nodeDescriptor = nodeDescriptor["machine"]
            listType = []
            
            for instance in session.query(MachineType):
 
                satisfy = (instance.numcpu >= nodeDescriptor["numcpu"] and
                        instance.numcore >= nodeDescriptor["numcore"] and
                        instance.frequency >= nodeDescriptor["frequency"] and
                        instance.numfpga >= nodeDescriptor["numfpga"] and
                        instance.memfpga >= nodeDescriptor["memfpga"] and
                        instance.numgpu >= nodeDescriptor["numgpu"] and
                        instance.typegpu == nodeDescriptor["typegpu"] and
                        instance.memory >= nodeDescriptor["memory"] and
                        instance.disk >= nodeDescriptor["disk"] and
                        instance.typefpga == nodeDescriptor["typefpga"])
                
                if satisfy:
                    
                    listType.append(instance.typeID)
                    
            for i in range(len(listType)):
                for resource in session.query(MachineResource).\
                                filter_by(typeID=listType[i]).\
                                filter_by(status="free"):

                    nodeIDs.append(resource.resourceID)
       
        except AttributeError:
            
            app.logger.error("Machine type not found")
                 
    elif nodeType == "router":
        
        try:

            nodeDescriptor = nodeDescriptor["router"]
            listType = []
            
            for instance in session.query(RouterType):
                
                satisfy = (instance.inbw >= nodeDescriptor["inbw"] and
                        instance.outbw >= nodeDescriptor["outbw"] and
                        instance.buffer >= nodeDescriptor["buffer"] and
                        instance.capacity >= nodeDescriptor["capacity"] and
                        instance.connector == nodeDescriptor["connector"] and
                        instance.version == nodeDescriptor["version"])
            
                if satisfy:
                    
                    listType.append(instance.typeID)
                    
            for i in range(len(listType)):
                for resource in session.query(RouterResource).\
                                filter_by(typeID=listType[i]).\
                                filter_by(status="free"):

                    nodeIDs.append(resource.resourceID)
       
        except AttributeError:
            
            app.logger.error("Router type not found")
        
    elif nodeType == "storage":

        try:

            nodeDescriptor = nodeDescriptor["storage"]
            listType = []
            
            for instance in session.query(StorageType):
                
                satisfy = (instance.capacity >= nodeDescriptor["capacity"] and
                        instance.ranbw >= nodeDescriptor["ranbw"] and
                        instance.seqbw >= nodeDescriptor["seqbw"])
            
                if satisfy:
                    
                    listType.append(instance.typeID)
                    
            for i in range(len(listType)):
                for resource in session.query(StorageResource).\
                                filter_by(typeID=listType[i]).\
                                filter_by(status="free"):

                    nodeIDs.append(resource.resourceID)
       
        except AttributeError:
            
            app.logger.error("Storage type not found")

    result = json.dumps({"result":nodeIDs[0:min(numNode,len(nodeIDs))]})

    return result
    
@app.route("/checkStaticNodeInfo",methods=[method])
def checkStaticNodeInfo():

    nodeInfo = json.loads(getStaticNodeInfo())["result"]
    nodeDescriptor = request.json["nodeDescriptor"]
    nodeType = json.loads(json.dumps(nodeDescriptor)).keys()[0]
    satisfy = False

    if nodeType == "machine":
        
        nodeDescriptor = nodeDescriptor["machine"]

        satisfy = (nodeInfo["numcpu"] >= nodeDescriptor["numcpu"] and
                    nodeInfo["numcore"] >= nodeDescriptor["numcore"] and
                    nodeInfo["frequency"] >= nodeDescriptor["frequency"] and
                    nodeInfo["numfpga"] >= nodeDescriptor["numfpga"] and
                    nodeInfo["memfpga"] >= nodeDescriptor["memfpga"] and
                    nodeInfo["numgpu"] >= nodeDescriptor["numgpu"] and
                    nodeInfo["typegpu"] == nodeDescriptor["typegpu"] and
                    nodeInfo["memory"] >= nodeDescriptor["memory"] and
                    nodeInfo["disk"] >= nodeDescriptor["disk"] and
                    nodeInfo["typefpga"] == nodeDescriptor["typefpga"])
                             
    elif nodeType == "router":
        
        nodeDescriptor = nodeDescriptor["router"]
        
        satisfy = (nodeInfo["inbw"] >= nodeDescriptor["inbw"] and
                nodeInfo["outbw"] >= nodeDescriptor["outbw"] and
                nodeInfo["buffer"] >= nodeDescriptor["buffer"] and
                nodeInfo["capacity"] >= nodeDescriptor["capacity"] and
                nodeInfo["connector"] == nodeDescriptor["connector"] and
                nodeInfo["version"] == nodeDescriptor["version"])
                   
    elif nodeType == "storage":
        
        nodeDescriptor = nodeDescriptor["storage"]
        
        satisfy = (nodeInfo["capacity"] >= nodeDescriptor["capacity"] and
                nodeInfo["ranbw"] >= nodeDescriptor["ranbw"] and
                nodeInfo["seqbw"] >= nodeDescriptor["seqbw"])
            
    result = json.dumps({"result":satisfy})

    return result

@app.route("/checkDynamicNodeInfo",methods=[method])
def checkDynamicNodeInfo():

    nodeInfo = json.loads(getDynamicNodeInfo())["result"]
    
    result = json.dumps({"result":False})
    
    if nodeInfo["status"] == "free":
        
        result = checkStaticNodeInfo()

    return result
        
@app.route("/createReservation",methods=[method])
def createReservation():

    reservID = str(uuid.uuid4())
    machineList = request.json["reservDescriptor"]["machine"]
    storageList = request.json["reservDescriptor"]["storage"]
    routerList = request.json["reservDescriptor"]["router"]        

    try:    
        for nodes in machineList:
            
            nodeID = json.loads(json.dumps(nodes)).keys()[0]
            
            session.query(MachineResource).filter_by(resourceID=nodeID).\
                    update({"reservID":reservID, "status":"reserved"})
            
        for nodes in storageList:
            
            nodeID = json.loads(json.dumps(nodes)).keys()[0]
            session.query(StorageResource).filter_by(resourceID=nodeID).\
                    update({"reservID":reservID, "status":"reserved"})   
                    
        for nodes in routerList:
            
            nodeID = json.loads(json.dumps(nodes)).keys()[0]
            session.query(RouterResource).filter_by(resourceID=nodeID).\
                    update({"reservID":reservID, "status":"reserved"})                
    
    except AttributeError:
            
        app.logger.error("Resource not found")
        
        reservID = "RESOURCE_NOT_FOUND"
                     
    result = json.dumps({"result":reservID})

    return result

@app.route("/cancelReservation",methods=[method])
def cancelReservation():
    
    session.query(MachineResource).filter_by(reservID = request.json["reservID"]).\
                update({"status":"free","reservID":""})
                
    session.query(StorageResource).filter_by(reservID = request.json["reservID"]).\
                update({"status":"free","reservID":""})

    session.query(RouterResource).filter_by(reservID = request.json["reservID"]).\
                update({"status":"free","reservID":""})    

    result = json.dumps({"result":True})

    return result

"""
Temporarily working as same as cancelReservation
"""
@app.route("/releaseReservation",methods=[method])
def releaseReservation():
    
    result = cancelReservation()

    return result

@app.route("/startVMs",methods=[method])
def startVMs():
    
    nodeIDs = request.json["nodeIDs"]
    
    hosts = []
    for index in range[len(nodeIDs)]:
        hosts = hosts + ["localhost"]
    
    result = json.dumps({"result":hosts})

    return result

@app.route("/shutdownVMs",methods=[method])
def shutdownVMs():

    result = json.dumps({"result":True})

    return result

@app.route("/deployCodeOnRouter",methods=[method])
def deployCodeOnRouter():

    result = json.dumps({"result":"deployID"})

    return result
    
@app.route("/createVolume",methods=[method])
def createVolume():

    result = json.dumps({"result":"volID"})

    return result

@app.route("/getNodeMonitoringInfo",methods=[method])
def getNodeMonitoringInfo():

    result = json.dumps({"result":"MonitoringInfo"})

    return result

@app.route("/getNodePrice",methods=[method])
def getNodePrice():

    result = json.dumps({"result":randrange(10,100)})

    return result

"""==================================================="""
"""
Implement network APIs
"""

@app.route("/getNodeList",methods=[method])
def getNodeList():
    
    result = json.dumps({"result":None})

    return result

@app.route("/getExpectedAvailableBW",methods=[method])
def getExpectedAvailableBW():

    result = json.dumps({"result":randrange(0,100)})

    return result

@app.route("/getGuaranteedAvailableBW",methods=[method])
def getGuaranteedAvailableBW():

    result = json.dumps({"result":randrange(0,100)})

    return result

@app.route("/getUpperboundAvailableBW",methods=[method])
def getUpperboundAvailableBW():

    result = json.dumps({"result":randrange(0,100)})

    return result

@app.route("/getAvailableInternalBW",methods=[method])
def getAvailableInternalBW():

    result = json.dumps({"result":None})

    return result

@app.route("/getExpectedLatency",methods=[method])
def getExpectedLatency():

    result = json.dumps({"result":randrange(0,100)})

    return result

@app.route("/getGuaranteedLatency",methods=[method])
def getGuaranteedLatency():

    result = json.dumps({"result":randrange(0,100)})

    return result

@app.route("/getLowerboundLatency",methods=[method])
def getLowerboundLatency():

    result = json.dumps({"result":randrange(0,100)})

    return result

@app.route("/getInternalLatency",methods=[method])
def getInternalLatency():

    result = json.dumps({"result":None})

    return result

@app.route("/getNetworkPrice",methods=[method])
def getNetworkPrice():

    result = json.dumps({"result":randrange(10,100)})

    return result

@app.route("/shutdown", methods=[method])
def shutdown():
    
    func = request.environ.get("werkzeug.server.shutdown")

    if func is None:

        raise RuntimeError("Not running with the Werkzeug Server")

    func()
    
    return "IaaS service shutting down...\n"

"""==================================================="""                        
"""
Start the faked api web service
"""

if __name__=="__main__": 

    hostname = config.get("IaaS","HOSTNAME")
    port = int(config.get("IaaS","PORT"))
    
    app.run(host=hostname,port=port,debug=False)
    
    