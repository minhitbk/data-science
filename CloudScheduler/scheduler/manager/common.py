"""
@author: Tran Ngoc Minh
"""

"""This module initializes the communication API between 
application management layer and resource management layer,
and implements common functions used in the manager package
"""

import json
from scheduler.common.loaders import config
from scheduler.common.wrappers import Connection
from scheduler.common import importers

"""
Create connection wrappers to internal schedulers
"""
schedulingHostnames = json.loads(config.get("InternalSchedulers","HOSTNAMES"))
schedulingPorts = json.loads(config.get("InternalSchedulers","PORTS"))
numSchedulers = len(schedulingHostnames)

workerConn = []

for index in range(numSchedulers):
    
    workerConn.append(Connection(schedulingHostnames[index],
                                 schedulingPorts[index]))

"""
Create load balancer
"""
loadBalancingPolicy = config.get("LoadBalancer","LOAD_BALANCING_POLICY")

"""
Use factory design pattern
"""
loadBalancerClassName = config.get("LoadBalancer",
                        ("%s_LOAD_BALANCER" %loadBalancingPolicy.upper()))

LoadBalancerClass = importers.importClass(loadBalancerClassName)

loadBalancer = LoadBalancerClass(numSchedulers)

"""
Create a connection to the IaaS layer
"""
hostname = config.get("IaaS","HOSTNAME")
port = int(config.get("IaaS","PORT"))

iaasConn = Connection(hostname,port)


    