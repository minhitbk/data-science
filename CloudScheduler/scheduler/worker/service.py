"""
@author: Tran Ngoc Minh
"""

"""This module implements the communication API between 
the manager and the internal schedulers"""

from flask import Flask, request
from scheduler.common.loaders import *
#logHandler, config, method
from scheduler.common import importers

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

@app.route("/doScheduling",methods=[method])
def doScheduling():

    """
    Use factory design pattern to dynamically get a proper scheduler
    """
    schedulerClassName = config.get("InternalSchedulers",
                                ("%s_SCHEDULER" %app.config["POLICY"].upper()))
    
    SchedulerClass = importers.importClass(schedulerClassName)
    
    scheduler = SchedulerClass()

    result = scheduler.doScheduling(request.json)

    return result

@app.route("/shutdown", methods=[method])
def shutdown():
    
    func = request.environ.get("werkzeug.server.shutdown")
    
    if func is None:
        
        raise RuntimeError("Not running with the Werkzeug Server")
    
    func()
    
    return ("%s_scheduler shutting down...\n" %app.config["POLICY"])
