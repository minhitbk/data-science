"""
@author: Tran Ngoc Minh
"""

import os
from ConfigParser import ConfigParser
from logging.handlers import RotatingFileHandler
from logging import Formatter

home = os.path.expanduser("~")
"""
Search for a config file from the environment
""" 
CONF_FILE = os.getenv("SCHEDULER_CFG")

"""
If cannot find the config file from the environment
"""
if not CONF_FILE:
    CONF_FILE = os.path.join(home, ".harness-crs", "scheduler.conf")

"""
Load the config file
"""
config = ConfigParser()
config.read(CONF_FILE)

"""
Open log file
"""
logFile = config.get("Logging","LOG_FILE")

logHandler = RotatingFileHandler(logFile,maxBytes=5000,backupCount=5)
logHandler.setFormatter(Formatter(
    "%(asctime)s %(levelname)s: %(message)s "
    "[in %(pathname)s:%(lineno)d]"
))

"""
Method of http connections
"""
method = "POST"




