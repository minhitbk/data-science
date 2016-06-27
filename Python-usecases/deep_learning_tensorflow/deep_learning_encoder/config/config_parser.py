'''
Created on Jun 9, 2016

@author: minhtran
'''
import ConfigParser
import os

PATH_TO_CONFIG_FILE = "app/dirs/cfg/apps_encoder.cfg"

# Check if file exists and readable
if not (os.path.isfile(PATH_TO_CONFIG_FILE) and os.access(PATH_TO_CONFIG_FILE, os.R_OK)):
    raise IOError("Either file %s is missing or is not readable" % PATH_TO_CONFIG_FILE)

# Create a configuration parser to parse all necessary parameters
config_parser = ConfigParser.RawConfigParser()
config_parser.read(PATH_TO_CONFIG_FILE)

