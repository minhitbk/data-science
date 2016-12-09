""" 
 Copyright (C) Arimo, Inc - All Rights Reserved.
 Unauthorized copying of this file, via any medium is strictly prohibited.
 Proprietary and confidential.
 Written by Minh Tran <minhtran@arimo.com>, Nov 2016.
"""

import os

import ConfigParser


PATH_TO_CONFIG_FILE = "/Users/minhtran/installed_softwares/adatao/" \
                      "DeepLearningApps/arimo/dlots_fr/etc/dlots_fw.cfg"

# Check if file exists and readable
if not (os.path.isfile(PATH_TO_CONFIG_FILE) and os.access(
        PATH_TO_CONFIG_FILE, os.R_OK)):
    raise IOError("Either file %s is missing or not "
                  "readable..." % PATH_TO_CONFIG_FILE)

# Create a configuration parser to parse all necessary parameters
config_parser = ConfigParser.RawConfigParser()
config_parser.read(PATH_TO_CONFIG_FILE)


