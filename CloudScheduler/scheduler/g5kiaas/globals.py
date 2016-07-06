#!/usr/bin/env python
"""
@author: Tran Ngoc Minh
"""
import os

class Environment: 
    CONF = os.path.join(os.path.expanduser("~"), ".g5kiaas/")
    ATTR = {"IP" : "ip", "HOSTNAME" : "host", "RAM" : "memnode", 
            "Frq" : "cpufreq"}    
    USER = None
    WALLTIME = None
    MOND_PATH = os.path.join(CONF, "mond")


