#!/usr/bin/env python
"""
@author: Tran Ngoc Minh
"""
import os

home = os.path.expanduser("~")

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
    extra = {}

if not os.path.exists(os.path.join(home, ".harness-crs")):
    os.mkdir(os.path.join(home, ".harness-crs"))

setup(name="harness-crs", version="1.0",
      description="Harness Cross Resource Scheduler",    
      scripts=["bin/harness-schedulers"],
      data_files=[(os.path.join(home, ".harness-crs"), 
                   ['conf/scheduler.conf'])],
      install_requires=['flask'],
      packages=["scheduler", "scheduler/worker", "scheduler/manager", 
                "scheduler/worker/filtering", "scheduler/worker/weighting", 
                "scheduler/unittester", "scheduler/database", 
                "scheduler/common"],
      classifiers = ["Operating System :: OS Independent", "Topic :: Internet",
                     "Programming Language :: Python :: 2.7"]
      )
os.system("rm -rf  build dist harness_crs.egg-info")
