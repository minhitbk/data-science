"""
@author: Tran Ngoc Minh
"""

from scheduler.manager.service import app
from gunicorn.app.base import Application

class SchedulingManager(Application):
    """
    This is a wrapper for Gunicorn application
    """

    def __init__(self, options={}):
        """
        Load the base config and assign some core attributes
        """
        
        self.usage = None
        self.callable = None
        self.prog = None
        self.options = options
        self.do_load_config()

    def init(self, *args):
        """
        Takes our custom options from self.options and creates a config
        dict which specifies custom settings
        """

        cfg = {}
        for k, v in self.options.items():
            if k.lower() in self.cfg.settings and v is not None:
                cfg[k.lower()] = v

        return cfg

    def load(self):
        """
        Imports our application and returns it to be run.
        """   

        return app        
        