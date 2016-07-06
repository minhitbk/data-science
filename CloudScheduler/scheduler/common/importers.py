"""
@author: Tran Ngoc Minh
"""
import sys,traceback

def importClass(importStr):
    """
    Returns a class from a string including module and class
    """
    moduleStr, _sep, classStr = importStr.rpartition(".")
    
    try:
        __import__(moduleStr)
        return getattr(sys.modules[moduleStr], classStr)
    except (ValueError, AttributeError):
        raise ImportError("Class %s cannot be found (%s)" %
                (classStr, traceback.format_exception(*sys.exc_info())))

def importObject(importStr, *args, **kwargs):
    """
    Import a class and return an instance of it
    """
    return importClass(importStr)(*args, **kwargs)

def importModule(importStr):
    """
    Import a module
    """
    __import__(importStr)
    return sys.modules[importStr]
