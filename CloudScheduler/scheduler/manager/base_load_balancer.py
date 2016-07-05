"""
@author: Tran Ngoc Minh
"""

"""
This class will be inherited by load balancer classes.
This is where load balancing algorithms will be implemented
and this forms the first research question. Therefore, inheritance 
of this class is open.
"""
class BaseLoadBalancer(object):
    """
    The base class for a load balancer
    """
       
    def __init__(self,numSchedulers):
        
        self.numSchedulers = numSchedulers
        self.currentScheduler = -1

    def doLoadBalancing(self):

        """
        Currently do nothing, will be extended later
        """
        
        return None
    
        