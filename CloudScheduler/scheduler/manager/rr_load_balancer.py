"""
@author: Tran Ngoc Minh
"""

from base_load_balancer import BaseLoadBalancer

class RRLoadBalancer(BaseLoadBalancer):
    """
    Implement a round robin load balancer
    """
   
    def __init__(self,numSchedulers):

        super(RRLoadBalancer,self).__init__(numSchedulers)
        
    """
    Return the index of the internal scheduler selected by round robin
    """     
    def doLoadBalancing(self):
        
        print "........The round robin load balancer is doing" \
                " load balancing"
                
        result = (self.currentScheduler + 1) % (self.numSchedulers)
             
        return result