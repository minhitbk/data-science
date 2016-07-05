"""
@author: Tran Ngoc Minh
"""

from base_load_balancer import BaseLoadBalancer
from random import randrange

class RandomLoadBalancer(BaseLoadBalancer):
    """
    Implement a random load balancer
    """

    def __init__(self,numSchedulers):

        super(RandomLoadBalancer,self).__init__(numSchedulers)
    
    """
    Return the index of the internal scheduler selected by random
    """     
    def doLoadBalancing(self):

        print "........The random load balancer is doing" \
                " load balancing"
                
        result = randrange(0,self.numSchedulers)
              
        return result