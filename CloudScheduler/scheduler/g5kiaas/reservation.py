#!/usr/bin/env python
"""
@author: Tran Ngoc Minh
"""
import httplib2

class Jobs: 
    reservations = {}
    
    def __init__(self, reservation):
        self.ID = str(id(self))
        Jobs.reservations[self.ID] = reservation
        
class ResMond:        
    @staticmethod
    def get_util(host):
        data, response = httplib2.Http().request('http://%s:9292' % host,
                'POST', None, headers={'Content-Type': 'application/json'})
        return dict(eval(response))
