"""
@author: Tran Ngoc Minh
"""

import httplib

class Connection(object):
    """
    This is a wrapper for http connections
    """
    
    def __init__(self,hostname,port):
        
        self.conn = httplib.HTTPConnection(hostname,port,timeout=None)
        
    """
    Use this function to send request to an http server
    """
    def request(self,method,url,body,headers={"Content-type":"application/json"}):
        
        self.conn.request(method,url,body,headers)
        response = self.conn.getresponse()
        
        return response.read()
    
    """
    Call this function if you do not access to the http server anymore
    """
    def close(self):
        
        self.conn.close()

        