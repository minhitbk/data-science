#!/usr/bin/env python
"""
@author: Tran Ngoc Minh
"""

import pxssh
import sys

class RemoteConn:
    
    def __init__(self):

        pass
    
    def run(self, host, user, cmd):

        conn = pxssh.pxssh() 
        
        if not conn.login (host, user):

            print "SSH session failed on site %s." % host
            
            sys.exit(1)
            
        print "Running '%s' on %s@%s" % (cmd, user, host)

        conn.sendline(cmd)
        conn.prompt(timeout = None)
        output = conn.before[:]
        
        conn.logout()
        del conn
        
        return output