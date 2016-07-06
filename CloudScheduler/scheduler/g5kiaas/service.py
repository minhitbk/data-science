#!/usr/bin/env python
"""
@author: Tran Ngoc Minh
"""

import subprocess, threading
import os, traceback

from twisted.web import server, resource
from twisted.internet import reactor
import json
from random import randrange
from globals import Environment

from resources import ResourceProvision, ResourceSet


class ResourcePool:
	def __init__(self, FAKEDATA=False):
		if not FAKEDATA:
			self.provisioner = ResourceProvision()
			resources = self.provisioner.MACHINES
		else:
			resources = {'lyon': {'taurus': {'AvailableMachineNum': 2,
                     						 'Characteristics': {'Cores': 12,
                                         						 'Frq': '2.3',
                                         	      				 'RAM': '32768'},
                     			  'JobID': 666453,
                     	    	  'Machines': [{'hostname': 'taurus-5.lyon.grid5000.fr', 'ip': '172.16.48.5'},
                                  			   {'hostname': 'taurus-2.lyon.grid5000.fr', 'ip': '172.16.48.2'}],
                    			  'TotalNum': 2}},
						
						'grenoble': {'genepi': {'AvailableMachineNum': 3,
                         						'Characteristics': {'Cores': 8,
                                             						'Frq': '2.5',
                                             	     				'RAM': '8192'},
                         			'JobID': 1476039,
                            		'Machines': [{'hostname': 'genepi-31.grenoble.grid5000.fr', 'ip': '172.16.16.31'},
                                      			 {'hostname': 'genepi-4.grenoble.grid5000.fr',  'ip': '172.16.16.4'},
                                      	   		 {'hostname': 'genepi-32.grenoble.grid5000.fr', 'ip': '172.16.16.32'}],
                         			'TotalNum': 3}}
						}
		
		self.resources = {}
		self.resource_list = {}
		for site in resources:
			for cluster in resources[site]:
				rs = ResourceSet(site, cluster, resources[site][cluster])
				self.resources[rs.Path] = rs
				for ip in rs.devices:				
					self.resource_list["/%s/%s/%s" % (site, cluster, ip)] = rs
		
	def __id(self, fullpath):
		"""
			fullpath - /site/cluster/ip
			returns ip
		"""
		return fullpath[fullpath.rfind("/") + 1:]
		
	def _monitor(self):
		for path in self.resource_list:
			self.resource_list[path].devices["Utilization"] = ResMond.get_util(self.__id(path))
			print self.resource_list[path].devices["Utilization"]
		threading.Timer(60, self._monitor).start()
	
	def getStaticNodeInfo(self, **kwargs):
		nodeID = kwargs["nodeID"]
		return self.resource_list[nodeID].attributes
		
	def getDynamicNodeInfo(self, **kwargs):
		nodeID = kwargs["nodeID"]
		response = self.resource_list[nodeID].devices[self.__id(nodeID)]["Available"]
		status = "Free"
		for attribute in response:
			if response[attribute] <= 0:
				status = "Busy"
				break
		response["status"] = status
		result = json.dumps({"result": response})
		return result
		
	def isNodeRunning(self, **kwargs):
		nodeID = kwargs["nodeID"]
		return True
	
	def getAvailableNodeList(self, **kwargs):
		nodeDescriptor = kwargs["nodeDescriptor"]["machine"]
		numNode = kwargs["numNode"]
		devices_list = []
		for dev, ch in zip(self.resource_list, map(lambda res: res.attributes, 
												self.resource_list.values())):
			ok = True
			for  attr in nodeDescriptor:
				if ch[attr] < nodeDescriptor[attr]:
					ok = False
			if ok: 
				devices_list.append(dev)
				
			if len(devices_list) >= numNode:
				break
		
		result = json.dumps({"result":devices_list})
		return result

	def getNodePrice(self, **kwargs):
		result = json.dumps({"result":randrange(10,100)})
		return result	

	def checkStaticNodeInfo(self, **kwargs):
		nodeDescriptor = kwargs["nodeDescriptor"]
		nodeID = kwargs["nodeID"]
		av = self.resource_list[nodeID].attributes
		return bool(set(av.items()).intersection(set(nodeDescriptor.items())))
	
	def checkDynamicNodeInfo(self, **kwargs):
		nodeDescriptor = kwargs["nodeDescriptor"]
		nodeID = kwargs["nodeID"]		
		av = self.resource_list[nodeID].devices[self.__id(nodeID)]["Available"]
		return bool(set(av.items()).intersection(set(nodeDescriptor.items())))
		
	def createReservation(self, **kwargs):
		reservDescriptor = kwargs["reservDescriptor"]
		jobResources = []

		for key in reservDescriptor:
			if key == "network" or key == "storage" or key == "router":		
				pass
			else:
				for item in reservDescriptor[key]:	
					mid = item.keys()[0]
					attr = item[mid]
					available_attributes = self.resource_list[mid].devices[
												self.__id(mid)]["Available"]
					ok = True
					for attr in item[mid]:
						if available_attributes[attr] < item[mid][attr]:
							ok = False
					if ok:
						jobResources.append(item)
					else:
						jobResources = None
						break
		
		if jobResources == None:
			return json.dumps({"result": False})

		##remove from available resources				
		for item in jobResources:
			ip = item.keys()[0]
			characteristics = item[ip]
			for k in characteristics:
				try:
					int(self.resource_list[ip].devices[self.__id(ip)]["Available"][k])
					self.resource_list[ip].devices[self.__id(ip)]["Available"][k] = \
							int(self.resource_list[ip].devices[self.__id(ip)]["Available"][k])\
								 - int(characteristics[k])
				except:
					continue
		result = json.dumps({"result":Jobs(jobResources).ID})
		return result
		
	def releaseReservation(self, **kwargs):
		reservID = kwargs["reservID"]
		try:
			resources = Jobs.reservations[reservID]
			for item in resources:
				ip = item.keys()[0]
				characteristics = item[ip]
				for k in characteristics:
					try:
						int(self.resource_list[ip].devices[self.__id(ip)]["Available"][k])
						self.resource_list[ip].devices[self.__id(ip)]["Available"][k] = \
							int(self.resource_list[ip].devices[self.__id(ip)]["Available"][k]) \
								+ int(characteristics[k])
					except:
						continue
			
			del Jobs.reservations[reservID]
		except:
			return json.dumps({"result":False})
		
		result = json.dumps({"result":True})
		return result
		
class Service(resource.Resource):
	
	isLeaf = True	
	
	def __init__(self):
		self.RP = ResourcePool()
		
	def http_handler(self, request):
		action = request.postpath[0]
		print "\nAction: %s" % action

		data = eval(request.content.read())
		print "Args    : ", data
		
		response = (lambda :ResourcePool.__dict__[action](self.RP, **data))()	
		
		print "Response ", response		
		return str(response)

	def render_POST(self, request):
		return self.http_handler(request)

	def render_GET(self, request):
		return self.http_handler(request)

def start_server(port):
	try:
		cinfo = filter(lambda line: "USERNAME" in line, 
								open(os.path.join(Environment.CONF,\
								g5kiaas.conf"),"r").readlines())[0]
		
		Environment.USER = cinfo.split("=")[1].strip()

	except:
		print "Config file not found - Grid5000 username not retrieved. Check README!"
		traceback.print_exc()
		sys.exit(1)
	
	cmd = "lsof -i:%d | grep 'python' | awk '{ print $(NF-8) }'" % port
	pid = subprocess.Popen(cmd, shell=True, 
						stdout=subprocess.PIPE).communicate()[0].strip()
	
	if not (pid in [None, ""]):
		pid = int(pid)
		subprocess.Popen("kill -9 %d" % int(pid), shell=True, 
						stdout=subprocess.PIPE).wait()
			
	site = server.Site(Service())
	print "Grid5000 based IaaS started....listening on port %d" % port

	reactor.listenTCP(port, site)
	reactor.run()
