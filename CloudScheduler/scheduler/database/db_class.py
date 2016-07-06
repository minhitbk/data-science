"""
@author: Tran Ngoc Minh
"""

"""
This module implements the database designed for the scheduler.
We use ORM technology to store database.
"""
from sqlalchemy.engine import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship, sessionmaker

from scheduler.common.loaders import config

"""
Create an engine, which the Session will use for resource connection
"""
dbEngine = create_engine(config.get("Database","DATABASE"))

"""
Create a configured "Session" class
"""
Session = sessionmaker(bind=dbEngine)

"""
Create a session
"""
session = Session()

Base = declarative_base()

"""
Store the scheduling result for an application configuration 
"""
class AppConfigSchedule(Base):
    
    __tablename__ = "AppConfigSchedule"
    configID = Column(String, primary_key=True)
    storageInfo = Column(String)
    machineInfo = Column(String)
    networkInfo = Column(String)
    routerInfo  = Column(String)
    
    def __init__(self, configID, storageInfo, machineInfo, networkInfo,
                 routerInfo):
        self.configID = configID
        self.storageInfo = storageInfo
        self.machineInfo = machineInfo
        self.networkInfo = networkInfo
        self.routerInfo = routerInfo

"""
Store storage types
"""    
class StorageType(Base):
    __tablename__ = "StorageType"
    typeID = Column(String, primary_key=True)
    capacity = Column(Float)
    ranbw = Column(Float)
    seqbw = Column(Float)
    
    """
    Create a relationship with StorageResource
    """
    resource = relationship("StorageResource", backref="type")
        
    def __init__(self, typeID, capacity, ranbw, seqbw):
        self.typeID = typeID
        self.capacity = capacity
        self.ranbw = ranbw
        self.seqbw = seqbw
        return

"""
Store storage resources
"""
class StorageResource(Base):
    __tablename__ = "StorageResource"
    resourceID = Column(String, primary_key=True)
    status = Column(String)
    reservID = Column(String)
    
    typeID = Column(String, ForeignKey("StorageType.typeID"))
   
    def __init__(self, resourceID, typeID, status="free", reservID=""):
        self.resourceID = resourceID
        self.typeID = typeID
        self.status = status
        self.reservID = reservID
        return

"""
Store router types
"""    
class RouterType(Base):
    __tablename__ = "RouterType"
    typeID = Column(String, primary_key=True)
    inbw = Column(Float)
    outbw = Column(Float)
    buffer = Column(Float)
    capacity = Column(Float)
    connector = Column(String)
    version = Column(String)
                    
    """
    Create a relationship with RouterResource
    """
    resource = relationship("RouterResource", backref="type")
        
    def __init__(self, typeID, inbw, outbw, buff, capacity, connector="", 
                 version=""):
        self.typeID = typeID
        self.inbw = inbw
        self.outbw = outbw
        self.buffer = buff
        self.capacity = capacity
        self.connector = connector
        self.version = version
        return

"""
Store router resources
"""
class RouterResource(Base):
    __tablename__ = "RouterResource"
    resourceID = Column(String, primary_key=True)
    status = Column(String)
    reservID = Column(String)
    
    typeID = Column(String, ForeignKey("RouterType.typeID"))
   
    def __init__(self, resourceID, typeID, status="free", reservID=""):
        self.resourceID = resourceID
        self.typeID = typeID
        self.status = status
        self.reservID = reservID
        return

"""
Store machine types
"""    
class MachineType(Base):
    __tablename__ = "MachineType"
    typeID = Column(String, primary_key=True)
    numcpu = Column(Float)
    numcore = Column(Float)
    frequency = Column(Float)
    numfpga = Column(Float)
    memfpga = Column(Float)
    typefpga = Column(String)
    numgpu = Column(Float)
    typegpu = Column(String)
    memory = Column(Float)  
    disk = Column(Float)  
                    
    """
    Create a relationship with MachineResource
    """
    resource = relationship("MachineResource", backref="type")
        
    def __init__(self, typeID, numcpu, numcore, frequency, numfpga, memfpga, 
                 numgpu, memory, disk, typegpu="", typefpga=""):
        self.typeID = typeID
        self.numcpu = numcpu
        self.numcore = numcore
        self.frequency = frequency
        self.numfpga = numfpga
        self.memfpga = memfpga
        self.numgpu = numgpu
        self.typegpu = typegpu
        self.memory = memory
        self.typefpga = typefpga
        self.disk = disk
        return

"""
Store machine resources
"""
class MachineResource(Base):
    __tablename__ = "MachineResource"
    resourceID = Column(String, primary_key=True)
    status = Column(String)
    reservID = Column(String)
    
    typeID = Column(String, ForeignKey("MachineType.typeID"))
   
    def __init__(self, resourceID, typeID, status="free", reservID=""):
        self.resourceID = resourceID
        self.typeID = typeID
        self.status = status
        self.reservID = reservID
        return

"""
Store network bandwidths, for future use
"""
class NetworkBandwidth(Base):
    __tablename__ = "NetworkBandwidth"
    id = Column(Integer, primary_key=True)
    segment = Column(String)
    bw = Column(Float)
    
    def __init__(self, segment, bandwidth):
        self.segment = segment
        self.bw = bandwidth
        return
    