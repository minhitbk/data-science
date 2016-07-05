"""
@author: Tran Ngoc Minh
"""

"""
The following information is used to form a faked cloud 
infrastructure for testing
"""

MACHINE_TYPE_1 = {"numcpu":8, 
                  "numcore":2, 
                  "frequency":3, 
                  "numfpga":5, 
                  "memfpga":10, 
                  "typefpga":"", 
                  "numgpu":4, 
                  "typegpu":"", 
                  "memory":20,
                  "disk":20,
                  "quantity":10}

MACHINE_TYPE_2 = {"numcpu":8, 
                  "numcore":2, 
                  "frequency":3, 
                  "numfpga":0, 
                  "memfpga":0, 
                  "typefpga":"", 
                  "numgpu":0, 
                  "typegpu":"", 
                  "memory":20,
                  "disk":20,
                  "quantity":100}

MACHINE_TYPE_3 = {"numcpu":0, 
                  "numcore":0, 
                  "frequency":0, 
                  "numfpga":8, 
                  "memfpga":10, 
                  "typefpga":"", 
                  "numgpu":0, 
                  "typegpu":"", 
                  "memory":0,
                  "disk":20,
                  "quantity":100}

ROUTER_TYPE_1 = {"inbw":20, 
                 "outbw":20, 
                 "buffer":100,
                 "capacity":20, 
                 "connector":"", 
                 "version":"",
                 "quantity":100} 

ROUTER_TYPE_2 = {"inbw":20, 
                 "outbw":20, 
                 "buffer":100,
                 "capacity":100, 
                 "connector":"", 
                 "version":"",
                 "quantity":100} 

STORAGE_TYPE_1 = {"capacity":100,
                  "ranbw":10,
                  "seqbw":50,
                  "quantity":100}

STORAGE_TYPE_2 = {"capacity":100,
                  "ranbw":50,
                  "seqbw":10,
                  "quantity":100}
