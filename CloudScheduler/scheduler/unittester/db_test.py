"""
@author: Tran Ngoc Minh
"""

import unittest
from scheduler.database.db_class import session, StorageType, RouterType, MachineType
from base_class import CommonUnitTester 

TEST_DB_DATA = {
                "storage":{"typeID":"STORAGE",
                           "capacity":100,
                           "ranbw":10,
                           "seqbw":50                    
                           },
                
                "machine":{"typeID":"MACHINE",
                           "numcpu":8, 
                           "numcore":2, 
                           "frequency":3, 
                           "numfpga":5, 
                           "memfpga":10, 
                           "typefpga":"", 
                           "numgpu":4, 
                           "typegpu":"", 
                           "memory":20,
                           "disk":20
                           },
                            
                "router":{"typeID":"ROUTER",
                          "inbw":20, 
                          "outbw":20, 
                          "buffer":100,
                          "capacity":20, 
                          "connector":"", 
                          "version":""
                          }
                }

class DatabaseUnitTester(CommonUnitTester):
    
    def testStorageType(self,data=TEST_DB_DATA["storage"]):
        
        storageType = StorageType(data["typeID"],
                                  data["capacity"],
                                  data["ranbw"],
                                  data["seqbw"])

        session.add(storageType)
        session.commit()

        dbType = session.query(StorageType).get("STORAGE")

        self.assertEqual(dbType.capacity,data["capacity"])
        self.assertEqual(dbType.ranbw,data["ranbw"])
        self.assertEqual(dbType.seqbw,data["seqbw"])

    def testMachineType(self,data=TEST_DB_DATA["machine"]):
        
        machineType = MachineType(data["typeID"],
                                  data["numcpu"],
                                  data["numcore"],
                                  data["frequency"],
                                  data["numfpga"],
                                  data["memfpga"],
                                  data["numgpu"],
                                  data["memory"],
                                  data["disk"],
                                  data["typegpu"],
                                  data["typefpga"])

        session.add(machineType)
        session.commit()

        dbType = session.query(MachineType).get("MACHINE")

        self.assertEqual(dbType.numcpu,data["numcpu"])
        self.assertEqual(dbType.numcore,data["numcore"])
        self.assertEqual(dbType.frequency,data["frequency"])
        self.assertEqual(dbType.numfpga,data["numfpga"])
        self.assertEqual(dbType.memfpga,data["memfpga"])
        self.assertEqual(dbType.numgpu,data["numgpu"])
        self.assertEqual(dbType.memory,data["memory"])
        self.assertEqual(dbType.disk,data["disk"])
        self.assertEqual(dbType.typegpu,data["typegpu"])
        self.assertEqual(dbType.typefpga,data["typefpga"])                
        
    def testRouterType(self,data=TEST_DB_DATA["router"]):
        
        routerType = RouterType(data["typeID"],
                                data["inbw"],
                                data["outbw"],
                                data["buffer"],
                                data["capacity"],
                                data["connector"],
                                data["version"])

        session.add(routerType)
        session.commit()

        dbType = session.query(RouterType).get("ROUTER")

        self.assertEqual(dbType.inbw,data["inbw"])
        self.assertEqual(dbType.outbw,data["outbw"])
        self.assertEqual(dbType.buffer,data["buffer"])                        
        self.assertEqual(dbType.capacity,data["capacity"])
        self.assertEqual(dbType.connector,data["connector"])
        self.assertEqual(dbType.version,data["version"])          
                                             
if __name__ == "__main__":
    
    suite = unittest.TestSuite()
    suite.addTest(DatabaseUnitTester("testRouterType"))
    unittest.TextTestRunner().run(suite)
#    unittest.main(defaultTest=testRouterType)
    