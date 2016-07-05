"""
@author: Tran Ngoc Minh
"""

import unittest
from scheduler.database.db_class import Base, dbEngine

class CommonUnitTester(unittest.TestCase):
    
    def setUp(self):

        Base.metadata.drop_all(bind=dbEngine)
        Base.metadata.create_all(bind=dbEngine)
