"""Classes that facilitate moving data from one store into another"""
from mercury_ml.common.utils import singleton
from pymongo import MongoClient

@singleton
class MongoClientSingleton:
    def __init__(self, **kwargs):
        self.client = MongoClient(**kwargs)
