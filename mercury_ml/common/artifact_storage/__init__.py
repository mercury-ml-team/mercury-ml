"""Classes that facilitate moving data from one store into another"""
from mercury_ml.common.utils import singleton


@singleton
class MongoClientSingleton:
    def __init__(self, **kwargs):
        from pymongo import MongoClient
        self.client = MongoClient(**kwargs)
