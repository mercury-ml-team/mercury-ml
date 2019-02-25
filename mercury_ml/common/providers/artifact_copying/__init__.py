"""Classes that facilitate moving data from one store into another"""
from mercury_ml.common.utils import singleton

@singleton
class S3Singleton:
    def __init__(self, **kwargs):
        import boto3
        session = boto3.Session(**kwargs)
        self.s3 = session.resource("s3")
