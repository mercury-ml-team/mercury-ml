"""Classes that facilitate moving data from one store into another"""

def singleton(TheClass):
    """ decorator for a class to make a singleton out of it """
    class_instances = {}

    def get_instance(*args, **kwargs):
        """ creating or just return the one and only class instance.
            The singleton depends on the parameters used in __init__ """
        key = (TheClass, args, str(kwargs))
        if key not in class_instances:
            class_instances[key] = TheClass(*args, **kwargs)
        return class_instances[key]

    return get_instance

@singleton
class S3Singleton:
    def __init__(self, **kwargs):
        import boto3
        session = boto3.Session(**kwargs)
        self.s3 = session.resource("s3")
