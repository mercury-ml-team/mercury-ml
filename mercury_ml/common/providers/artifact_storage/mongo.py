def store_dict_on_mongo(data, document_id, database_name, collection_name, mongo_client_params, document_key="",
                        document_key_separator="/", reuse_existing=True, overwrite=True):
    """
    Takes a dictionary input and stores it in an appropriate key in MongoDB

    :param dict data: A dictionary with the data to be stored
    :param document_id: The unique id, corresponding to the _id of the documented to be upserted (updated / inserted to) is found
    :param string database_name: The name of the database where the document to be upserted to is found
    :param string collection_name: The name of the collection where the document to be upserted to is found
    :param dict mongo_client_params: Paramters to be passed to the MongoClient constructor
    :param string document_key: a string representing the key where the data should be stored. For example, a key
    "metrics/test" will store will store the data in the format:
    {
        "_id": document_id
        "metrics" : {
            "test": data
        }
    }
    :param string document_key_separator: The separator with which to parse the document_key. Default "/"
    :param bool reuse_existing: If False a new MongoClient connection will be created every time the function is called.
    If True an existing connection will be reused.

    :return:
    """
    if not reuse_existing:
        from pymongo import MongoClient
        mongo_client = MongoClient(**mongo_client_params)
    else:
        from mercury_ml.common.providers.artifact_storage import MongoClientSingleton
        mongo_client = MongoClientSingleton(**mongo_client_params).client

    mongo_db = getattr(mongo_client, database_name)
    mongo_collection = getattr(mongo_db, collection_name)

    document_key = document_key.replace(document_key_separator, ".")

    if overwrite or not _document_exists(mongo_collection, document_id):
        mongo_collection.update_one(
            {"_id": document_id},
            {"$set": {document_key: data}},
            upsert=True
        )
    else:
        print("Document with _id {} in collection {} on database {} already exists and 'overwrite' has been set to False. Nothing will be written to MongoDB".format(document_id, collection_name, database_name))


def _document_exists(mongo_collection, document_id):
    cursor = mongo_collection.find({document_id: {"$exists": True}}).limit(1)

    if cursor.count() > 0:
        return True
    else:
        return False
