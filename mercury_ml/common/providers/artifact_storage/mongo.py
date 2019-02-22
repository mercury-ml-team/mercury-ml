from pymongo import MongoClient

KEY_SEPARATOR = "/"

def __format_doc_key(mongo_doc_key, sep):
    """
    Parses the mongo_doc_key parameter, splitting the given string into parts, using a separator.
    Takes the first part as a header key and bundles the rest of the keys in a list.
    :param string mongo_doc_key: Path-like key, representing a hierarchy for a nested document structure
    :param string sep: Separator string to parse the nested keys
    :return: (string, list) tuple containing (header_key, body_keys), where header_key is the first value as a string
    and the body_keys are the rest of the keys as a list
    """
    header_key = mongo_doc_key
    body_keys = []

    if KEY_SEPARATOR in mongo_doc_key:
        split_keys = mongo_doc_key.split(sep)
        header_key = split_keys[0]
        body_keys = split_keys[1:]

    return header_key, body_keys


def store_dict_on_mongo(data, mongo_params_dict):
    """
    Saves the given dict on MongoDB, by wrapping it as a nested dictionary with the given keys.

    :param dict data: Data to be wrapped and saved
    :param dict mongo_params_dict: Dictionary for the keys to wrap the data to be saved as a document and the connection params.
    Expected keys are:
        "doc_key": Hierarchical keys in the form of key1\key2\key3... to wrap the underlying data to be stored.
        "conn_params": MongoDB connection params, where "host", "database" and "collection" keys are present within.
    :return:
    """

    mongo_client = MongoClient(host=mongo_params_dict["conn_params"]["host"])

    mongo_db = getattr(mongo_client, mongo_params_dict["conn_params"]["database"])
    mongo_collection = getattr(mongo_db, mongo_params_dict["conn_params"]["collection"])

    # Create a nested dictionary with the supplied nested keys
    # e.g. if you configure your keys as "{model_id}/custom_metrics/test"
    # the dict to be stored will be {"MODEL_ID"}:
    # ...................................{"custom_metrics":
    # .......................................{"test":
    # ............................................DATA"
    # .......................................}

    mongo_doc_key = mongo_params_dict["doc_key"]
    header_key, body_keys = __format_doc_key(mongo_doc_key, KEY_SEPARATOR)

    header_data = {"model_id": header_key}

    wrapped_data = data
    for key in reversed(body_keys):
        wrapped_data = {key:wrapped_data}

    mongo_collection.update(
        header_data,
        {
            "$set": wrapped_data,
            "$setOnInsert": header_data
        },
        upsert= True
    )