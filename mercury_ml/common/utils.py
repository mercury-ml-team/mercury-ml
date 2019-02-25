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


def load_referenced_json_config(filepath):
    """
    Loads a json file that has "jsonref" style references and resolves the references
    :param string filepath: The path from which to reads the JSON
    :return: dict
    """
    import jsonref
    from copy import deepcopy
    with open(filepath, "r") as f:
        config = jsonref.load(f)

    # this dereferences the dictionary produced by jsonref.load. See here https://github.com/gazpachoking/jsonref/issues/9
    config = deepcopy(config)
    return config

def recursively_update_config(config, string_formatting_dict):
    """
    Recursively updates a dictionary using string formatting.

    For example:
    > config = {
    >    "filename": "{model_purpose}_{session_id}_hdf5.h5",
    >    "save_path": "./results/{model_purpose}"
    > }
    >
    > string_formatting_dict = {"model_purpose": "test", "model_id": "12345"}
    >
    > recursively_update_config(config, string_formatting_dict)
    > print(config)
    {"filename": "test_12345_hdf5.h5", "save_path": "./results/test"}

    :param iterable config: A dictionary or list to be updated
    :param dict string_formatting_dict: A dictionary containing the information with which "config" is to be formatted
    :return:
    """

    for k in _iterate_list_or_dict(config):
        v = config[k]
        if isinstance(v, dict) or isinstance(v, list):
            recursively_update_config(v, string_formatting_dict)
        else:
            if _key_in_string(v, string_formatting_dict):
                config[k] = v.format(**string_formatting_dict)

def _iterate_list_or_dict(obj):
    """Iterates over object if list or dictionary"""
    return obj if isinstance(obj, dict) else range(len(obj))


def _key_in_string(string, string_formatting_dict):
    """Checks which formatting keys are present in a given string"""
    key_in_string = False
    if isinstance(string, str):
        for key, value in string_formatting_dict.items():
            if "{" + key + "}" in string:
                key_in_string = True
    return key_in_string