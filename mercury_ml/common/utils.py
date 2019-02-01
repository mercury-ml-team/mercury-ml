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
    >    "filename": "{model_purpose}_{model_id}_hdf5.h5",
    >    "save_path": "./results/{model_purpose}"
    > }
    >
    > string_formatting_dict = {"model_purpose": "test", "model_id": "12345"}
    >
    > recursively_update_config(config, string_formatting_dict)
    > print(config)
    {"filename": "test_12345_hdf5.h5", "save_path": "./results/test"}

    :param dict config: A dictionary to be updated
    :param dict string_formatting_dict: A dictionary containing the information with which "config" is to be formatted
    :return:
    """

    for k, v in config.items():
        if isinstance(v, dict):
            recursively_update_config(v, string_formatting_dict)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    recursively_update_config(item, string_formatting_dict)
                else:
                    if _key_in_string(item, string_formatting_dict):
                        config[k] = item.format(**string_formatting_dict)
        else:
            if _key_in_string(v, string_formatting_dict):
                config[k] = v.format(**string_formatting_dict)

def _key_in_string(string, string_formatting_dict):
    """Checks which formatting keys are present in a given string"""
    key_in_string = False
    if isinstance(string, str):
        for key, value in string_formatting_dict.items():
            if "{" + key + "}" in string:
                key_in_string = True
    return key_in_string