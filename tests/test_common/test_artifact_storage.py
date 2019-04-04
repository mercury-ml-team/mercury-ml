import pytest
import os
import sys
import json
from random import randint
from mercury_ml.common.artifact_storage.local import store_dict_json, store_pandas_json, store_pandas_pickle, \
    store_h2o_frame
import shutil

input_dict = {"hello": [randint(0,100),randint(0,100),randint(0,100),randint(0,100)]}
dir = "./results"

if os.path.isdir(dir):
    shutil.rmtree(dir)

os.makedirs(dir)

@pytest.mark.parametrize("input_dict, directory, filename, force, parts",
                         [(input_dict, dir, "test_h2o", False, 1),
                          (input_dict, dir, "test_h2o_dir", False, 2)])
def test_store_h2o_frame(input_dict, directory, filename, force, parts):
    import pandas as pd
    import h2o
    h2o.init()
    data = h2o.H2OFrame(pd.DataFrame(input_dict))
    store_h2o_frame(data, directory, filename, force, parts)

    filepath = os.path.join(directory, filename)

    if parts > 1:
        assert os.path.isdir(filepath)
    else:
        assert os.path.isfile(filepath)

    #TODO
    # stored_data = ...(filepath)
    # assert stored_data.equals(data)


@pytest.mark.parametrize("input_dict, directory, filename, compression",
                         [(input_dict, dir,"test_pickle", None)])
def test_store_pandas_pickle(input_dict, directory, filename, compression):
    import pandas as pd
    data = pd.DataFrame(input_dict)
    store_pandas_pickle(data, directory, filename, compression)

    filepath = os.path.join(directory, filename + ".pkl")
    assert os.path.isfile(filepath)

    stored_data = pd.read_pickle(filepath)
    assert stored_data.equals(data)


@pytest.mark.parametrize("input_dict, directory, filename, orient, compression",
                         [(input_dict, dir,"test_json", "table", None)])
def test_store_pandas_json(input_dict, directory, filename, orient, compression):
    import pandas as pd
    data = pd.DataFrame(input_dict)
    store_pandas_json(data, directory, filename, orient, compression)

    filepath = os.path.join(directory, filename + ".json")
    assert os.path.isfile(filepath)

    stored_data = pd.read_json(filepath, orient=orient)
    assert stored_data.equals(data)



@pytest.mark.parametrize("input_dict, directory, filename",
                         [(input_dict, dir,"test_dict")])
def test_store_dict_json(input_dict, directory, filename):
    data = input_dict
    store_dict_json(data, directory, filename)

    filepath = os.path.join(directory, filename+ ".json")
    assert os.path.isfile(filepath)

    with open(filepath, "r") as f:
        stored_data = json.load(f)

    assert stored_data == data


