# The previous three examples contain a lot of boiler plate code to perform something that is conceptually very simple: read
# data from disk and group it together logically using DataWrappers, DataSets and DataBunches.

# In this example we'll show how you can use the source_reading API significantly simplify the code you need to write
# in order to achieve this goal

from mercury_ml.common.source_reading.disk import read_pandas_data_set
from mercury_ml.common.data_bunch import DataBunch

# In the previous example we read Pandas DataFrames in manually and the assigned them as the underlyning data in
# DataWrappers. Here the "read_pandas_data_set" function will take care of that for us

data_wrappers_params_dict = {
    "index": ["ID", "ID2"],
    "features": ["field1_num", "field2_num", "field3_num"],
    "targets": ["field4_target", "field5_target", "field6_target"],
    "full_data": ["ID", "ID2", "field1_num", "field2_num", "field3_num", "field4_target", "field5_target", "field6_target"]
}

data_sets_dict = {
    "train": read_pandas_data_set(path="./example_data/train.csv",
                                  data_wrappers_params=data_wrappers_params_dict,
                                  input_format=".csv"),
    "valid": read_pandas_data_set(path="./example_data/valid.csv",
                                  data_wrappers_params=data_wrappers_params_dict,
                                  input_format=".csv"),
    "test": read_pandas_data_set(path="./example_data/test.csv",
                                  data_wrappers_params=data_wrappers_params_dict,
                                  input_format=".csv")
}

data_bunch = DataBunch(data_sets_dict)
print(data_bunch)