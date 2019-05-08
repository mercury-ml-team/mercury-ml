# DataBunches are logical groupings of DataSets. A typicall DataBunch would group together three DataSets: "train", "valid"
# and "test".

# In this example we'll create a DataBunch of DataSets consisting of PandasDataWrappers

# NOTE: the code below might seem messy and repetitive, and it is. In the next examples (4 and 5) we'll show how this can be
# cleaned up and made more generic

import pandas as pd
from mercury_ml.common.data_wrappers.pandas import PandasDataWrapper
from mercury_ml.common.data_set import DataSet
from mercury_ml.common.data_bunch import DataBunch

# read Pandas DFs as example inputs
df_train = pd.read_csv("./example_data/train.csv")
df_valid = pd.read_csv("./example_data/valid.csv")
df_test = pd.read_csv("./example_data/test.csv")

index_columns = ["ID","ID2"]
features_columns = ["field1_num", "field2_num", "field3_num"]
targets_columns = ["field4_target", "field5_target", "field6_target"]
full_data_columns= index_columns + features_columns + targets_columns

data_bunch = DataBunch(data_sets_dict = {
    "train": DataSet({
        "full_data": PandasDataWrapper(df_train, full_data_columns),
        "index": PandasDataWrapper(df_train[index_columns], index_columns),
        "features": PandasDataWrapper(df_train[features_columns], features_columns),
        "targets": PandasDataWrapper(df_train[targets_columns], targets_columns)
    }),
    "valid": DataSet(
    {
        "full_data": PandasDataWrapper(df_valid, full_data_columns),
        "index": PandasDataWrapper(df_valid[index_columns], index_columns),
        "features": PandasDataWrapper(df_valid[features_columns], features_columns),
        "targets": PandasDataWrapper(df_valid[targets_columns], targets_columns)
    }),
    "test": DataSet(
    {
        "full_data": PandasDataWrapper(df_test, full_data_columns),
        "index": PandasDataWrapper(df_test[index_columns], index_columns),
        "features": PandasDataWrapper(df_test[features_columns], features_columns),
        "targets": PandasDataWrapper(df_test[targets_columns], targets_columns)
    })
})

# let's see what our data_bunch consists of
print(data_bunch)

# we can also add data_sets to data_bunch using data_bunch.add_data_set, for example:
data_bunch.add_data_set("my_data_set", DataSet({"my_data_wrapper": PandasDataWrapper(df_valid, full_data_columns)}))
print(data_bunch)

# we can transform an entire data bunch (or a only a subset of data_sets within it if we so choose) to a different data
# format using a single command
numpy_data_bunch = data_bunch.transform(data_set_names=["valid", "test"],
                                        params={"transform_to": "numpy"},
                                        transform_then_slice=False)

print(numpy_data_bunch)
