# DataSets are logical groupings of DataWrappers. A DataSet might for example be the complete "train" data used in a
# Machine Learning Workflow, consisting of (typically) features, targets and index DataWrappers.

# in this example we'll build a DataSet for "train" data consisting of PandasDataWrappers

import pandas as pd
from mercury_ml.common.data_wrappers.pandas import PandasDataWrapper
from mercury_ml.common.data_set import DataSet

# read a Pandas DF as example
df = pd.read_csv("./example_data/train.csv")
print("field names:", list(df.columns))


# The above dataframe (as is of the case with tabular data) can be logically divided in three parts: index, features and
# targets.  We create a PandasDataWrapper for each of these and use the DataSet constructor to initialise the dataset

index_columns = ["ID","ID2"]
features_columns = ["field1_num", "field2_num", "field3_num"]
targets_columns = ["field4_target", "field5_target", "field6_target"]

train_data_set = DataSet(
    {
        "index": PandasDataWrapper(df[index_columns], index_columns),
        "features": PandasDataWrapper(df[features_columns], features_columns),
        "targets": PandasDataWrapper(df[targets_columns], targets_columns)
    }
)

# our object train_data_set now consists for three DataWrappers
print(train_data_set)

# The underlying data can be accessed via train_data_set.<data_wrapper_name>.underlying and train_data_set.<data_wrapper_name>.field_names
print("underlying:",type(train_data_set.index.underlying),
      "field_names:", train_data_set.index.field_names)

print("underlying:", type(train_data_set.features.underlying),
      "field_names:", train_data_set.features.field_names)

print("underlying:", type(train_data_set.targets.underlying),
      "field_names:", train_data_set.targets.field_names)

# We can also add DataWrappers afterwards, for example, the following will add a wrapper that consists of the full data set with all columns
# "full_data" is regarded as a special DataWrapper, the use of which becomes apparent when transforming data.
full_data_columns= index_columns+features_columns+targets_columns
train_data_set.add_data_wrapper(data_wrapper_name="full_data",
                                data_wrapper=PandasDataWrapper(df[full_data_columns], full_data_columns))

print("underlying:", type(train_data_set.full_data.underlying),
      "field_names:", train_data_set.full_data.field_names)

# Now let's have a look at how the transforming the contents of a DataSet to numpy (or any other supported format)
numpy_train_data_set = train_data_set.transform(transformation_params={"transform_to": "numpy"})
print(numpy_train_data_set)

# In the above example, each of the DataWrappers are individually converted to Numpy. We could however all just have converted
# the "full_data", and then created the other by slicing the resulting Numpy array. This could be done as follows
numpy_train_data_set2 = train_data_set.transform(transformation_params={"transform_to": "numpy"},
                                                 transform_then_slice=True)
print(numpy_train_data_set)