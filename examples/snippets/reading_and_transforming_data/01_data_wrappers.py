# In this snippet, we'll wrap a Pandas DataFrame within a PandasDataWrapper class, and show some of the functionality that the wrapper allows us.

import pandas as pd
from mercury_ml.common.data_wrappers.pandas import PandasDataWrapper

# read a Pandas DF as example
df = pd.read_csv("./example_data/train.csv")
print("field names:",list(df.columns))

# this df consists of index, feature and target data as would typically be the case for tabular data used in Machine Learning
# we will create a PandasDataWrapper of the feature data using the class constructor, as follows:

feature_columns = ["field1_num", "field2_num", "field3_num"]

data_wrapper = PandasDataWrapper(underlying=df[feature_columns], # this is the dataframe that is wrapped
                                 field_names=feature_columns # these are the fieldnames related to the underlying dataframe
                                 )

print("data_wrapper:", type(data_wrapper))

# The underlying dataframe can be accessed via data_wrapper.underlying
print("data_wrapper.underlying:", type(data_wrapper.underlying))


# The PandasDataWrapper class has several built-in functions that allow transformation to other data wrappers, for example

numpy_data_wrapper = data_wrapper.to_numpy()
print("numpy_data_wrapper:", type(numpy_data_wrapper))
print("numpy_data_wrapper.underlying:", type(numpy_data_wrapper.underlying))