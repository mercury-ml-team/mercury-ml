# Mercury-ML includes several implementations for storing different types of data. In this example, we'll store a pandas
# dataframe as a pickle file. Pandas has a well-defined built-in function for this, in the form of df.to_pickle. It may
# not be immediately obvious why it make sense to use mercury_ml.common.artifact_storage.local.store_pandas_pickle
# instead.

# There are two main reasons: the one is convenience, since the mercury_ml implementation also takes care of certain
# necessary steps, such as creating the folders needed for the storage.

# The second is to achieve modularity, allowing us to easily switch storage functions via a configuration file, without
# needing to change our code. We'll see how this in a later example

# create dummy data
import pandas as pd
df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})

# save dummy data
from mercury_ml.common.artifact_storage.local import store_pandas_pickle
path = store_pandas_pickle(data=df, directory="./stored_data/local", filename="my_dummy_data")

# Often it is the case that we want to store the produced artifacts in a remote storage local (such as S3, GCS und HDFS)
# Mercury-ML has built in functions to copy to different locations. In this example we'll simply assume that a different
# folder on the local drive is our "remote" location
from mercury_ml.common.artifact_copying.from_disk import copy_from_disk_to_disk
import os
copy_from_disk_to_disk(source_dir="./stored_data/local",
                       target_dir="./stored_data/remote",
                       filename=os.path.basename(path))
