#  `mercury-ml`

![logo](docs/images/logo_07_200x200.jpg)


In the ancient Roman mythology, the god Mercury was known as the messenger of the gods. Wearing winged shoes and a winged
hat he zipped between Mount Olympus and the kingdoms of men and saw to it that the will of the gods was known.

We've chosen `mercury-ml` as the name of this package because we see its role as very similar.

Recent developments in Machine Learning and Data Processing tools have led to a myriad of except open source libraries
each of which provide well developed and transparent APIs. Where it becomes more complicated is when functions from
different libraries need to be strung together to form a machine learning workflow. `mercury-ml` is "a messenger of the gods"
that enables this to happen. It seeks to break down a Machine Learning project into its typical generic components
(such as `read data`, `transform data`, `fit model`, `evaluate model` etc.) and offers a generic modular structure where
implementations for specific methods and technologies can slot in.

These broken down components can then be chained together into a coherent, easily configurable workflow for fitting,
evaluating and (coming soon!) serving Machine Learning models.


## Components

The package is split firstly into two broad sections:
* `common`: where functions and classes that are commonly useful, regardless of the machine learning engine used, are  found
* Then there are sections that are specific to machine learning engines (currently this includes `h2o` and `keras`).

Within these sections there is a further subdivision into three APIs:
* `providers`: The individual (modular) building-blocks used to build up a workflow
* `containers`: IoC containers over which the desired providers can be fetched.
* `tasks`: Small predefined chunks of work, mostly stringing together a handful of logical steps to be executed by various providers

You can interact with `mercury-ml` via any or all of these APIs. They deliver different levels of abstraction depending
on what you need. You can also easily mix in your own custom providers.

To understand the purpose and function of each individual provider, container and task, please refer to the `mercury-ml`
API documentation.

## Dependencies

Since `mercury-ml` functions as a facilitator for workflows based on various different packages its dependencies will
be determined by which functions are used. The core dependencies have been minimized to only a handful of packages.

##### Core dependencies
* `python>=3.5`
* `pandas`
* `numpy`
* `scikit-learn`

##### Workflows using Keras:
* `tensorflow` or `tensorflow-gpu`
* `keras`
* `Pillow`

##### Workflow using H2O:
* `h2o`

##### Workflows using H2O Sparkling:
* `pyspark`
* `h2o-pysparkling-{spark-version}` (e.g. if you installed `pyspark==2.4`, you should install `h2o-pysparkling-2.4`)

##### Remote storage dependencies:
* AWS S3: `boto3`
* GCP Cloud Storage: `google-cloud-storage`

## Installation

`mercury-ml` can be installed from PyPi with `pip install mercury-ml`. This will also install the core dependencies. 


## Usage


`mercury-ml` aims to offer simplified access to functionality at different levels of abstraction.

Below are four simple examples that each do the same thing: save a Keras model to S3. They do so at different levels
of abstraction:
1. Without using `mercury-ml` (i.e. directly using the underlying dependencies)
2. Using the `providers` API
3. Using the `containers` API
4. Using the `tasks` API (in conjunction with the `containers` API)

Each of these examples
are perfectly valid, though in certain circumstances one may make more sense than the other.

For more complete examples, please see the `examples` directory in this repository.

##### Parameterization:

Let's assume we have the following inputs:

```python
model = ... # assume a fitted Keras model fetched here
filename = "my_model"
local_dir = "./local_models"
extension = ".h5"
remote_dir = "my-bucket/remote-model"
```


##### 1. Example via directly accessing the underlying libraries (i.e. without using `mercury-ml`)

Using the underlying libraries rather than using the `mercury-ml` APIs makes sense when you want the maximum flexibility to
configure how these libraries are used.

```python
import os

# save model
if not os.path.exists(local_dir):
    os.makedirs(local_dir)

filename = filename + extension
local_path = os.path.join(local_dir + "/" + filename)
model.save(local_path)

# copy to s3
import boto3
session = boto3.Session()
s3 = session.resource("s3")

s3_bucket, s3_partial_key = remote_dir.split("/", 1)
s3_key = s3_partial_key + "/" + filename + extension

s3.Object(s3_bucket, s3_key).put(Body=open(local_path, "rb"))
```


##### 2. Example via ``providers``

Using the `providers` API makes the most sense if you want to hardcode the providers you want to use. For example
in the code snippet be, you can only use `model_saving.save_keras_hdf5` and `from_disk.copy_from_disk_to_s3`. If you
want to save the model in a different format, or copy it to a different store you must change your code to do so.

```python
from mercury_ml.keras.providers import model_saving
from mercury_ml.common.providers.artifact_copying import from_disk
import os

# save model
path = model_saving.save_keras_hdf5(model=model,
                                    filename=filename,
                                    local_dir=local_dir,
                                    extension=extension)

# copy to s3
from_disk.copy_from_disk_to_s3(source_dir=local_dir,
                               target_dir=remote_dir,
                               filename=os.path.basename(path))
```


##### 3. Example via ``containers``

Using the `containers` API makes the most sense when you want to steer your workflow via a configuration file. The containers
are just light-weight classes that allow you to access various similar providers from a single location. For example,
the function used above, `model_saving.save_keras_hdf5` can be accessed via a container as `ModelSavers.save_hdf5`. Using the
`getattr` function this can also be accessed as `getattr(ArtifactCopiers, "save_hdf5")` allowing us to easily parameterize
this in a config.

```python
from mercury_ml.keras.containers import ModelSavers
from mercury_ml.common.containers import ArtifactCopiers
import os

config = {
    "save_model": "save_hdf5",
    "copy_model": "copy_from_disk_to_s3"
}

save_model = getattr(ModelSavers, config["save_model"])
copy_from_local_to_remote = getattr(ArtifactCopiers, config["copy_model"])

# save model
path = save_model(model=model,
                  filename=filename,
                  local_dir=local_dir,
                  extension=extension)

# copy to s3
copy_from_local_to_remote(source_dir=local_dir,
                          target_dir=remote_dir,
                          filename=os.path.basename(path)
                          )
```

##### 4. Example via ``tasks`` (in conjunction with `containers`)

Using the tasks API makes sense when you want to use a single function that defines a small workflow that involves more
than one `provider` and requires multiple steps. For example, the `store_model` task below is injected with a `save_model`
and a `copy_from_local_to_remote` provider and proceeds to use these `providers` first to save a model locally and then
to copy it to a remote location (in this example, to S3)

```python
from mercury_ml.common.tasks import store_model
from mercury_ml.keras.containers import ModelSavers
from mercury_ml.common.containers import ArtifactCopiers

save_model = getattr(ModelSavers, config["save_model"])
copy_from_local_to_remote = getattr(ArtifactCopiers, config["copy_model"])

# save model and copy to s3
store_model(save_model=save_model,
            copy_from_local_to_remote=copy_from_local_to_remote,
            model=model,
            filename=filename,
            local_dir=local_dir,
            remote_dir=remote_dir,
            extension=extension
            )
```


## Data in `mercury-ml`

It is worth saying a few words about how `mercury-ml` deals with data as this forms the backbone of how it is able to
facilitate robust machine learning workflows. There are three concepts to understand:
1. `DataWrapper`. An instance of the `DataWrapper` class wraps an underlying structure (for example a `Pandas DataFrame`,
`Spark DataFrame`, `Numpy Array` or `Keras ImageDataGenerator`) into an object that has the following characteristics:
    * It has the attributes `underlying` (which gives you direct access to the data structure that has been wrapped)
    and `field_names`, which is a list with the names of the in the underlying data (`field_names` is not always relevant,
    and may be set to `None`).
    * It has various functions that transform from one DataWrapper to another. For example, `PandasDataWrapper.to_numpy()`
    will yield an instance of `NumpyDataWrapper`
2. `DataSet`. An instance of the `DataSet` class is a container for various instances of `DataWrapper`. A `DataSet` will
typically consist of `DataWrappers` for `full_data`, `index`, `features` and `targets` but this is not predefined. It
also contains some functionality that facilitates the transformation into a `DataSet` with `DataWrappers` of a different
type.
3. `DataBunch`. An instance of the `DataBunch` class is essentially just a container that holds one or more `DataSet`
instances. A `DataBunch` will typically consist of `train`, `valid` and `test` `DataSets`.

### Example usage:

As an example of how this works, let's create a `DataBunch` for a model training that uses Pandas DataFrames as inputs:

```python

import pandas as pd
from mercury_ml.common.providers.data_wrappers.pandas import PandasDataWrapper
from mercury_ml.common.providers.data_set import DataSet
from mercury_ml.common.providers.data_bunch import DataBunch

path_to_input_data = "./example_data_train.csv"

full_data_columns=["ID", "field1_num", "field2_num", "field3_factor", "field4_target"] #the full columns relevant to training
index_columns=["ID"] # the columns that make up the unique index
features_columns=["field1_num", "field2_num", "field3_factor"] # the columns to be used as features when training
targets_columns=["field6_target"] # the columns with the targets to be trained on        

df = pd.read_csv(path_to_input_data, usecols=full_data_columns)

train_data_set = DataSet(data_wrappers_dict={
    "full_data": PandasDataWrapper(underlying=df, field_names=full_data_columns),
    "index": PandasDataWrapper(underlying=df[index_columns], field_names=index_columns),
    "features": PandasDataWrapper(underlying=df[features_columns], field_names=features_columns),
    "targets": PandasDataWrapper(underlying=df[targets_columns], field_names=targets_columns)
})

data_bunch = DataBunch(data_sets_dict={
    "train": train_data_set
})

```

You could also add additional DataSets to the DataBunch, either when initially constructing:

```python
data_bunch = DataBunch(data_sets_dict={
    "train": train_data_set,
    "valid": valid_data_set,
    "test": test_data_set
})
```

Or afterwards via `DataBunch.add_data_set`:

```python
from mercury_ml.common.providers.data_bunch import DataBunch
data_bunch = DataBunch()
data_bunch.add_data_set(data_set_name="train", data_set=train_data_set)
data_bunch.add_data_set(data_set_name="valid", data_set=valid_data_set)
data_bunch.add_data_set(data_set_name="test", data_set=test_data_set)
```
