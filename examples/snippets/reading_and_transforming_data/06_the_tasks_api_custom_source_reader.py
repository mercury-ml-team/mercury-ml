# In this example we will repeat what was done in example number 5, but instead of using SourceReaders.read_disk_pandas we'll
# use a custom reader. We would however like to maintain the ability to configure everything via the external config file,
# So we'll be registering the custom reader as an alias under SourceReaders

# read in the config file
import json
with open("./06_the_tasks_api_custom_source_reader_config.json", "r") as f:
    config = json.load(f)

# with open("./05_the_tasks_api_config.json", "r") as f:
#     config = json.load(f)

from mercury_ml.common.tasks import read_data_bunch
from mercury_ml.common.source_reading.disk import read_pandas_data_set
from mercury_ml.common import SourceReaders

# register custom reader
def read_numpy_data_set(path, data_wrappers_params_dict, input_format):
    # this is a rather arbitrary example, but for the sake of simplicity, let's assume that our custom reader simply
    # wants to reuse the "read_pandas_data_set" source reader, and then convert the result to Numpy
    return read_pandas_data_set(path, data_wrappers_params_dict, input_format).transform({"transform_to":"numpy"})

# We register this under SourceReaders in order to be able to resolve the reader to use via the config file
# in the config, we'll need to name  "read_data_set" exactly as we name the alias here, i.e. as "read_disk_numpy"
SourceReaders.read_disk_numpy = read_numpy_data_set

# resolve the source reader to use
read_data_set = getattr(SourceReaders, config["read_data_set"])
print(read_data_set, "\n")

# read in the data bunch
data_bunch = read_data_bunch(read_data_set=read_data_set,
                             params_dict=config["params_dict"])
print(data_bunch)

# Try using the "./05_the_tasks_api_config.json" input config with this set of code. What happens? Does it still work?
