# As a further level of abstraction, the "tasks" API can be used to make the code used in the previous four examples
# cleaner and more generic. We will also combine this with the "aliases" API. This will allow us to place all of the
# complexity in a single config file

# read in the config file
import json
with open("./05_the_tasks_api_config.json", "r") as f:
    config = json.load(f)

from mercury_ml.common.tasks import read_data_bunch
from mercury_ml.common import SourceReaders

# resolve the source reader to use
read_data_set = getattr(SourceReaders, config["read_data_set"]) # this resolves to SourceReaders.read_data_set
print(read_data_set, "\n")

# read in the data bunch
data_bunch = read_data_bunch(read_data_set=read_data_set,
                             params_dict=config["params_dict"])
print(data_bunch)


# Note that the "read_data_bunch" function is completely abstracted. You can use any source reader found in
# mercury_ml.common.source_reading (which you can also access via its alias in mercury_ml.common.SourceReaders) as long as
# you pass the appropriate "params_dict" to work with that reader. You can of course also define your own source readers
# to pass to "read_data_bunch". In the next example we'll show how this could be done