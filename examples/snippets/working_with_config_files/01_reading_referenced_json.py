# Although Mercury-ML does not require you to work with config files, it is encouraged as this allows you to make full
# use of the code abstraction capabilities that are on offer.

# We have included a few small utils that help with dealing with config files. In this example, we'll show how you can
# create and read from a referenced JSON file

# Let's first try to read our JSON file using the json library:
# Note the following entry: "filename": {"$ref": "#meta_info/model_object_name"}.
import json
with open("./01_reading_referenced_json_config.json", "r") as f:
    config = json.load(f)
print(json.dumps(config, indent=2), "\n")

# Now let's use mercury_ml.common.utils.load_referenced_json_config (which builds on the jsonref library)
# Note the following entry: "filename": "my_model_123.h5". The line {"$ref": "#meta_info/model_object_name"} has been
# replaced by the value in meta_info/model_object_name.
from mercury_ml.common.utils import load_referenced_json_config
config = load_referenced_json_config("./01_reading_referenced_json_config.json")
print(json.dumps(config, indent=2))

