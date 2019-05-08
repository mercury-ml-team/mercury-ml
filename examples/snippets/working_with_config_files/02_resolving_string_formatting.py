# Although Mercury-ML does not require you to work with config files, it is encouraged as this allows you to make full
# use of the code abstraction capabilities that are on offer.

# We have included a few small utils that help with dealing with config files. In this example, we'll show how you can
# resolve string formatting within a config file

# Let's first try to read our JSON file using the json library:
# Note the following entries:
#     "model_object_name": "my_model_{model_id}.h5"
#     "filepath": "/some/path/{model_object_name}"
# The {...} values are placeholders for string formatting
import json
with open("./02_resolving_string_formatting_config.json", "r") as f:
    config = json.load(f)
print("Raw input: ")
print(json.dumps(config, indent=2), "\n")


# Now let's use mercury_ml.common.utils.recursively_update_config to replace the placeholders in config["meta_info"]
# with and actual value
from mercury_ml.common.utils import recursively_update_config
recursively_update_config(config["meta_info"], {"model_id": "abc_123"})
print("After updating meta_info: ")
print(json.dumps(config, indent=2), "\n")

# Lastly, let's use config["meta_info"] to update the rest of the config
recursively_update_config(config, config["meta_info"])
print("After updating the entire config: ")
print(json.dumps(config, indent=2), "\n")