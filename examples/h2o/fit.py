# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # H2O workflow

# ## Imports

# +
import sys
import os
sys.path.append(os.path.split(os.path.split(os.getcwd())[0])[0])
config_filepath = os.path.join(os.getcwd(),"config/fit_config_h2o.json")
notebook_filepath = os.path.join(os.getcwd(),"fit.ipynb")
import uuid
import json
import datetime
import getpass

from mercury_ml.common import tasks
from mercury_ml.common import utils
import mercury_ml.common 
import mercury_ml.h2o

# +
#For testing purposes only!

if os.path.isdir("./example_results"):
    import shutil
    shutil.rmtree("./example_results")


# -

# ## Helpers
#
# These functions will help with the flow of this particular notebook

# +
def print_data_bunch(data_bunch):

    for data_set_name, data_set in data_bunch.__dict__.items():
        print("{} <{}>".format(data_set_name, type(data_set).__name__))
        for data_wrapper_name, data_wrapper in data_set.__dict__.items():
            print("  {} <{}>".format(data_wrapper_name, type(data_wrapper).__name__))
        print()
        
def maybe_transform(data_bunch, pre_execution_parameters):
    if pre_execution_parameters:
        return data_bunch.transform(**pre_execution_parameters)
    else:
        return data_bunch
        
def print_dict(d):
    print(json.dumps(d, indent=2))

def get_installed_packages():
    import pip
    try:
        from pip._internal.operations import freeze
    except ImportError:  # pip < 10.0
        from pip.operations import freeze

    packages = []
    for p in freeze.freeze():
        packages.append(p)

    return packages


# -

# ## Config

# #### Load config

config = utils.load_referenced_json_config(config_filepath)

print_dict(config)

# #### Set model_id

session_id = str(uuid.uuid4().hex)

print(session_id)

# #### Update config
#
# The function `utils.recursively_update_config(config, string_formatting_dict)` allows us to use string formatting to replace placeholder strings with acctual values.
#
# for example: 
#
# ```python
# >>> config = {"some_value": "some_string_{some_placeholder}"}
# >>> string_formatting_dict = {"some_placeholder": "ABC"}
# >>> utils.recursively_update_config(config, string_formatting_dict)
# >>> print(config)
# {"some_value": "some_string_ABC}"}
# ```
#
#

# First update `config["meta_info"]`

utils.recursively_update_config(config["meta_info"], {
    "session_id": session_id,
    "model_purpose": config["meta_info"]["model_purpose"],
    "config_filepath": config_filepath,
    "notebook_filepath": notebook_filepath
})

# Then use `config["meta_info"]` to update the rest.

utils.recursively_update_config(config, config["meta_info"])

# ## Session
#
# Create a small dictionary with the session information. This will later be stored as a dictionary artifact with all the key run infomration

session = {
    "time_stamp": datetime.datetime.utcnow().isoformat()[:-3] + "Z",
    "run_by": getpass.getuser(),
    "meta_info": config["meta_info"],
    "installed_packages": get_installed_packages()
}

print("Session info")
print(json.dumps(session, indent=2))


# ## Initialization
#
# These are the functions or classes we will be using in this workflow. We get / instatiate them all at the beginning using parameters under `config["initialization"]`.
#
# Here we use mainly use `getattr` to fetch them via the `containers` module based on a string input in the config file. Providers could however also be fetched directly. The following three methods are all equivalent:
#
# ```python
# # 1. (what we are using in this notebook)
# import mercury_ml.common
# source_reader=getattr(mercury_ml.common.SourceReaders, "read_pandas_data_set")
#
# # 2. 
# import mercury_ml.common
# source_reader=mercury_ml.common.SourceReaders.read_pandas_data_set
#
# # 3.
# from mercury_ml.common.source_reading import read_pandas_data_set
# source_reader=read_pandas_data_set
# ```
#

# ### Helpers
#
# These helper functions will create instantiate class providers (`create_and_log`) or fetch function providers (`get_and_log`) based on the parameters provided

# +
def create_and_log(container, class_name, params):
    provider = getattr(container, class_name)(**params)
    print("{}.{}".format(container.__name__, class_name))
    print("params: ", json.dumps(params, indent=2))
    return provider

def get_and_log(container, function_name):
    provider = getattr(container, function_name)
    print("{}.{}".format(container.__name__, function_name))
    return provider


# -

# ### Common
#
# These are providers that are universally relevant, regardless of which Machine Learning engine is used.

# a function for storing dictionary artifacts to local disk
store_artifact_locally = get_and_log(mercury_ml.common.LocalArtifactStorers,
                                     config["init"]["store_artifact_locally"]["name"])

# a function for storing data-frame-like artifacts to local disk
store_prediction_artifact_locally = get_and_log(mercury_ml.common.LocalArtifactStorers,
                                                config["init"]["store_prediction_artifact_locally"]["name"])

# a function for copy artifacts from local disk to a remote store
copy_from_local_to_remote = get_and_log(mercury_ml.common.ArtifactCopiers, config["init"]["copy_from_local_to_remote"]["name"])

# a function for reading source data. When called it will return an instance of type DataBunch 
read_source_data_set = get_and_log(mercury_ml.common.SourceReaders, config["init"]["read_source_data"]["name"])

# a dictionary of functions that calculate custom metrics
custom_metrics_dict = {
    custom_metric_name: get_and_log(mercury_ml.common.CustomMetrics, custom_metric_name) for custom_metric_name in config["init"]["custom_metrics"]["names"]
}


# a dictionary of functions that calculate custom label metrics
custom_label_metrics_dict = {
    custom_label_metric_name: get_and_log(mercury_ml.common.CustomLabelMetrics, custom_label_metric_name) for custom_label_metric_name in config["init"]["custom_label_metrics"]["names"]
}


# ### H2O

# a function to initiate the h2o (or h2o sparkling) session
initiate_session = get_and_log(mercury_ml.h2o.SessionInitiators, config["init"]["initiate_session"]["name"])

# fetch a built-in h2o model
model = get_and_log(mercury_ml.h2o.ModelDefinitions, 
                    config["init"]["model_definition"]["name"])(**config["init"]["model_definition"]["params"])

# a function that fits an h2o model
fit = get_and_log(mercury_ml.h2o.ModelFitters, config["init"]["fit"]["name"])

# a dictionary of functions that save h2o models in various formats
save_model_dict = {
    save_model_function_name: get_and_log(mercury_ml.h2o.ModelSavers, save_model_function_name) for save_model_function_name in config["init"]["save_model"]["names"]
}


# a function that generates metrics from an h2o model
evaluate = get_and_log(mercury_ml.h2o.ModelEvaluators, config["init"]["evaluate"]["name"])

# a function that generates metrics from an h2o model
evaluate_threshold_metrics = get_and_log(mercury_ml.h2o.ModelEvaluators, config["init"]["evaluate_threshold_metrics"]["name"])

# a function that produces predictions using an h2o model
predict = get_and_log(mercury_ml.h2o.PredictionFunctions, config["init"]["predict"]["name"])

# ## Execution
#
# Here we use the providers defined above to execute various tasks

# ### Save (formatted) config

tasks.store_artifacts(store_artifact_locally, copy_from_local_to_remote, config,
                      **config["exec"]["save_formatted_config"]["params"])

print("Config stored with following parameters")
print_dict(config["exec"]["save_formatted_config"]["params"])

# ### Save Session

# ##### Save session info

tasks.store_artifacts(store_artifact_locally, copy_from_local_to_remote, session,
                                 **config["exec"]["save_session"]["params"])

print("Session dictionary stored with following parameters")
print_dict(config["exec"]["save_session"]["params"])

# ##### Save session artifacts

for artifact_dict in config["exec"]["save_session_artifacts"]["artifacts"]:
    
    artifact_dir=os.path.dirname(artifact_dict["artifact_path"]) 
    artifact_filename=os.path.basename(artifact_dict["artifact_path"])
    
    # save to local artifact store
    mercury_ml.common.ArtifactCopiers.copy_from_disk_to_disk(
        source_dir=artifact_dir,
        target_dir=artifact_dict["local_dir"],
        filename=artifact_filename,
        overwrite=False,
        delete_source=False)

    # copy to remote artifact store
    copy_from_local_to_remote(source_dir=artifact_dict["local_dir"],
                              target_dir=artifact_dict["remote_dir"],
                              filename=artifact_filename,
                              overwrite=False,
                              delete_source=False)


print("Session artifacts stored with following parameters")
print_dict(config["exec"]["save_session_artifacts"])

# ### Start H2O

initiate_session(**config["exec"]["initiate_session"]["params"])

# ### Get source data

data_bunch_source = tasks.read_train_valid_test_data_bunch(read_source_data_set,**config["exec"]["read_source_data"]["params"] )
print("Source data read using following parameters: \n")
print_dict(config["exec"]["read_source_data"]["params"])

print("Read data_bunch consists of: \n")
print_data_bunch(data_bunch_source)

# ### Fit model

# ##### Transform data

# +
data_bunch_fit = maybe_transform(data_bunch_source, config["exec"]["fit"].get("pre_execution_transformation"))

print("Data transformed with following parameters: \n")
print_dict(config["exec"]["fit"].get("pre_execution_transformation"))
# -

print("Transformed data_bunch consists of: \n")
print_data_bunch(data_bunch_fit)

# ##### Perform fitting

model = fit(model = model,
            data_bunch = data_bunch_fit,
            **config["exec"]["fit"]["params"])

# ### Save model

for model_format, save_model in save_model_dict.items():
    
    tasks.store_model(save_model=save_model,
                      model=model,
                      copy_from_local_to_remote = copy_from_local_to_remote,
                      **config["exec"]["save_model"][model_format]
                      )

print("Model saved with following paramters: \n")
print_dict(config["exec"]["save_model"])

# ### Evaluate metrics

# ##### Transform data

# +
data_bunch_metrics = maybe_transform(data_bunch_fit, config["exec"]["evaluate"].get("pre_execution_transformation"))

print("Data transformed with following parameters: \n")
print_dict(config["exec"]["evaluate"].get("pre_execution_transformation"))
# -

print("Transformed data_bunch consists of: \n")
print_data_bunch(data_bunch_metrics)

# ##### Calculate metrics

metrics = {}
for data_set_name in config["exec"]["evaluate"]["data_set_names"]:
    data_set = getattr(data_bunch_metrics, data_set_name)
    metrics[data_set_name] = evaluate(model, data_set, data_set_name, **config["exec"]["evaluate"]["params"])

print("Resulting metrics: \n")
print_dict(metrics)

# ##### Calculate metrics

threshold_metrics = {}
for data_set_name in config["exec"]["evaluate"]["data_set_names"]:
    data_set = getattr(data_bunch_metrics, data_set_name)
    threshold_metrics[data_set_name] = evaluate_threshold_metrics(model, data_set, data_set_name,
                                                                  **config["exec"]["evaluate_threshold_metrics"]["params"])

print("Resulting metrics: \n")
print_dict(threshold_metrics)

# ### Save metrics

for data_set_name, params in config["exec"]["save_metrics"]["data_sets"].items():
    tasks.store_artifacts(store_artifact_locally, copy_from_local_to_remote, metrics[data_set_name], **params)

for data_set_name, params in config["exec"]["save_threshold_metrics"]["data_sets"].items():
    tasks.store_artifacts(store_artifact_locally, copy_from_local_to_remote, metrics[data_set_name], **params)

# ### Predict

# ##### Transform data

# +
data_bunch_predict = maybe_transform(data_bunch_metrics, config["exec"]["predict"].get("pre_execution_transformation"))
    
print("Data transformed with following parameters: \n")
print_dict(config["exec"]["predict"].get("pre_execution_transformation"))
# -

print("Transformed data_bunch consists of: \n")
print_data_bunch(data_bunch_predict)

# ##### Perform prediction

for data_set_name in config["exec"]["predict"]["data_set_names"]:
    data_set = getattr(data_bunch_predict, data_set_name)
    data_set.predictions = predict(model=model, data_set=data_set, **config["exec"]["predict"]["params"])

print("Data predicted with following parameters: \n")
print_dict(config["exec"]["predict"].get("params"))


data_bunch_predict.test.predictions.underlying

# ### Evaluate custom metrics

# ##### Transform data

data_bunch_custom_metrics = maybe_transform(data_bunch_predict, 
                                            config["exec"]["evaluate_custom_metrics"].get("pre_execution_transformation"))

print("Data transformed with following parameters: \n")
print_dict(config["exec"]["evaluate_custom_metrics"].get("pre_execution_transformation"))

print("Transformed data_bunch consists of: \n")
print_data_bunch(data_bunch_custom_metrics)


# ##### Calculate custom metrics
#

custom_metrics = {}
for data_set_name in config["exec"]["evaluate_custom_metrics"]["data_set_names"]:
    data_set = getattr(data_bunch_custom_metrics, data_set_name)
    custom_metrics[data_set_name]  = tasks.evaluate_metrics(data_set, custom_metrics_dict)


print("Resulting custom metrics: \n")
print_dict(custom_metrics)


# ##### Calculate custom label metrics

custom_label_metrics = {}
for data_set_name in config["exec"]["evaluate_custom_label_metrics"]["data_set_names"]:
    data_set = getattr(data_bunch_custom_metrics, data_set_name)
    custom_label_metrics[data_set_name] = tasks.evaluate_label_metrics(data_set, custom_label_metrics_dict)

print("Resulting custom label metrics: \n")
print_dict(custom_label_metrics)

for data_set_name, params in config["exec"]["save_custom_metrics"]["data_sets"].items():
    tasks.store_artifacts(store_artifact_locally, copy_from_local_to_remote,
                          custom_metrics[data_set_name], **params)

print("Custom metrics saved with following parameters: \n")
print_dict(config["exec"]["save_custom_metrics"])

for data_set_name, params in config["exec"]["save_custom_label_metrics"]["data_sets"].items():
    tasks.store_artifacts(store_artifact_locally, copy_from_local_to_remote,
                          custom_label_metrics[data_set_name], **params)

print("Custom label metrics saved with following parameters: \n")
print_dict(config["exec"]["save_custom_label_metrics"])

# ### Prepare predictions for storage

# ##### Transform data

data_bunch_prediction_preparation = maybe_transform(data_bunch_predict, 
                                                    config["exec"]["prepare_predictions_for_storage"].get("pre_execution_transformation"))

print("Transformed data_bunch consists of: \n")
print_data_bunch(data_bunch_prediction_preparation)

# ##### Prepare predictions and targets

for data_set_name in config["exec"]["prepare_predictions_for_storage"]["data_set_names"]:
    data_set = getattr(data_bunch_prediction_preparation, data_set_name)
    data_set.add_data_wrapper_via_concatenate(**config["exec"]["prepare_predictions_for_storage"]["params"]["predictions"])
    data_set.add_data_wrapper_via_concatenate(**config["exec"]["prepare_predictions_for_storage"]["params"]["targets"])

print_data_bunch(data_bunch_prediction_preparation)

# ### Save predictions

# ##### Transform data

data_bunch_prediction_storage = maybe_transform(data_bunch_prediction_preparation, 
                                                config["exec"]["save_predictions"].get("pre_execution_transformation"))

print("Transformed data_bunch consists of: \n")
print_data_bunch(data_bunch_prediction_storage)

# ##### Save predictions

for data_set_name, data_set_params in config["exec"]["save_predictions"]["data_sets"].items():
    data_set = getattr(data_bunch_prediction_storage, data_set_name)
    data_wrapper = getattr(data_set, data_set_params["data_wrapper_name"])
    
    data_to_store = data_wrapper.underlying
   
    tasks.store_artifacts(store_prediction_artifact_locally, copy_from_local_to_remote,
                          data_to_store, **data_set_params["params"])

print("Predictions saved with following parameters: \n")
print_dict(config["exec"]["save_predictions"])

# ##### Save targets

for data_set_name, data_set_params in config["exec"]["save_targets"]["data_sets"].items():
    data_set = getattr(data_bunch_prediction_storage, data_set_name)
    data_wrapper = getattr(data_set, data_set_params["data_wrapper_name"])
    
    data_to_store = data_wrapper.underlying
   
    tasks.store_artifacts(store_prediction_artifact_locally, copy_from_local_to_remote,
                          data_to_store, **data_set_params["params"])

print("Targets saved with following parameters: \n")
print_dict(config["exec"]["save_targets"])


