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

# # Tensorflow Workflow - minimal example
#
# The following is intended as an example that simplifies the number of steps, and uses the Mercury-ML function directly, rather than referencing and resolving function via a config file. It is not meant as a recommended "best practice" (for that, please see fit.ipynb), but instead is meant as a way to understand what is happening under the hood. Before going through this example, consider also first reviewing the small example scripts found under /examples/snippets/

# + {"pycharm": {}, "cell_type": "markdown"}
# ## Imports

# + {"pycharm": {"is_executing": false}}
import sys
import os
sys.path.append(os.path.split(os.path.split(os.getcwd())[0])[0])
import uuid
import json
import json_tricks
import copy

from mercury_ml.common import tasks, utils, SourceReaders, ArtifactCopiers, CustomMetrics, CustomLabelMetrics
from mercury_ml.tensorflow import ModelDefinitions, CallBacks, ModelCompilers, OptimizerFetchers, ModelFitters, \
LossFunctionFetchers, ModelSavers, ModelEvaluators, PredictionFunctions
# -


# ## Session
#
# We assign a unique session ID that will be used throughout to store artifacts produced during the run

# + {"pycharm": {"is_executing": false}}
session_id = str(uuid.uuid4().hex[:8])
# -

print(session_id)

# + {"pycharm": {}, "cell_type": "markdown"}
# ## Source
# Here we set the parameters needed to reading the source data, and then proceed to use the task "read_train_valid_test_data_bunch" from the mercury_ml.common.tasks API
#

# + {"pycharm": {"is_executing": false}}
input_data_folder= "images_456"

source_params ={
    "train":{
          "generator_params": {
            "data_format": "channels_last",
            "rescale": 1./255,
          },
          "iterator_params": {
            "directory": "./example_data/{}/train".format(input_data_folder),
            "batch_size": 2,
            "class_mode": "categorical",
            "color_mode": "rgb",
            "shuffle": True,
            "target_size": [10, 10]
          }
        },
    "valid": {
          "generator_params": {
            "data_format": "channels_last",
            "rescale": 1./255,
          },
          "iterator_params": {
            "directory": "./example_data/{}/valid".format(input_data_folder),
            "batch_size": 2,
            "class_mode": "categorical",
            "color_mode": "rgb",
            "shuffle": False,
            "target_size": [10, 10]
          }
        },
    "test":{
          "generator_params": {
            "data_format": "channels_last",
            "rescale": 1./255,
          },
          "iterator_params": {
            "directory": "./example_data/{}/test".format(input_data_folder),
            "batch_size": 2,
            "class_mode": "categorical",
            "color_mode": "rgb",
            "shuffle": False,
            "target_size": [10, 10]
          }
        }
}


# + {"pycharm": {"is_executing": false, "metadata": false, "name": "#%%\n"}}
read_source_data_set = SourceReaders.read_disk_keras_single_input_iterator

data_bunch_fit = tasks.read_train_valid_test_data_bunch(read_source_data_set,
                                                        source_params["train"],
                                                        source_params["valid"],
                                                        source_params["test"])
# -

print(data_bunch_fit)

# + {"pycharm": {}, "cell_type": "markdown"}
# ## Define model
#
# Here we define the model we will use (in this case, we use the small function in function "define_conv_simple", but any valid tensorflow.keras model would work here)

# + {"pycharm": {"metadata": false, "name": "#%%\n"}}
model = ModelDefinitions.define_conv_simple(
    input_size = [10, 10],
    nb_classes=2,
    final_activation="softmax",
    dropout_rate=0.1)
# -

# Get optimizer and loss function

# + {"pycharm": {}}
optimizer_params = {
        "optimizer_name": "Adam",
        "optimizer_params": {
          "lr": 0.001
        }
      }
optimizer = OptimizerFetchers.get_keras_optimizer(**optimizer_params)
loss_function = LossFunctionFetchers.get_keras_loss("categorical_crossentropy")

# + {"pycharm": {"metadata": false, "name": "#%% md\n"}, "cell_type": "markdown"}
# Compile the model

# + {"pycharm": {"metadata": false, "name": "#%%\n"}}
model = ModelCompilers.compile_model(
    model=model,
    optimizer=optimizer,
    loss=loss_function,
    metrics=["acc"])
# -

model.summary()

# + {"pycharm": {}, "cell_type": "markdown"}
# ## Define the callsbacks for training process

# + {"pycharm": {}}
callback_params_early_st = {
          "patience": 2,
          "monitor": "val_loss",
          "min_delta": 0.001
        }
callback_params_early_model_ch = {
    "filepath": "./example_results/local/"+session_id+"/model_checkpoint/last_best_model.h5",
    "save_best_only": True
}

callbacks = [CallBacks.early_stopping(callback_params_early_st),
             CallBacks.model_checkpoint(callback_params_early_model_ch)]


# + {"pycharm": {"name": "#%% md\n"}, "cell_type": "markdown"}
# ## Fit the model

# + {"pycharm": {"metadata": false, "name": "#%%\n"}}
model = ModelFitters.fit_generator(
    model = model,
    data_bunch = data_bunch_fit,
    callbacks = callbacks,
    epochs=5)

# + {"pycharm": {}, "cell_type": "markdown"}
# ## Save the model

# + {"pycharm": {"name": "#%% md\n"}, "cell_type": "markdown"}
# First we specifiy the model savers (i.e. in what format should be model be saved, and which paramters should be used for the storage)

# + {"pycharm": {"metadata": false, "name": "#%%\n"}}
save_model_dict = {
    "save_hdf5":ModelSavers.save_hdf5,
    "save_tensorflow_saved_model":ModelSavers.save_tensorflow_saved_model
}

model_local_dir ="./example_results/local/"+session_id+"/models"
model_remote_dir = "./example_results/remote/"+session_id+"/models"
model_object_name= "fit_example__"+session_id
save_model_params = {
      "save_hdf5": {
        "local_dir": model_local_dir,
        "remote_dir": model_remote_dir,
        "filename": model_object_name+"__hdf5",
        "extension": ".h5",
        "overwrite_remote": True
      },
      "save_tensorflow_saved_model": {
        "local_dir": model_local_dir,
        "remote_dir": model_remote_dir,
        "filename": model_object_name+"__tf_serving_predict",
      }
    }

# + {"pycharm": {"metadata": false, "name": "#%% md\n"}, "cell_type": "markdown"}
# Then we use the "tasks" API to store the model in each of the provided formats, as well as copying the model to a "remote" location (in this example we simply copy to another local folder, but this would normally be used in combination with S3, GCS, HDFS etc)

# + {"pycharm": {"metadata": false, "name": "#%%\n"}}
for model_format, save_model in save_model_dict.items():
    
    tasks.store_model(save_model=save_model,
                      model=model,
                      copy_from_local_to_remote = ArtifactCopiers.copy_from_disk_to_disk,
                      **save_model_params[model_format]
                      )


# + {"pycharm": {}, "cell_type": "markdown"}
# ## Evaluate model
# -

# First we use the built-in evaluate functions that are available in tensorflow.keras

# + {"pycharm": {"metadata": false, "name": "#%%\n"}}
data_bunch_fit.test.predictions = PredictionFunctions.predict_generator(model=model, data_set=data_bunch_fit.test)
metrics =  ModelEvaluators.evaluate_generator(model, data_bunch_fit.test)
print(json_tricks.dumps(metrics, indent=2))    

# + {"pycharm": {"metadata": false, "name": "#%% md\n"}, "cell_type": "markdown"}
# Next we use the mercury_ml.common.tasks API to produce cutom metric evaluations. For these, we will evaluate metrics based on Numpy calculations, and therefore need to first transform our data_bunch to Numpy:
# -

transformation_params = {
        "data_set_names": ["test"],
        "params": {
          "transform_to": "numpy",
          "data_wrapper_params": {
            "predictions": {},
            "index": {},
            "targets": {}
          }
        }
      }
data_bunch_metric = data_bunch_fit.transform(**transformation_params)

print(data_bunch_metric)

custom_label_metrics_dict = {
    "evaluate_numpy_accuracy":CustomLabelMetrics.evaluate_numpy_accuracy,
    "evaluate_numpy_auc":CustomLabelMetrics.evaluate_numpy_auc
}

# + {"pycharm": {"metadata": false, "name": "#%%\n"}}
custom_label_metrics = tasks.evaluate_label_metrics(data_bunch_metric.test, custom_label_metrics_dict)
print(json_tricks.dumps(custom_label_metrics, indent=2))
