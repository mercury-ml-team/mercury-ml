
# coding: utf-8

# ## Imports

# In[ ]:


import sys
import os
sys.path.append(os.path.split(os.path.split(os.getcwd())[0])[0])
import uuid
import json
import copy

from mercury_ml.common import tasks
from mercury_ml.common import utils
from mercury_ml.common import containers as common_containers
from mercury_ml.keras import containers as keras_containers


# In[ ]:


session_id = str(uuid.uuid4().hex)


# ### load config file and update placeholder

# In[ ]:


data_bunch_name= "images_456"


# ## Source
# First set the parameter for the source

# In[ ]:


read_source_train_param = {
          "generator_params": {
            "channel_shift_range": 0.0,
            "data_format": "channels_last",
            "featurewise_center": False,
            "featurewise_std_normalization": False,
            "fill_mode": "nearest",
            "height_shift_range": 0.1,
            "horizontal_flip": True,
            "rescale": 0.00392156862745098,
            "rotation_range": 0.2,
            "samplewise_center": True,
            "samplewise_std_normalization": True,
            "shear_range": 0.1,
            "vertical_flip": True,
            "width_shift_range": 0.1,
            "zca_epsilon": 1e-6,
            "zca_whitening": False,
            "zoom_range": 0.1
          },
          "iterator_params": {
            "directory": "./example_data/"+data_bunch_name+"/train",
            "batch_size": 2,
            "class_mode": "categorical",
            "color_mode": "rgb",
            "seed": 12345,
            "shuffle": True,
            "target_size": [
              10,
              10
            ]
          }
        }

read_source_data_parm_valid = copy.deepcopy(read_source_train_param)
read_source_dara_parm_test =  copy.deepcopy(read_source_train_param)
read_source_data_parm_valid["iterator_params"]["shuffle"]= False
read_source_dara_parm_test["iterator_params"]["shuffle"]= False
read_source_data_parm_valid["iterator_params"]["directory"]= "./example_data/"+data_bunch_name+"/valid"
read_source_dara_parm_test["iterator_params"]["directory"]= "./example_data/"+data_bunch_name+"/test"


# In[ ]:


read_source_data_set = common_containers.SourceReaders.read_disk_keras_single_input_iterator
data_bunch_fit = tasks.read_train_valid_test_data_bunch(read_source_data_set,
                                                        read_source_train_param,
                                                        read_source_data_parm_valid,
                                                        read_source_dara_parm_test)


# ## Model and Training

# Set parameter and load optimizer and loss function

# In[ ]:


optimizer_parm = {
        "optimizer_name": "adam",
        "optimizer_params": {
          "lr": 0.001
        }
      }
optimizer =keras_containers.OptimizerFetchers.get_keras_optimizer(**optimizer_parm)
loss_function = keras_containers.LossFunctionFetchers.get_keras_loss("categorical_crossentropy")


# specify model

# In[ ]:


model = keras_containers.ModelDefinitions.define_conv_simple(
    input_size = [10, 10],
    nb_classes=2,
    final_activation="softmax",
    dropout_rate=0.1)


# compile model

# In[ ]:


model = keras_containers.ModelCompilers.compile_model(model=model,
                      optimizer=optimizer,
                      loss=loss_function,
                      metrics=["acc"])


# ### Fit the model

# Callbacks for monitoring training process

# In[ ]:


callback_params_early_st = {
          "patience": 2,
          "monitor": "val_loss",
          "min_delta": 0.001
        }
callback_params_early_model_ch = {
    "filepath": "./example_results/local/"+session_id+"/model_checkpoint/last_best_model.h5",
    "save_best_only": True
}
callbacks = [keras_containers.CallBacks.early_stopping(callback_params_early_st),
             keras_containers.CallBacks.model_checkpoint(callback_params_early_model_ch)]


# Train the model

# In[ ]:


model = keras_containers.ModelFitters.fit_generator(model = model,
            data_bunch = data_bunch_fit,
            callbacks = callbacks,
            epochs=5)


# #### Save the model

# specify model savers

# In[ ]:


save_model_dict = {"save_hdf5":keras_containers.ModelSavers.save_hdf5,
"save_tensorflow_serving_predict_signature_def":keras_containers.ModelSavers.save_tensorflow_serving_predict_signature_def}


model_local_dir ="./example_results/local/"+session_id+"/models"
model_remote_dir = "./example_results/remote/"+session_id+"/models"
model_object_name= "fit_example__"+session_id
save_model_parm = {
      "save_hdf5": {
        "local_dir": model_local_dir,
        "remote_dir": model_remote_dir,
        "filename": model_object_name+"__hdf5",
        "extension": ".h5",
        "overwrite_remote": True
      },
      "save_tensorflow_serving_predict_signature_def": {
        "local_dir": model_local_dir,
        "remote_dir": model_remote_dir,
        "filename": model_object_name+"__tf_serving_predict",
        "temp_base_dir": "c:/tf_serving/_tmp_model/"+model_object_name+"__tf_serving_predict",
        "extension": ".zip",
        "overwrite_remote": True,
        "do_save_labels_txt": True,
        "input_name": "input",
        "output_name": "output",
        "labels_list": ["cat","dog"]
      }
    }


# save model

# In[ ]:


for model_format, save_model in save_model_dict.items():
    
    tasks.store_model(save_model=save_model,
                      model=model,
                      copy_from_local_to_remote = common_containers.ArtifactCopiers.copy_from_disk_to_disk,#get_and_log(common_containers.ArtifactCopiers, config["init"]["copy_from_local_to_remote"]["name"]),
                      **save_model_parm[model_format]
                      )


# #### Evaluate model

# In[ ]:


evaluate = keras_containers.ModelEvaluators.evaluate_generator
predict = keras_containers.PredictionFunctions.predict_generator

custom_label_metrics_dict = {"evaluate_numpy_accuracy":common_containers.CustomLabelMetrics.evaluate_numpy_accuracy,
        "evaluate_numpy_confusion_matrix":common_containers.CustomLabelMetrics.evaluate_numpy_confusion_matrix}


# In[ ]:


data_bunch_fit.test.predictions = predict(model=model, data_set=data_bunch_fit.test)


# In[ ]:


result = evaluate(model, data_bunch_fit.test)
print(json.dumps(result, indent=2))


# transform test images with numpy

# In[ ]:


transformation_param = {
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
data_bunch_metric = data_bunch_fit.transform(**transformation_param)


# In[ ]:


confMat = tasks.evaluate_label_metrics(data_bunch_metric.test, custom_label_metrics_dict)
print(json.dumps(confMat, indent=2))

