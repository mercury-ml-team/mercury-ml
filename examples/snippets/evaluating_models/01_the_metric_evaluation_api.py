# In this example, we'll assume that we already have a trained TensorFlow model (we'll simply load a small existing model)
# and that we have a "test" dataset consisting of NumpyDataWrappers. We'll then produce two small evaluations using the
# metric_evaluation api

from mercury_ml.common.source_reading.disk import read_pandas_data_set
from mercury_ml.tensorflow.prediction import predict
# get data_set
data_set = read_pandas_data_set(path="./example_data/test.csv",
                                data_wrappers_params_dict={
                                    "features": ["field1_num", "field2_num", "field3_num"],
                                    "targets": ["field4_target", "field5_target", "field6_target"]
                                },
                                input_format=".csv").transform({"transform_to":"numpy"})

# load model
from tensorflow.keras.models import load_model
model = load_model("./example_data/model.h5")

# predict
data_set.predictions = predict(data_set, model)

# we can now produce evaluations with any function that take a signature (y_true, y_pred) and returns a number, such as:
from mercury_ml.common import CustomMetrics

auc_value = CustomMetrics.evaluate_numpy_auc(y_true = data_set.targets.underlying,
                                             y_pred = data_set.predictions.underlying)
print("AUC: ", auc_value, "\n")


# We can also produce label-specific evaluations with functions that that a signature (y_true, y_pred, labels), such as

from mercury_ml.common import CustomLabelMetrics


auc_dict = CustomLabelMetrics.evaluate_numpy_auc(y_true = data_set.targets.underlying,
                                                 y_pred = data_set.predictions.underlying,
                                                 labels=["A", "B", "C"])
import json
print("AUC by label: ")
print(json.dumps(auc_dict, indent=2))

