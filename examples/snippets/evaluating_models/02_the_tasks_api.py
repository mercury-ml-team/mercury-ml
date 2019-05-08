# In this example, we'll assume that we already have a trained TensorFlow model (we'll simply load a small existing model)
# and that we have a "test" dataset consisting of NumpyDataWrappers. We'll then run several evaluations using the tasks
# api

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

# calculate metrics
from mercury_ml.common import CustomMetrics
from mercury_ml.common.tasks import evaluate_metrics
custom_metrics = ["evaluate_numpy_micro_auc", "evaluate_numpy_macro_auc"]
custom_metrics_dict = {
    custom_metric_name: getattr(CustomMetrics, custom_metric_name) for custom_metric_name in custom_metrics
}
print("Metric functions")
for metric_name, metric_function in custom_metrics_dict.items():
    print(metric_name, metric_function)
print("")


metrics = evaluate_metrics(data_set, custom_metrics_dict)

import json
print("Metrics:")
print(json.dumps(metrics, indent=2), "\n")


# calculate label metrics
from mercury_ml.common import CustomLabelMetrics
from mercury_ml.common.tasks import evaluate_label_metrics
custom_label_metrics = ["evaluate_numpy_auc", "evaluate_numpy_accuracy"]
custom_label_metrics_dict = {
    custom_label_metric_name: getattr(CustomLabelMetrics, custom_label_metric_name) for custom_label_metric_name in custom_label_metrics
}
print("Label Metric functions")
for metric_name, metric_function in custom_label_metrics_dict.items():
    print(metric_name, metric_function)
print("")

label_metrics = evaluate_label_metrics(data_set, custom_label_metrics_dict)

import json
print("Label Metrics:")
print(json.dumps(label_metrics, indent=2))