import math

def evaluate(model, data_set, dataset_name, metric_names=None):
    """
    Generates the train, valid or test performance metrics of a trained model

    :param model: an H2O model object
    :param data_set: A DataSet with H2ODataWrappers or H2OSparklingDataWrappers as "full_data" attribute
    :param dataset_name: "train", "valid" or "test", indicating the data being evaluated on
    :return: Dictionary of the form {"metric_name":"metric_value"}
    """

    h2o_df = data_set.full_data.underlying

    if not metric_names:
        metric_names = [
            "logloss", "mse", "mae", "mean_residual_deviance", "rmse", "rmsle", "residual_deviance",
            "residual_degrees_of_freedom", "null_deviance", "null_degrees_of_freedom", "aic", "auc", "gini", "r2"
        ]

    # http://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/_modules/h2o/model/model_base.html
    if dataset_name == "train":
        performance = model.model_performance(test_data=None, train=True, valid=False)
    elif dataset_name == "valid":
        performance = model.model_performance(test_data=None, train=False, valid=True)
    elif dataset_name == "test":
        performance = model.model_performance(test_data=h2o_df, train=False, valid=False)
    else:
        raise ValueError(
            "Invalid dataset type. Expecting 'train', 'valid' or 'test'")

    metrics = {}
    for metric_name in metric_names:

        # get all metrics. Ones that have not been calculated are return as None
        metrics[metric_name] = _try_param(getattr(performance, metric_name))

        # remove metrics that have no present calculation
        if metrics[metric_name] is None:
            del metrics[metric_name]

    return metrics

def evaluate_threshold_metrics(model, data_set, dataset_name, threshold_metric_names=None):
    """
    Generates the train, valid or test performance metrics of a trained model

    :param model: an H2O model object
    :param data_set: A DataSet with H2ODataWrappers or H2OSparklingDataWrappers as "full_data" attribute
    :param dataset_name: "train", "valid" or "test", indicating the data being evaluated on
    :return: Dictionary of the form {"metric_name":"metric_value"}
    """

    if not threshold_metric_names:
        #TODO must filter on threshold_metric_names
        threshold_metric_names= [
            "f1", "f2", "f0point5", "accuracy", "precision", "recall", "specificity", "absolute_mcc",
            "min_per_class_accuracy","mean_per_class_accuracy", "tns", "fns", "fps", "tps", "tnr", "fnr",
            "fpr", "tpr"
        ]


    h2o_df = data_set.full_data.underlying

    if dataset_name == "train":
        performance = model.model_performance(
            test_data=None, train=True, valid=False)
    elif dataset_name == "valid":
        performance = model.model_performance(
            test_data=None, train=False, valid=True)
    elif dataset_name == "test":
        performance = model.model_performance(
            test_data=h2o_df, train=False, valid=False)
    else:
        raise ValueError(
            "Invalid dataset type. Expecting 'train', 'valid' or 'test'")

    if performance._metric_json.get("thresholds_and_metric_scores"):
        threshold_metrics = performance._metric_json["thresholds_and_metric_scores"].as_data_frame().to_dict()
    else:
        threshold_metrics = {}
    return threshold_metrics










def _try_param(get_param):
    """
    Tries to execute the function "get_param" and returns its value. If this fails it returns None

    :param get_param: A function such as "logloss", "mse" etc. that is used to fetch the standard H2O metics
    :return:
    """

    try:
        if math.isnan(get_param()):
            return -9999.0000
        else:
            return get_param()
    except Exception:
        return None