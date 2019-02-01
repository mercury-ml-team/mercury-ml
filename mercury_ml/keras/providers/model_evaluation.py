def evaluate(model, data_set):
    """
    Fetches the built-in Keras metrics, when using an array as input
    :param model: A trainied Keras model
    :param data_set: A DataSet with NumpyDataWrappers called "features" and "targets"
    :return: A dictionary of the form {metric_name:metric_value,...}
    """

    metric_values = model.evaluate(x=data_set.features.underlying, y=data_set.targets.underlying)
    metrics = __generate_metric_dict(model.metrics_names, metric_values)

    return metrics

def evaluate_generator(model, data_set):
    """
    Fetches the built-in Keras metrics, when using an generator as input
    :param model: A trainied Keras model
    :param data_set: A DataSet with a KerasIteratorFeaturesDataWrapper called "features" and a KerasIteratorTargetsDataWrapper called "targets"
    :return: A dictionary of the form {metric_name:metric_value,...}
    """

    iterator = data_set.features.underlying
    metric_values = model.evaluate_generator(iterator)
    metrics = __generate_metric_dict(model.metrics_names, metric_values)

    return metrics


def __generate_metric_dict(metric_names, metric_values):

    metrics = {}
    for i in range(len(metric_names)):
        metrics[metric_names[i]] = metric_values[i]

    return metrics