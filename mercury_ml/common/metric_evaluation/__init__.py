import mercury_ml

# metric evaluation
def evaluate_metrics(data_set, custom_metric_names):
    y_true = data_set.targets.underlying
    y_pred = data_set.predictions.underlying

    custom_metrics = {}

    for custom_metric_name in custom_metric_names:
        evaluate_custom_metric = getattr(mercury_ml.common.CustomMetrics, custom_metric_name)
        custom_metric_value = evaluate_custom_metric(y_true, y_pred)
        custom_metrics[custom_metric_name] = custom_metric_value

    return custom_metrics


def evaluate_label_metrics(data_set, custom_label_metric_names):
    y_true = data_set.targets.underlying
    y_pred = data_set.predictions.underlying

    custom_label_metrics = {}

    for custom_label_metric_name in custom_label_metric_names:
        evaluate_custom_label_metric = getattr(mercury_ml.common.CustomLabelMetrics, custom_label_metric_name)
        custom_label_metric_dict = evaluate_custom_label_metric(y_true, y_pred)
        custom_label_metrics.update(custom_label_metric_dict)

    return custom_label_metrics