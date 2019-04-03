from sklearn.metrics import roc_auc_score


def evaluate_macro_auc(y_true, y_pred):
    """
    Calculates a the marco AUC for the (potentially multi-target) observations vs the predicted probabilities

    :param array y_true: The observed targets
    :param array y_pred: The prediction probabilities
    :return: A dictionary of the form {metric_name:metric_value}
    """

    return roc_auc_score(y_true=y_true.astype("float"), y_score=y_pred.astype("float"), average="macro")

def evaluate_micro_auc(y_true, y_pred):
    """
    Calculates a the micro AUC for the (potentially multi-target) observations vs the predicted probabilities

    :param array y_true: The observed targets
    :param array y_pred: The prediction probabilities
    :return: A dictionary of the form {metric_name:metric_value}
    """

    return roc_auc_score(y_true=y_true.astype("float"), y_score=y_pred.astype("float"), average="micro")
