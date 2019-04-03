from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from copy import copy
import numpy as np


def evaluate_auc(y_true, y_pred, labels):
    """
    Calculates the AUC for each each of the individual observed targets / predictions within y_true / y_pred and returns
    a formatted dictionary with the results.

    :param ndarray y_true: The (potentially multiple) observed targets
    :param ndarray y_pred: The (potentially multiple) prediction probabilities
    :param list labels: Labels for which metric should be calculated
    :return: A dictionary of the form {label_metric_name: {label_name, label_metric_value}}
    """

    i = 0
    auc_dict = {}
    for label in labels:
        try:
            auc_dict[label] = np.asscalar(roc_auc_score(y_true=y_true[:, i].astype("float"),
                                                        y_score=y_pred[:, i].astype("float")))
        except:
            auc_dict[label] = None

        i = i + 1

    dict_to_return = {}
    dict_to_return["AUC"] = auc_dict
    return dict_to_return


def evaluate_confusion_matrix(y_true, y_pred, labels):
    """
    Calculates a Confusion for each each of for y_true / y_pred and returns a formatted dictionary with the results.

    :param ndarray y_true: The (potentially multiple) observed targets
    :param ndarray y_pred: The (potentially multiple) prediction probabilities
    :param list labels: Labels for which metric should be calculated
    :return: A dictionary of the form {label_metric_name: {label_name, label_metric_value}}
    """

    y_true_argmax = y_true.argmax(axis=-1)
    y_pred_argmax = y_pred.argmax(axis=-1)

    matrix = confusion_matrix(y_pred=y_pred_argmax,
                              y_true=y_true_argmax)

    cm_count_dict = {}
    cm_rate_dict = {}
    dict_to_return = {}
    for col in range(len(labels)):
        for row in range(len(labels)):
            cm_count_dict[labels[row]] = np.asscalar(matrix[row, col])
            cm_rate_dict[labels[row]] = np.asscalar(matrix[row, col] / matrix[row, :].sum())
        dict_to_return["ConfMat_Count_" + labels[col]] = copy(cm_count_dict)
        dict_to_return["ConfMat_Rate_" + labels[col]] = copy(cm_rate_dict)

    return dict_to_return



def evaluate_accuracy(y_true, y_pred, labels):
    """
    Calculates the Accuracy for each each of the individual observed targets / predictions within y_true / y_pred and returns
    a formatted dictionary with the results.

    :param ndarray y_true: The (potentially multiple) observed targets
    :param ndarray y_pred: The (potentially multiple) prediction probabilities
    :param list labels: Labels for which metric should be calculated
    :return: A dictionary of the form {label_metric_name: {label_name, label_metric_value}}
    """

    i = 0
    acc_dict = {}

    y_pred = y_pred.astype("float")
    y_true = y_true.astype("float")

    for label in labels:
        acc_dict[label] = np.asscalar(accuracy_score(y_true=_apply_threshold(y_true[:, i]),
                                                     y_pred=_apply_threshold(y_pred[:, i])))
        i = i + 1

    dict_to_return = {}
    dict_to_return["Accuracy"] = acc_dict
    return dict_to_return



def evaluate_multi_label_confusion_matrix(y_true, y_pred, labels):
    """
    Calculates binary confusion matrices for each each of the individual observed targets / predictions within y_true / y_pred and returns
    a formatted dictionary with the results.

    :param ndarray y_true: The (potentially multiple) observed targets
    :param ndarray y_pred: The (potentially multiple) prediction probabilities
    :param list labels: Labels for which metric should be calculated
    :return: A dictionary of the form {label_metric_name: {label_name, label_metric_value}}
    """

    dict_to_return = {"ConfMat_Count__00": {},
                      "ConfMat_Count__01": {},
                      "ConfMat_Count__10": {},
                      "ConfMat_Count__11": {},
                      "ConfMat_Rate__00": {},
                      "ConfMat_Rate__01": {},
                      "ConfMat_Rate__10": {},
                      "ConfMat_Rate__11": {}
                      }

    y_pred = y_pred.astype("float")
    y_true = y_true.astype("float")

    for label_col in range(len(labels)):
        y_true_flat = _apply_threshold(y_true[:, label_col])
        y_pred_flat = _apply_threshold(y_pred[:, label_col])
        matrix = confusion_matrix(y_pred=y_pred_flat, y_true=y_true_flat)
        for col in [0, 1]:
            for row in [0, 1]:
                count_dict = {labels[label_col]: np.asscalar(matrix[row, col])}
                rate_dict = {labels[label_col]: np.asscalar(matrix[row, col] / matrix[row, :].sum())}
                dict_to_return["ConfMat_Count__" + str(row) + str(col)].update(count_dict)
                dict_to_return["ConfMat_Rate__" + str(row) + str(col)].update(rate_dict)

    return dict_to_return




def _apply_threshold(array, threshold=0.5):
    thresholded_array = np.copy(array)
    thresholded_array[thresholded_array >= threshold] = 1
    thresholded_array[thresholded_array < threshold] = 0
    return thresholded_array
