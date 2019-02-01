from keras import backend as K
import numpy as np
import pandas as pd
import os


def get_mock_loss_function(mock_value):
    """
    A loss function that just returns a hardcoded value. Used for testing only.

    :param mock_value:
    :return:
    """

    def mock_loss(y_true, y_pred):
        return mock_value
    return mock_loss

def get_weighted_loss_function(label_df_path, label_columns, label_df_read_params=None):
    """
    Returns a weighted_binary_crossentropy loss function. Calculates the weights based on an input DataFrame that can
    be read from a local disk location

    :param string label_df_path: The path where a DataFrame with the observed labels can be found
    :param list label_columns: The columns within the DataFrame where the observed labels are kept
    :param dict label_df_read_params: The parameters to be used to read the label DataFrame
    :return: callable weighted_binary_crossentropy
    """

    y_true = _load_label_df(label_df_path, label_columns, label_df_read_params)
    weights = _get_class_weights(y_true)
    def weighted_binary_crossentropy(y_true, y_pred):
        """
        A loss function that returns a weighted binary crossentropy. This will result in equal penalties being placed on
        all classes. Particularly useful when modeling mutliple binary labels (i.e. "attributes") where so labels are
        observed far less freuently than others

        :param y_true:
        :param y_pred:
        :return:
        """

        return K.mean((weights[:,0]**(1-y_true))*(weights[:,1]**(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)
    return weighted_binary_crossentropy

def _get_class_weights(y_true):
    """
    Computes balanced class weighted based on the observed values
    """

    from sklearn.utils.class_weight import compute_class_weight
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight("balanced", [0.,1.], y_true[:, i])
    return weights

def _load_label_df(label_df_path, label_columns, label_df_read_params=None):
    """
    Loads a label_df from a local path
    """
    extension = os.path.splitext(label_df_path)[1]
    if not label_df_read_params:
        label_df_read_params = {}
    if extension == ".pkl":
        label_df = pd.read_pickle(label_df_path, **label_df_read_params)
    elif extension == ".csv":
        label_df = pd.read_csv(label_df_path, **label_df_read_params)
    elif extension == ".json":
        label_df = pd.read_json(label_df_path, **label_df_read_params)
    else:
        raise NotImplementedError("Extension '{}' has not yet been implemented")

    return label_df[label_columns].values