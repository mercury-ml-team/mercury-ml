"""
Functions that are typically injected with one or more providers and performs a cohesive set of work that might involve
multiple steps
"""

import os
from mercury_ml.common.providers.data_bunch import DataBunch

# source data reading
def read_train_valid_test_data_bunch(read_data_set, train_params, valid_params, test_params):
    """
    Takes the provider "read_data_set" and uses it to read "train, "valid" and "test" DataSets and add them to a
    new DataBunch

    :param callable read_data_set: A function that reads source data and returns a DatSet
    :param dict train_params: The parameters according to which to read the "train" DataSet
    :param dict valid_params: The parameters according to which to read the "valid" DataSet
    :param dict test_params: The parameters according to which to read the "test" DataSet
    :return: a DataBunch
    """
    return DataBunch(
        {
            "train": read_data_set(**train_params),
            "valid": read_data_set(**valid_params),
            "test": read_data_set(**test_params)
        }
    )

def read_test_data_bunch(read_data_set, test_params):
    """
    Takes the provider "read_data_set" and uses it to read a "test" DataSet and add it to a new DataBunch

    :param callable read_data_set: A function that reads source data and returns a DatSet
    :param dict test_params: The parameters according to which to read the "test" DataSet
    :return: a DataBunch
    """
    return DataBunch(
        {
            "test": read_data_set(**test_params)
        }
    )


# artifact storage
def store_artifacts(store_artifact_locally, copy_from_local_to_remote, data, local_dir, filename, remote_dir=None,
                    overwrite_remote=True, keep_local=True, copy_from_local_to_remote_kwargs=None):

    """
    Uses "store_artifact_locally" to store artifacts to local disk. Then uses "copy_from_local_to_remote" to also store
    the artifacts in a remote location.

    :param callable store_artifact_locally: A function that stores artifacts to local disk.
    :param callable copy_from_local_to_remote: A function that copies artifacts from local disk to a remote location.
    :param data: The artifact to be stored
    :param string local_dir: The local directory where the artifact should be stored
    :param string filename: The filename under which the artifact should be stored
    :param string remote_dir: The remote "directory" where the artifact should be copied. If None then copying to remote is skipped
    :param bool overwrite_remote: If true then if the artifact already exists in the remote location it will be overwritten
    :param bool keep_local: If false then the local artifact will be deleted after copying to remote
    :return:
    """

    # save to local artifact store
    filepath = store_artifact_locally(data=data,
                                      directory=local_dir,
                                      filename=filename)

    # copy to remote artifact store
    if not copy_from_local_to_remote_kwargs:
        copy_from_local_to_remote_kwargs={}

    if remote_dir:
        copy_from_local_to_remote(source_dir=local_dir,
                                  target_dir=remote_dir,
                                  filename=os.path.basename(filepath),
                                  overwrite=overwrite_remote,
                                  delete_source=not keep_local,
                                  **copy_from_local_to_remote_kwargs)

# metric evaluation
def evaluate_metrics(data_set, custom_metrics_dict):
    """
    Loops through the entries in custom_metrics_dict and calculates each metric on the given dataset

    :param DataSet data_set: A DataSet that has DataWrappers called "targets" and "predictions" as attributes.
    :param dict custom_metrics_dict: A dictionary of custom metrics of the form { {custom_metric_name: custom_metric_function}...}
    where custom_metric_function is a callable that takes the inputs "y_true" and "y_pred" and returns the scalar value of
    the calculated metric.

    :return: A dictionary of the form {{custom_metric_name: metric_value}, ...}
    """
    y_true = data_set.targets.underlying
    y_pred = data_set.predictions.underlying

    custom_metrics = {}

    if not custom_metrics_dict:
        return custom_metrics
    else:
        for custom_metric_name, evaluate_custom_metric in custom_metrics_dict.items():
            custom_metric_value = evaluate_custom_metric(y_true, y_pred)
            custom_metrics[custom_metric_name] = custom_metric_value

        return custom_metrics


def evaluate_label_metrics(data_set, label_specific_custom_metrics_dict):
    """
    Loops through the entries in label_specific_custom_metrics_dict and calculates each metric for each label on the given dataset

    :param DataSet data_set: A DataSet that has DataWrappers called "targets" and "predictions" as attributes.
    :param dict label_specific_custom_metrics_dict: A dictionary of custom label metrics of the form {{custom_label_metric_name: custom_label_metric_function}...}
    where custom_label_metric_function is a callable that takes the inputs "y_true", "y_pred" and "label" and returns a dictionary
    of the form {{label_metric_name: {label_name, label_metric_value}},...}

    :return: A dictionary of the form {{label_metric_name: {label_name, label_metric_value}},...}
    """


    y_true = data_set.targets.underlying
    y_pred = data_set.predictions.underlying
    labels = data_set.targets.field_names

    all_metrics_dict = {}

    if not label_specific_custom_metrics_dict:
        return all_metrics_dict

    for custom_label_metric_name, evaluate_custom_label_metric in label_specific_custom_metrics_dict.items():
        custom_label_metric_dict = evaluate_custom_label_metric(y_true, y_pred, labels)
        all_metrics_dict.update(custom_label_metric_dict)

    return all_metrics_dict



# model storages
def store_model(save_model, model, filename, local_dir, extension=None, copy_from_local_to_remote=None,
                remote_dir=None, overwrite_remote=None, keep_local=True, **kwargs):
    """
    Uses "save_model" and "model" to save a model to local disk. Then uses "copy_from_local_to_remote" to also store
    the model in a remote location.

    :param callable save_model: A function that stores model to local disk.
    :param model: A model object.
    :param string filename: The filename under which the model should be stored
    :param string local_dir: The local directory where the model should be stored
    :param string extension: The extension to the filename (e.g. ".h5", ".pb") with which the model should be saved
    :param callable copy_from_local_to_remote: A function that copies artifacts from local disk to a remote location.
    :param string remote_dir: The remote "directory" where the model should be copied. If None then copying to remote is skipped
    :param bool overwrite_remote: If true then if the model already exists in the remote location it will be overwritten
    :param bool keep_local: If false then the local model will be deleted after copying to remote
    :param kwargs: Additional arguments passed to "save_model"
    :return:
    """

    path = save_model(model=model,
                      filename=filename,
                      local_dir=local_dir,
                      extension=extension,
                      **kwargs)

    if copy_from_local_to_remote and remote_dir:
        copy_from_local_to_remote(source_dir=local_dir,
                                  target_dir=remote_dir,
                                  filename=os.path.basename(path),
                                  overwrite=overwrite_remote,
                                  delete_source=not keep_local)

    return path

def load_model(load_model, filename, local_dir, extension=None, remote_dir=None, copy_from_remote_to_local=None,
               always_fetch_remote=False, **kwargs):
    """
    First checks if the desired model is avaible at the local location. If not it is copied from remote to local. Thereafter
    the model is loaded and returned.

    :param callable load_model: Function that will return a loaded model
    :param string filename: The filename under which the model is stored
    :param string local_dir: The local directory where the model is stored
    :param string extension: The extension to the filename (e.g. ".h5", ".pb") with which the model is saved
    :param string remote_dir: The remote "directory" where the model is saved.
    :param callable copy_from_remote_to_local: Function that copies data from a remote location to a local one
    :param bool always_fetch_remote: If true the the model will always be fetched from the remote location, even if
    it appears to be present locally
    :param kwargs: Additional arguments that are passed to "load_model"
    :return: A model object
    """

    path = os.path.join(local_dir, filename + extension)

    if not os.path.isfile(path) or always_fetch_remote:
        copy_from_remote_to_local(source_dir=remote_dir,
                                  target_dir=local_dir,
                                  filename=filename + extension,
                                  overwrite=True,
                                  delete_source=False)

    model = load_model(filename=filename,
                       local_dir=local_dir,
                       extension=extension,
                       **kwargs)

    return model
