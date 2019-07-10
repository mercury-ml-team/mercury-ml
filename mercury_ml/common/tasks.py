"""
Functions that are typically injected with one or more providers and performs a cohesive set of work that might involve
multiple steps
"""

import os
from mercury_ml.common.data_bunch import DataBunch
import mercury_ml

# source data reading
def read_data_bunch(**kwargs): #TODO will this be sufficient?

    data_sets_dict = {}
    for data_set_name, read_data_set_params in kwargs.items():
        data_sets_dict[data_set_name] = read_data_set(**read_data_set_params)

    return DataBunch(data_sets_dict)


def read_data_set(read_source_name, read_source_params):
    read_source = getattr(mercury_ml.common.SourceReaders, read_source_name)

    return read_source(**read_source_params)


# metric evaluation
def evaluate_metrics(data_bunch, evaluate_metrics_name, evaluate_metrics_params, data_set_names=None):

    metrics = {}
    for data_set_name in data_set_names:
        data_set = getattr(data_bunch, data_set_name)
        evaluate = getattr(mercury_ml.common.MetricEvaluators, evaluate_metrics_name)
        metrics[data_set_name] = evaluate(data_set, **evaluate_metrics_params)

    return metrics


# artifact storage
def store_artifacts(store_artifact_locally_name,  store_artifact_locally_params, copy_from_local_to_remote_name=None,
                    copy_from_local_to_remote_params=None):

    # save to local artifact store
    store_artifact_locally = getattr(mercury_ml.tensorflow.ModelSavers, store_artifact_locally_name)
    store_artifact_locally(**store_artifact_locally_params)

    # copy to remote artifact store
    if copy_from_local_to_remote_name and copy_from_local_to_remote_params:
        copy_from_local_to_remote = getattr(mercury_ml.common.ArtifactCopiers, copy_from_local_to_remote_name)
        copy_from_local_to_remote(**copy_from_local_to_remote_params)



# model storage
def store_model(
        save_model_name, save_model_params, copy_from_local_to_remote_name=None, copy_from_local_to_remote_params=None):

    save_model=getattr(mercury_ml.tensorflow.ModelSavers, save_model_name)
    path = save_model(**save_model_params)

    if copy_from_local_to_remote_name and copy_from_local_to_remote_params:
        copy_from_local_to_remote = getattr(mercury_ml.common.ArtifactCopiers, copy_from_local_to_remote_name)
        copy_from_local_to_remote(**copy_from_local_to_remote_params)

    return path

def load_model(load_model_name, load_model_params, copy_from_remote_to_local_name=None,
               copy_from_remote_to_local_params=None):

    if copy_from_remote_to_local_name and copy_from_remote_to_local_params:
        copy_from_remote_to_local=getattr(mercury_ml.common.ArtifactCopiers, copy_from_remote_to_local_name)
        copy_from_remote_to_local(**copy_from_remote_to_local_params)

    load_model_function = getattr(mercury_ml.tensorflow.ModelLoaders, load_model_name)
    model = load_model_function(**load_model_params)

    return model
