import pytest
from mercury_ml.common.tasks import read_train_valid_test_data_bunch, read_test_data_bunch, store_artifacts, \
    evaluate_metrics, evaluate_label_metrics, store_model, load_model
from mercury_ml.common.providers.data_set import DataSet

def mock_read_data_set(return_value):
    return return_value

train_params = {"return_value":123}
valid_params = {"return_value":456}
test_params = {"return_value":789}
expected = {"train": 123, "valid":456, "test":789}

@pytest.mark.parametrize("read_data_set, train_params, valid_params, test_params, expected",
                         [(mock_read_data_set, train_params, valid_params, test_params, expected)])
def test_read_train_valid_test_data_bunch(read_data_set, train_params, valid_params, test_params, expected):
    data_bunch = read_train_valid_test_data_bunch(read_data_set, train_params, valid_params, test_params)

    assert data_bunch.train == expected["train"]
    assert data_bunch.valid == expected["valid"]
    assert data_bunch.test == expected["test"]


@pytest.mark.parametrize("read_data_set, test_params, expected",
                         [(mock_read_data_set, test_params, expected)])
def test_read_test_data_bunch(read_data_set, test_params, expected):
    data_bunch = read_test_data_bunch(read_data_set, test_params)

    assert data_bunch.test == expected["test"]


mock_local_state = {}
mock_remote_state = {}

def mock_store_artifact_locally(data, directory, filename):
    global mock_local_state
    mock_local_state["data"]=data
    mock_local_state["filepath"] = directory + "/" + filename

    return mock_local_state["filepath"]


def mock_copy_from_local_to_remote(source_dir, target_dir, filename, overwrite, delete_source):
    global mock_local_state
    global mock_remote_state

    if mock_local_state["filepath"] == source_dir + "/" + filename:
        mock_remote_state["data"]=mock_local_state["data"]

data = 123
local_dir = "./local"
remote_dir = "./remote"
filename = "abc.txt"

overwrite_remote = True
keep_local = False


@pytest.mark.parametrize("store_artifact_locally, copy_from_local_to_remote, data, local_dir, filename, remote_dir, overwrite_remote, keep_local",
                         [(mock_store_artifact_locally, mock_copy_from_local_to_remote, data, local_dir, filename, remote_dir, overwrite_remote, keep_local)])
def test_store_artifacts(store_artifact_locally, copy_from_local_to_remote, data, local_dir, filename, remote_dir, overwrite_remote, keep_local):
    global mock_local_state
    global mock_remote_state

    store_artifacts(store_artifact_locally, copy_from_local_to_remote, data, local_dir,
                    filename, remote_dir, overwrite_remote, keep_local)

    assert mock_local_state["data"]==mock_remote_state["data"]



class MockDataWrapper():
    def __init__(self, underlying, field_names):
        self.underlying = underlying
        self.field_names = field_names

data_set = DataSet({
    "predictions": MockDataWrapper(5, ["a"]),
    "targets": MockDataWrapper(10, ["a"])
})

custom_metrics_dict = {
    "metric_plus": lambda y_true, y_pred: y_true+y_pred,
    "metric_minus": lambda y_true, y_pred: y_true-y_pred
}
expected_dict = {
    "metric_plus": 15,
    "metric_minus": 5
}

@pytest.mark.parametrize("data_set, custom_metrics_dict, expected_dict",
                         [(data_set, custom_metrics_dict, expected_dict)])
def test_evaluate_metrics(data_set, custom_metrics_dict, expected_dict):
    metrics=evaluate_metrics(data_set, custom_metrics_dict)
    for metric_name, expected_value in expected_dict.items():
        assert metrics[metric_name]==expected_value

label_specific_custom_metrics_dict = {
    "metric_plus": lambda y_true, y_pred, labels: {"metric_plus": {labels[0]:y_true+y_pred}},
    "metric_minus": lambda y_true, y_pred, labels: {"metric_minus": {labels[0]:y_true-y_pred}}
}

expected_label_dict = {
    "metric_plus": {"a": 15},
    "metric_minus": {"a": 5}
}

@pytest.mark.parametrize("data_set, label_specific_custom_metrics_dict, expected_label_dict",
                         [(data_set, label_specific_custom_metrics_dict, expected_label_dict)])
def test_evaluate_label_metrics(data_set, label_specific_custom_metrics_dict, expected_label_dict):
    metrics=evaluate_label_metrics(data_set, label_specific_custom_metrics_dict)
    for metric_name, expected_value in expected_label_dict.items():
        assert metrics[metric_name]==expected_value


mock_local_model_state = {}
mock_remote_model_state = {}

def mock_save_model(model, filename, local_dir, extension):
    global mock_local_model_state
    mock_local_model_state["model"]=model
    mock_local_model_state["filepath"] = local_dir + "/" + filename + extension
    return mock_local_model_state["filepath"]


def mock_copy_model_from_local_to_remote(source_dir, target_dir, filename, overwrite, delete_source):
    global mock_local_model_state
    global mock_remote_model_state

    if mock_local_model_state["filepath"] == source_dir + "/" + filename:
        mock_remote_model_state["model"]=mock_local_model_state["model"]

model = 123
local_dir = "./local"
remote_dir = "./remote"
filename = "abc"
extension= ".h5"

overwrite_remote = True
keep_local = False
@pytest.mark.parametrize("save_model, model, filename, local_dir, extension, copy_from_local_to_remote,  remote_dir, overwrite_remote, keep_local",
                         [(mock_save_model, model, filename, local_dir, extension, mock_copy_model_from_local_to_remote,  remote_dir,
                           overwrite_remote, keep_local)])
def test_store_model(save_model, model, filename, local_dir, extension, copy_from_local_to_remote,  remote_dir,
                     overwrite_remote, keep_local):
    global mock_local_model_state
    global mock_remote_model_state

    store_model(save_model, model, filename, local_dir, extension, copy_from_local_to_remote, remote_dir,
                overwrite_remote, keep_local)

    assert mock_local_model_state["model"] == mock_remote_model_state["model"]




mock_local_load_model_state = {}
mock_remote_load_model_state = {
    "model": 123
}

def mock_load_model(filename, local_dir, extension):
    global mock_local_load_model_state
    global mock_remote_load_model_state

    if mock_remote_load_model_state["filepath"] == local_dir + "/" + filename + extension:
        mock_local_load_model_state["model"]=mock_remote_load_model_state["model"]


def mock_copy_model_from_remote_to_local(source_dir, target_dir, filename, overwrite, delete_source):
    global mock_remote_load_model_state
    mock_remote_load_model_state["filepath"] = target_dir + "/" + filename

local_dir = "./local"
remote_dir = "./remote"
filename = "abc"
extension= ".h5"
always_fetch_remote = True

@pytest.mark.parametrize("do_load_model, filename, local_dir, extension, remote_dir, copy_from_remote_to_local, always_fetch_remote",
                         [(mock_load_model, filename, local_dir, extension, remote_dir, mock_copy_model_from_remote_to_local, always_fetch_remote)])
def test_load_model(do_load_model, filename, local_dir, extension, remote_dir, copy_from_remote_to_local, always_fetch_remote):
    global mock_local_load_model_state
    global mock_remote_load_model_state

    load_model(do_load_model, filename, local_dir, extension, remote_dir, copy_from_remote_to_local, always_fetch_remote)

    assert mock_local_load_model_state["model"] == mock_remote_load_model_state["model"]