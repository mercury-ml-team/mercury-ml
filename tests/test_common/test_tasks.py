import pytest
from mercury_ml.common.tasks import read_train_valid_test_data_bunch, read_test_data_bunch, store_artifacts

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

#
# mock_state_dict = {}
# mock_remote_dir =  {}
#
# def mock_store_artifact_locally(data, directory, filename):
#     data["directory"]=directory
#     data["filename"]=filename
#
# def mock_copy_from_local_to_remote(source_dir, target_dir, filename, overwrite, delete_source):
#     target_dir["source_dir"]=source_dir
#     target_dir["filename"]=filename
#     target_dir["overwrite"]=overwrite
#     target_dir["delete_source"]=delete_source
#
# local_dir = "./local"
# filename = "abc.txt"
# overwrite_remote = True
# keep_local = False
#
# expected_local = {
#         "directory": local_dir,
#         "filename": filename
#     }
# expected_remote = {
#         "source_dir": local_dir,
#         "filename": filename,
#         "overwrite": overwrite_remote,
#         "delete_source": not keep_local
#     }
#
#
# @pytest.mark.parametrize("store_artifact_locally, copy_from_local_to_remote, data, local_dir, filename, remote_dir, overwrite_remote, keep_local, expected_local, expected_remote",
#                          [(mock_store_artifact_locally, mock_copy_from_local_to_remote, mock_state_dict, local_dir, filename, mock_remote_dir, overwrite_remote, keep_local, expected_local, expected_remote)])
# def test_store_artifacts(store_artifact_locally, copy_from_local_to_remote, data, local_dir, filename, remote_dir, overwrite_remote, keep_local, expected_local, expected_remote):
#
#     store_artifacts(store_artifact_locally, copy_from_local_to_remote, data, local_dir,
#                     filename, remote_dir, overwrite_remote, keep_local)
#
#     assert expected_local == data
#     assert expected_remote == remote_dir


# TODO