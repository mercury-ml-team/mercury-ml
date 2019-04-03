import pytest
from mercury_ml.common.data_bunch import DataBunch

data_sets_dict = {
    "train": "hello",
    "valid": "bye"
}

@pytest.mark.parametrize("data_sets_dict",
                         [(data_sets_dict)])
def test_init(data_sets_dict):
    data_bunch = DataBunch(data_sets_dict)
    for data_set_name, data_set in data_sets_dict.items():
        assert getattr(data_bunch, data_set_name)==data_set


@pytest.mark.parametrize("data_sets_dict",
                         [(data_sets_dict)])
def test_add_data_sets(data_sets_dict):
    data_bunch = DataBunch()
    data_bunch.add_data_sets(data_sets_dict)
    for data_set_name, data_set in data_sets_dict.items():
        assert getattr(data_bunch, data_set_name)==data_set


@pytest.mark.parametrize("data_sets_dict",
                         [(data_sets_dict)])
def test_add_data_set(data_sets_dict):
    data_bunch = DataBunch()
    for data_set_name, data_set in data_sets_dict.items():
        data_bunch.add_data_set(data_set_name, data_set)


    for data_set_name, data_set in data_sets_dict.items():
        assert getattr(data_bunch, data_set_name)==data_set


class MockDataSet():
    def __init__(self, value):
        self.value = value

    def transform(self, transform_then_slice, transformation_params):
        new_data_set = MockDataSet(self.value * 100)
        new_data_set.transform_then_slice = transform_then_slice
        new_data_set.transformation_params = transformation_params

        return new_data_set

data_sets_dict = {
    "train": MockDataSet(123),
    "valid": MockDataSet(456),
    "test": MockDataSet(789)
}

data_bunch = DataBunch(data_sets_dict)


@pytest.mark.parametrize("data_bunch, data_set_names, params, transform_then_slice",
                         [(data_bunch, ["train", "valid"], {"blah": "456"}, True)])
def test_transform(data_bunch, data_set_names, params, transform_then_slice):
    new_data_bunch = data_bunch.transform(data_set_names, params, transform_then_slice)

    for data_set_name in data_set_names:
        new_data_set = getattr(new_data_bunch, data_set_name)
        assert new_data_set.value == getattr(data_bunch, data_set_name).value * 100
        assert new_data_set.transform_then_slice == transform_then_slice
        assert new_data_set.transformation_params == params

