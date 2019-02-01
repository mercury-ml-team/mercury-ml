import pytest
from mercury_ml.common.utils import load_referenced_json_config, recursively_update_config

referenced_config_filepath = "./inputs/referenced_config.json"
dereferenced_config = {"hello": 12345, "bye": 12345}

@pytest.mark.parametrize("filepath, expected",
                         [(referenced_config_filepath, dereferenced_config)])
def test_load_referenced_json_config(filepath, expected):
    config = load_referenced_json_config(filepath)

    assert config==expected


config_to_format = {"hello": "{update_this}_123"}
string_formatting_dict = {"update_this": "bye"}
formatted_config = {"hello": "bye_123"}

@pytest.mark.parametrize("config, string_formatting_dict, expected",
                         [(config_to_format, string_formatting_dict, formatted_config)])
def test_load_referenced_json_config(config, string_formatting_dict, expected):
    recursively_update_config(config, string_formatting_dict)

    assert config==expected