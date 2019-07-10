def get_mock_optimizer(mock_value):
    """
    A loss function that just returns a hardcoded value. Used for testing only.

    :param mock_value:
    :return:
    """

    def mock_optimizer(y_true, y_pred):
        return mock_value
    return mock_optimizer