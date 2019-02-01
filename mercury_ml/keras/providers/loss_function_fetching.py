def get_keras_loss(loss_name):
    """
    Fetches built in keras losses

    :param loss_name: The name of the loss to fetch
    :return: A vanilla Keras loss function
    """

    from keras import losses

    return getattr(losses, loss_name)


def get_custom_loss(loss_function_getter_name, loss_function_getter_params):
    """
    Fetches custom Keras losses

    :param string loss_function_getter_name: The name of the function to used in order to fetch the loss function
    :param dict loss_function_getter_params: The paramters to pass to the getter

    :return: A vanilla Keras loss function
    """

    from mercury_ml.keras.providers import custom_loss_functions

    return getattr(custom_loss_functions, loss_function_getter_name)(**loss_function_getter_params)