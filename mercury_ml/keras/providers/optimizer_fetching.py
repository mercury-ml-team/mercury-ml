def get_keras_optimizer(optimizer_name, optimizer_params):
    """
    Fetches a vanilla Keras model optimizer

    :param string optimizer_name: The name of the optimizer
    :param dict optimizer_params: The parameters with which the optimizer should be initialized
    :return:
    """

    from keras import optimizers

    return getattr(optimizers, optimizer_name)(**optimizer_params)
