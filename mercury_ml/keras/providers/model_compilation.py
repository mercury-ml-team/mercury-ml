def compile_model(model, optimizer, loss=None, **kwargs):

    """
    Compiles a Keras model

    :param model: A keras model
    :param optimizer: The optimizer to use when compiling
    :param loss: The loss function to use when compiling
    :param kwargs: Additional arguments to pass to the compiler
    :return: a (compiled) Keras model
    """


    model.compile(optimizer=optimizer,
                  loss=loss,
                  **kwargs)

    return model
