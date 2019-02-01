def predict(data_set, model, **kwargs):
    """
    Produces predictions with a trained Keras model where inputs are arrays

    :param DataSet data_set: A DataSet with a NumpyDataWrapper called "features"
    :param model: A (fitted) Keras model
    :param kwargs: Additional parameters to be passed to model.predict
    :return: A NumpyDataWrapper with an array of predictions as its underlying
    """

    prediction_array = model.predict(x=data_set.features.underlying, **kwargs)
    from mercury_ml.common.providers.data_wrappers.numpy import NumpyDataWrapper
    return NumpyDataWrapper(underlying=prediction_array, field_names=data_set.targets.field_names)


def predict_generator(data_set, model, **kwargs):
    """
    Produces predictions with a trained Keras model where inputs are generators

    :param DataSet data_set: A DataSet with a KerasIteratorFeaturesDataWrapper called "features"
    :param model: A (fitted) Keras model
    :param kwargs: Additional parameters to be passed to model.predict
    :return: A NumpyDataWrapper with an array of predictions as its underlying
    """

    prediction_array = model.predict_generator(generator=data_set.features.underlying, **kwargs)
    from mercury_ml.common.providers.data_wrappers.numpy import NumpyDataWrapper
    return NumpyDataWrapper(underlying=prediction_array, field_names=data_set.features.underlying.get_labels_dummies())