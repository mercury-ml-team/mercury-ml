def fit(model, data_bunch, **kwargs):

    """
    Fits an H2O Model
    :param model: An H2O Model
    :param data_bunch: A DataBunch with "train" and "valid" datasets that consist of H2ODataWrapper or H2OSparklingDataWrapper
    :param kwargs: Keyword arguments to be passed to model.train
    :return: A (fitted) H2O Model
    """

    train_df = data_bunch.train.full_data.underlying
    valid_df = data_bunch.valid.full_data.underlying
    features_list = data_bunch.train.features.field_names
    targets_list = data_bunch.valid.targets.field_names[0]

    model.train(x=features_list,
                y=targets_list,
                training_frame=train_df,
                validation_frame=valid_df,
                **kwargs
                )

    return model


