def predict(data_set, model, return_columns = None, **kwargs):
    """
    Predicts with a trained model and returns the results as an H2ODataWrapper or H2OSparklingDataWrapper

    :param DataSet data_set: A DataSet with H2ODataWrappers or H2OSparklingDataWrappers as "full_data" attribute
    :param model: An H2O Model
    :param list return_columns: A list of columns to return from the resulting prediction
    :param kwargs: Additional keyword arguments to be passed to model.predict
    :return: And H2ODataWrapper or H2OSparklingDataWrapper with predictions
    """

    h2o_df = data_set.full_data.underlying
    prediction_h2o_df = model.predict(h2o_df, **kwargs)

    if return_columns:
        prediction_h2o_df = prediction_h2o_df[return_columns]

    from mercury_ml.common.providers.data_wrappers.h2o import H2ODataWrapper
    #TODO this should return an H2OSparklingDataWrapper in the input type is H2OSparklingDataWrapper
    return H2ODataWrapper(underlying=prediction_h2o_df, field_names=list(prediction_h2o_df.names))