

class SparkDataWrapper():

    def __init__(self, underlying, field_names):
        self.underlying = underlying
        self.field_names = field_names

    def slice_on_column_names(self, column_names):
        """
        Returns a Spark DataFrame consisting of the columns that are present in the column_names list

        :param list list column_names: the names of the columns to return
        :return: Spark DataFrame consisting of the selected columns
        """
        return self.underlying[column_names]

    def to_h2o_sparkling(self, factor_columns_list=None, h2o_sparkling_params=None):
        """
        Creates a new H2OSparklingDataWrapper based on this SparkDataWrapper and returns it as a new object

        :param list factor_columns_list: The names of the columns that should be treated as factors
        :param list h2o_sparkling_params: A dictionary of parameters according to which to get or initialize an H2OSparkling session

        :return: a new H2OSparklingDataWrapper
        """

        from mercury_ml.common.providers.data_wrappers.h2o import H2OSparklingDataWrapper
        from mercury_ml.h2o.providers.session import get_or_create_h2o_sparkling

        # get H2OSparkling context (#TODO: if context doesn't exist yet, raise error! Should not be created here)
        if not h2o_sparkling_params:
            h2o_sparkling_params = {}
        h2o_context = get_or_create_h2o_sparkling(**h2o_sparkling_params)

        # convert to H2OFrame
        df_h2o = h2o_context.as_h2o_frame(self.underlying)

        # convert factor columns
        if not factor_columns_list:
            factor_columns_list = []

        for factor_column_name in factor_columns_list:
            df_h2o[factor_column_name] = df_h2o[factor_column_name].asfactor()

        return H2OSparklingDataWrapper(df_h2o, self.field_names)

    def to_numpy(self):
        """
        Creates a new NumpyDataWrapper based on this SparkDataWrapper and returns it as a new object

        :return: a new NumpyDataWrapper
        """

        return self.to_pandas().to_numpy()


    def to_pandas(self):
        """
        Creates a new PandasDataWrapper based on this SparkDataWrapper and returns it as a new object

        :return: a new PandasDataWrapper
        """

        from mercury_ml.common.providers.data_wrappers.pandas import PandasDataWrapper
        df = self.underlying.toPandas()
        return PandasDataWrapper(df, self.field_names)

    def to_spark(self):
        return self

    def concatenate(self, right_data_wrapper):
        """
        Concatenates this SparkDataWrapper with the one given in right_data_wrapper and returns a new SparkDataWrapper

        :param SparkDataWrapper right_data_wrapper: to be concatenated to the right of "self"
        :return: a new SparkDataWrapper
        """

        raise NotImplementedError("Concatenating Spark DFs not implemented yet")

        # spark = _get_spark()
        #
        # new_underlying = ... # TODO how to concatenate while maintaining row order?
        # new_field_names = self.field_names + right_data_wrapper.field_names
        # return SparkDataWrapper(
        #     underlying = new_underlying,
        #     field_names = new_field_names
        # )