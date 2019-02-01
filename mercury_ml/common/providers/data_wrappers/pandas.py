class PandasDataWrapper():
    """
    A DataWrapper with a Pandas DataFrame as its underlying data
    """

    def __init__(self, underlying, field_names):
        self.underlying = underlying
        self.field_names = field_names

    def slice_on_column_names(self, column_names):
        """
        Returns a Pandas DataFrame consisting of the columns that are present in the column_names list

        :param list list column_names: the names of the columns to return
        :return: Pandas DataFrame consisting of the selected columns
        """

        return self.underlying[column_names]

    def to_h2o(self, factor_columns_list=None):
        """
        Creates a new H2ODataWrapper based on this PandasDataWrapper and returns it as a new object

        :params list factor_columns_list: The names of the columns that should be treated as factors
        :return: a new H2ODataWrapper
        """

        from mercury_ml.common.providers.data_wrappers.h2o import H2ODataWrapper
        import h2o
        df_h2o = h2o.H2OFrame(self.underlying)

        # convert columns
        if not factor_columns_list:
            factor_columns_list = []

        for factor_column_name in factor_columns_list:
            df_h2o[factor_column_name] = df_h2o[factor_column_name].asfactor()

        return H2ODataWrapper(df_h2o, self.field_names)

    def to_numpy(self):
        """
        Creates a new NumpyDataWrapper based on this PandasDataWrapper and returns it as a new object

        :return: a new NumpyDataWrapper
        """

        from mercury_ml.common.providers.data_wrappers.numpy import NumpyDataWrapper
        return NumpyDataWrapper(self.underlying.values, self.field_names)

    def to_spark(self, spark_session_params=None):
        """
        Creates a new SparkDataWrapper based on this PandasDataWrapper and returns it as a new object

        :return: a new SparkDataWrapper
        :param: dict spark_session_params: A dictionary of parameters according to which to get or initialize a Spark session
        """

        from mercury_ml.common.providers.data_wrappers.spark import SparkDataWrapper
        from mercury_ml.spark.providers.session import get_or_create_spark_session
        self.underlying.columns = self.underlying.columns.astype(str)

        if not spark_session_params:
            spark_session_params = {}
        spark = get_or_create_spark_session(**spark_session_params)

        return SparkDataWrapper(
            underlying=spark.createDataFrame(self.underlying),
            field_names=self.field_names
        )

    def to_h2o_sparkling(self, spark_session_params=None, factor_columns_list=None, h2o_sparkling_params=None):
        """
        Creates a new H2ODataWrapper based on this PandasDataWrapper and returns it as a new object

        :param list factor_columns_list: The names of the columns that should be treated as factors
        :param list spark_session_params: A dictionary of parameters according to which to get or initialize a Spark session
        :param list h2o_sparkling_params: A dictionary of parameters according to which to get or initialize an H2OSparkling session

        :return: a new H2ODataWrapper
        """

        if spark_session_params is None:
            spark_session_params = {}

        if h2o_sparkling_params is None:
            h2o_sparkling_params = {}

        return self.to_spark(spark_session_params).to_h2o_sparkling(factor_columns_list=factor_columns_list,
                                                                    h2o_sparkling_params=h2o_sparkling_params)

    def to_pandas(self):
        return self

    def concatenate(self, right_data_wrapper):
        """
        Concatenates this PandasDataWrapper with the one given in right_data_wrapper and returns a new PandasDataWrapper

        :param PandasDataWrapper right_data_wrapper: to be concatenated to the right of "self"
        :return: a new PandasDataWrapper
        """

        import pandas as pd
        new_underlying = pd.concat([self.underlying, right_data_wrapper.underlying], axis=1)
        new_field_names = self.field_names + right_data_wrapper.field_names
        return PandasDataWrapper(
            underlying=new_underlying,
            field_names=new_field_names
        )

