class H2ODataWrapper():
    """
    A DataWrapper with an H2OFrame as its underlying data
    """

    def __init__(self, underlying, field_names):
        self.underlying = underlying
        self.field_names = field_names

    def slice_on_column_names(self, column_names):
        """
        Returns an H2OFrame consisting of the columns that are present in the column_names list

        :param list column_names: the names of the columns to return
        :return: H2OFrame consisting of the selected columns
        """

        return self.underlying[column_names]

    def to_pandas(self):
        """
        Creates a new PandasDataWrapper based on this H2ODataWrapper and returns it as a new object

        :return: a new PandasDataWrapper
        """

        from mercury_ml.common.providers.data_wrappers.pandas import PandasDataWrapper
        return PandasDataWrapper(self.underlying.as_data_frame(use_pandas=True), self.field_names)

    def to_numpy(self):
        """
        Creates a new NumpyDataWrapper based on this H2ODataWrapper and returns it as a new object

        :return: a new NumpyDataWrapper
        """

        from mercury_ml.common.providers.data_wrappers.numpy import NumpyDataWrapper
        return NumpyDataWrapper(self.underlying.as_data_frame(use_pandas=True).values, self.field_names)

    def to_h2o(self):
        return self

    def concatenate(self, right_data_wrapper):
        """
        Concatenates this H2ODataWrapper with the one given in right_data_wrapper and returns a new H2ODataWrapper

        :param H2ODataWrapper right_data_wrapper: to be concatenated to the right of "self"
        :return: a new H2ODataWrapper
        """

        new_underlying =self.underlying.cbind(right_data_wrapper.underlying)
        new_field_names = self.field_names + right_data_wrapper.field_names
        return H2ODataWrapper(
            underlying = new_underlying,
            field_names = new_field_names
        )


class H2OSparklingDataWrapper():
    """
    A DataWrapper with an H2OFrame within an H2OSparkling session) as its underlying data
    """

    def __init__(self, underlying, field_names):
        self.underlying = underlying
        self.field_names = field_names

    def slice_on_column_names(self, column_names):
        """
        Returns an H2OFrame consisting of the columns that are present in the column_names list

        :param list column_names: the names of the columns to return
        :return: H2OFrame consisting of the selected columns
        """

        return self.underlying[column_names]

    def to_pandas(self, h2o_sparkling_params=None):
        """
        Creates a new PandasDataWrapper based on this H2OSparklingDataWrapper and returns it as a new object

        :param: dict h2o_sparkling_params: Parameters passed to "get_or_create_h2o_sparkling" which will get or initiate Spark and H2OSparkling sessions
        :return: a new PandasDataWrapper
        """

        return self.to_spark(h2o_sparkling_params).to_pandas()

    def to_numpy(self, h2o_sparkling_params=None):
        """
        Creates a new NumpyDataWrapper based on this H2OSparklingDataWrapper and returns it as a new object

        :param: dict h2o_sparkling_params: Parameters passed to "get_or_create_h2o_sparkling" which will get or initiate Spark and H2OSparkling sessions
        :return: a new NumpyDataWrapper
        """

        return self.to_spark(h2o_sparkling_params).to_pandas().to_numpy()

    def to_spark(self, h2o_sparkling_params=None):
        """
        Creates a new SparkDataWrapper based on this H2OSparklingDataWrapper and returns it as a new object

        :param: dict h2o_sparkling_params: A dictionary of parameters according to which to get or initialize an H2OSparkling session
        :return: a new SparkDataWrapper
        """

        from mercury_ml.h2o.providers.session import get_or_create_h2o_sparkling
        from mercury_ml.common.providers.data_wrappers.spark import SparkDataWrapper

        # get H2OSparkling context #TODO: if context doesn't exist yet, raise error! Should not be created here
        if not h2o_sparkling_params:
            h2o_sparkling_params = {}
        h2o_context = get_or_create_h2o_sparkling(**h2o_sparkling_params)

        return SparkDataWrapper(
            underlying=h2o_context.as_spark_frame(self.underlying),
            field_names=self.field_names
        )

    def concatenate(self, right_data_wrapper):
        """
        Concatenates this H2OSparklingDataWrapper with the one given in right_data_wrapper and returns a new H2ODataWrapper

        :param H2OSparklingDataWrapper right_data_wrapper: to be concatenated to the right of "self"
        :return: a new H2OSparklingDataWrapper
        """

        new_underlying=self.underlying.cbind(right_data_wrapper.underlying)
        new_field_names=self.field_names + right_data_wrapper.field_names
        return H2OSparklingDataWrapper(
            underlying=new_underlying,
            field_names=new_field_names
        )
