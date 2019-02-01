class NumpyDataWrapper():
    """
    A DataWrapper with a Numpy array as its underlying data
    """

    def __init__(self, underlying, field_names):
        self.underlying = underlying
        self.field_names = field_names

    def slice_on_column_names(self, column_names):
        """
        Returns a Numpy array consisting of the columns that are present in the column_names list

        :param list list column_names: the names of the columns to return
        :return: Numpy array consisting of the selected columns
        """

        column_number_list =self._get_field_name_indices(column_names)
        return self.underlying[:, column_number_list]

    def to_h2o(self, factor_columns_list=None):
        """
        Creates a new H2ODataWrapper based on this NumpyDataWrapper and returns it as a new object

        :return: a new H2ODataWrapper
        """
        return self.to_pandas().to_h2o(factor_columns_list)

    def to_pandas(self):
        """
        Creates a new PandasDataWrapper based on this NumpyDataWrapper and returns it as a new object

        :return: a new PandasDataWrapper
        """
        import pandas as pd
        from mercury_ml.common.providers.data_wrappers.pandas import PandasDataWrapper
        df = pd.DataFrame(data=self.underlying, columns=self.field_names)

        return PandasDataWrapper(df, self.field_names)

    def to_spark(self):
        """
        Creates a new SparkDataWrapper based on this NumpyDataWrapper and returns it as a new object

        :return: a new SparkDataWrapper
        """
        return self.to_pandas().to_spark()

    def to_h2o_sparkling(self, spark_session_params=None, h2o_sparkling_params=None):
        """
        Creates a new H2OSparklingDataWrapper based on this NumpyDataWrapper and returns it as a new object

        :return: a new H2OSparklingDataWrapper
        """
        return self.to_pandas().to_h2o_sparkling(spark_session_params, h2o_sparkling_params)

    def to_numpy(self):
        return self

    def _get_field_name_indices(self, field_names_subset):
        """
        Takes a list of field names, compares if with self.field_names and return a list of the indices where field_names_subset
        intersects field self.field_names
        :param list field_names_subset: A list of strings with field names
        :return: A list of integers (indices)
        """
        ls = []
        for field_name in field_names_subset:
            ls = ls + [self.field_names.index(field_name)]
        return ls
