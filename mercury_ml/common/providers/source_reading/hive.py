#TODO possibly split into two modules: data_set readers and data_bunch readers

from mercury_ml.common.providers.data_bunch import DataBunch
from mercury_ml.common.providers.data_set import DataSet
from mercury_ml.common.providers.data_wrappers.spark import SparkDataWrapper

def read_spark_data_set(source_table, where_clause, full_data_columns, index_columns, features_columns, targets_columns):
    from mercury_ml.h2o.providers.session import get_or_create_spark
    spark = get_or_create_spark() #TODO this should be only "get" at this stage
    spark_df = spark.sql(
        "SELECT * FROM {} WHERE {}".format(source_table, where_clause)
    )

    return DataSet(
                {
                    "full_data": SparkDataWrapper(spark_df.select(*full_data_columns), full_data_columns),
                    "index": SparkDataWrapper(spark_df.select(*index_columns), index_columns),
                    "features": SparkDataWrapper(spark_df.select(*features_columns), features_columns),
                    "targets": SparkDataWrapper(spark_df.select(*targets_columns), targets_columns)
                }
    )
