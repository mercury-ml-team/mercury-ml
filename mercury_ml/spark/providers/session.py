def get_or_create_spark_session(spark_config=None, log_level="ERROR", enableHiveSupport=True):
    """
    Gets or initiates an Spark session.

    :param dict spark_config: the parameters based on which the Spark session is to be initiated
    :param log_level: The log level of the Spark session
    :param bool enableHiveSupport: If set to true Hive support will be enabled
    :return: A spark session object
    """

    from pyspark.sql import SparkSession

    builder = SparkSession.builder
    if spark_config:
        # http://spark.apache.org/docs/latest/configuration.html
        for key, value in spark_config.items():
            builder = SparkSession.builder.config(key, value)

    if enableHiveSupport:
        builder = builder.enableHiveSupport()

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(log_level)

    #print(spark.sparkContext.getConf().getAll())
    return spark

