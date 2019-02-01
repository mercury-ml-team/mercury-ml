def get_or_create_h2o():
    """
    Gets or initiates and H2O session
    :return:
    """

    import h2o
    h2o.init()
    return h2o

def get_or_create_h2o_sparkling(h2o_context_params=None,  h2o_log_level="ERROR", spark_session_params=None):
    """
    Gets or initiates an H2OSparkling session.

    :param dict h2o_context_params: The parameters based on which the H2OSparkling session is to be initialized
    :param string h2o_log_level: The log level of the H2OSparkling Session
    :param dict spark_session_params: The parameters based on which the Spark session is to be initialized
    :return:
    """

    from pysparkling import H2OConf, H2OContext

    # Start SparkSession
    #TODO possibly change this to create spark session outside and pass "spark" as variable
    from mercury_ml.spark.providers.session import get_or_create_spark_session

    if not spark_session_params:
        spark_session_params = {}

    spark = get_or_create_spark_session(**spark_session_params)

    # Start H2OContext
    h2o_conf = H2OConf(spark)
    h2o_conf.set_h2o_node_log_level(h2o_log_level)

    if not h2o_context_params:
        h2o_context_params = {}

    if h2o_context_params.get("auth"):  # requires h2o-pysparkling>=2.2.28
        h2o_context_params["auth"] = tuple(h2o_context_params["auth"])

    h2o_context = H2OContext.getOrCreate(spark, conf=h2o_conf, **h2o_context_params)

    return h2o_context
