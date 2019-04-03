def store_hive_table(data, directory, file_name):
    """
    Saves predictions to HIVE

    :param DataFrame prediction_df: The predictions to be saved
    :param string directory: Path where the prediction file should be saved
    :param string file_name: Name that the prediction file should be saved with
    :return: None
    """
    table_name = directory + "." + file_name
    data.write.saveAsTable(table_name)

