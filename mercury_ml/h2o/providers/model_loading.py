def load_h2o_model(local_dir, filename, extension=""):
    """
    Loads a saved H2O Model

    :param string local_dir: Local directory where the model is saved
    :param string filename: Filename with which the model is saved
    :param string extension: Extension to the filename with which the model is saved
    :return:
    """

    from h2o import load_model
    return load_model(local_dir + "/" + filename + extension)

def load_mojo_model(local_dir, filename, extension=""):
    """
    Loads a saved H2OMOJOModel (can be used with Spark without a running H2OSparkling Session)

    :param string local_dir: Local directory where the model is saved
    :param string filename: Filename with which the model is saved
    :param string extension: Extension to the filename with which the model is saved
    :return:
    """

    from pysparkling.ml import H2OMOJOModel
    return H2OMOJOModel.create_from_mojo(local_dir + "/" + filename + extension)

