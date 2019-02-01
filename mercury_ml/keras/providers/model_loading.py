def load_hdf5_model(local_dir, filename, extension=".h5", custom_objects=None):
    """
    Loads a Keras model that was saved in ".h5" format

    :param string local_dir: Local directory where the model is saved
    :param string filename: Filename with which the model is saved
    :param string extension: Extension to the filename with which the model is saved
    :param dict custom_objects: Any custom objects  (such as custom loss functions) that were included when the model was saved
    :return: A Keras model
    """
    from keras.models import load_model
    return load_model(local_dir + "/" + filename + extension, custom_objects=custom_objects)
