import json
import os
import shutil

def save_keras_hdf5(model, filename, local_dir, extension=None):
    """
    Saves a Keras model in .h5 format

    :param model: A Keras model
    :param string local_dir: Local directory where the model is to be saved
    :param string filename: Filename with which the model is to be saved
    :param string extension: Extension to the filename with which the model is to be saved
    :return: The filepath where the model is saved
    """

    if not extension:
        extension = ".h5"

    _make_dirs(local_dir)
    filename = filename + extension
    path = os.path.join(local_dir + "/" + filename)
    model.save(path)

    return path


def save_tensorflow_saved_model_archived(model, local_dir, filename, extension=".zip", model_number="1", temp_base_dir=None):

    if not temp_base_dir:
        temp_base_dir = os.path.join(os.getcwd(), "_tmp_model", filename)

    root_dir=save_tensorflow_saved_model(model=model,
                                         local_dir=temp_base_dir,
                                         filename=filename,
                                         model_number=model_number)

    if not os.path.isdir(local_dir):
        os.makedirs(local_dir)

    shutil.make_archive(
        base_name=os.path.join(local_dir, filename),
        format=extension[1:],
        root_dir=root_dir,
        base_dir=model_number)

    shutil.rmtree(root_dir, ignore_errors=True)

    return os.path.join(local_dir, filename + extension)


def save_tensorflow_saved_model(model, local_dir, filename, model_number="1"):

    from tensorflow.saved_model import save

    root_dir = os.path.join(local_dir, filename)
    sub_dir = os.path.join(root_dir, model_number)

    if not os.path.isdir(sub_dir):
        os.makedirs(sub_dir)

    save(model, sub_dir)

    return root_dir

def _make_dirs(dir):
    import os
    if not os.path.exists(dir):
        os.makedirs(dir)
