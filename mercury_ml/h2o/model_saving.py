import json
import os

from h2o.utils.typechecks import assert_is_type
from h2o.model.model_base import ModelBase, H2OValueError
import h2o


def save_h2o_model(model, filename, extension=None, local_dir=None, force=False):
    """
    Adapts the "h2o.save_model" method in the h2o module in order to control the name of the saved model

    :param model: An H2O Model
    :param string local_dir: Local directory where the model is to be saved
    :param string filename: Filename with which the model is to be saved
    :param string extension: Extension to the filename with which the model is to be saved
    :param bool force: If true will overwrite existing files
    :return: The full filepath where the model is saved
    """


    if not extension:
        extension = ""

    assert_is_type(model, ModelBase)
    assert_is_type(local_dir, str)
    assert_is_type(force, bool)

    local_dir = os.getcwd() if not local_dir else _make_local_path(local_dir)
    print(local_dir)

    
    filename = filename + extension
    path = os.path.join(local_dir, filename)

    _make_dirs(local_dir)

    final_path = h2o.api("GET /99/Models.bin/%s" % model.model_id, data={"dir": path, "force": force})["dir"]

    return final_path


def save_pojo(model, filename, extension=None, local_dir=None):
    """
    Adapts the "h2o.download_pojo" method in the h2o module in order to control the name of the saved model, and to
    split "pojo" and "pojo_jar" downloads into two different functions

    :param model: An H2O Model
    :param string local_dir: Local directory where the model is to be saved
    :param string filename: Filename with which the model is to be saved
    :param string extension: Extension to the filename with which the model is to be saved
    :return: The full filepath where the model is saved
    """

    if not extension:
        extension = ".java"

    assert_is_type(model, ModelBase)

    local_dir = os.getcwd() if not local_dir else _make_local_path(local_dir)
    filename = filename + extension
    path = os.path.join(local_dir, filename)

    _make_dirs(local_dir)

    if not model.have_pojo:
        raise H2OValueError("Export to POJO not supported")

    final_path = h2o.api("GET /3/Models.java/%s" % model.model_id, save_to=path)

    return final_path


def save_pojo_jar(model, filename, extension=None, local_dir=None):
    """
    adapts the "h2o.download_pojo" method in the h2o module in order to control the name of the saved model, and to
    split "pojo" and "pojo_jar" downloads into two different functions

    :param model: An H2O Model
    :param string local_dir: Local directory where the model is to be saved
    :param string filename: Filename with which the model is to be saved
    :param string extension: Extension to the filename with which the model is to be saved
    :return: The full filepath where the model is saved
    """

    if not extension:
        extension = ".jar"

    assert_is_type(model, ModelBase)

    if not model.have_pojo:
        raise H2OValueError("Export to POJO not supported")

    local_dir = os.getcwd() if not local_dir else _make_local_path(local_dir)
    filename = filename + extension
    path = os.path.join(local_dir, filename)

    _make_dirs(local_dir)

    final_path = h2o.api("GET /3/h2o-genmodel.jar", save_to=path)

    return final_path


def save_mojo(model, filename, extension=None, local_dir=None, force=False):
    """
    adapts the "model_base.ModelBase.save_mojo" method in the h2o module in order to control the name of the saved model, and to
    split "mojo" and "mojo_jar" downloads into two different functions

    :param model: An H2O Model
    :param string local_dir: Local directory where the model is to be saved
    :param string filename: Filename with which the model is to be saved
    :param string extension: Extension to the filename with which the model is to be saved
    :param bool force: If true will overwrite existing files
    :return: The full filepath where the model is saved
    """


    if not extension:
        extension = ".zip"

    assert_is_type(local_dir, str)
    assert_is_type(force, bool)

    if not model.have_mojo:
        raise H2OValueError("Export to MOJO not supported")

    local_dir = os.getcwd() if not local_dir else _make_local_path(local_dir)
    filename = filename + extension
    path = os.path.join(local_dir, filename)

    _make_dirs(local_dir)

    final_path = h2o.api("GET /99/Models.mojo/%s" % model.model_id, data={"dir": path, "force": force})["dir"]

    return final_path


def save_mojo_jar(model, filename, extension=None, local_dir=None, force=False):
    """
    adapts the "model_base.ModelBase.save_mojo" method in the h2o module in order to control the name of the saved model, and to
    split "mojo" and "mojo_jar" downloads into two different functions

    :param model: An H2O Model
    :param string local_dir: Local directory where the model is to be saved
    :param string filename: Filename with which the model is to be saved
    :param string extension: Extension to the filename with which the model is to be saved
    :param bool force: If true will overwrite existing files
    :return: The full filepath where the model is saved
    """



    if not extension:
        extension = ".jar"

    assert_is_type(local_dir, str)
    assert_is_type(force, bool)

    if not model.have_mojo:
        raise H2OValueError("Export to MOJO not supported")

    local_dir = os.getcwd() if not local_dir else _make_local_path(local_dir)
    filename = filename + extension
    path = os.path.join(local_dir, filename)

    _make_dirs(local_dir)

    final_path = h2o.api("GET /3/h2o-genmodel.jar", save_to=path)

    return final_path

def save_model_details(model, filename, extension=None, local_dir=None, force=False):
    """
    adapts the "model_base.ModelBase.save_model_details" method in the h2o module in order to control the name of the saved model

    :param model: An H2O Model
    :param string local_dir: Local directory where the model is to be saved
    :param string filename: Filename with which the model is to be saved
    :param string extension: Extension to the filename with which the model is to be saved
    :param bool force: If true will overwrite existing files
    :return: The full filepath where the model is saved
    """

    if not extension:
        extension = ".jar"


    assert_is_type(local_dir, str)
    assert_is_type(force, bool)

    local_dir = os.getcwd() if not local_dir else _make_local_path(local_dir)
    filename = filename + extension
    path = os.path.join(local_dir, filename)

    _make_dirs(local_dir)

    final_path = h2o.api("GET /99/Models/%s/json" % model.model_id, data={"dir": path, "force": force})["dir"]

    return final_path

def _make_dirs(dir):
    import os
    if not os.path.exists(dir):
        os.makedirs(dir)

def _make_local_path(path_name):
    if path_name[0] == ".":
        path_name = os.path.join(os.getcwd(), path_name)
        path_name = os.path.abspath(path_name)
    return path_name