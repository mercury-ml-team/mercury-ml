import os
import json

def store_h2o_frame(data, directory, filename, force=False, parts=1):
    """
    Export a given H2OFrame to a path on the machine this python session is currently connected to.

    :param data: the Frame to save to disk.
    :param directory: the directory to the save point on disk.
    :param filename: the name to save the frame to.
    :param force: if True, overwrite any preexisting file with the same path
    :param parts: enables export to multiple 'part' files instead of just a single file.
        Convenient for large datasets that take too long to store in a single file.
        Use parts=-1 to instruct H2O to determine the optimal number of part files or
        specify your desired maximum number of part files. Path needs to be a directory
        when exporting to multiple files, also that directory must be empty.
        Default is ``parts = 1``, which is to export to a single file.
    :return string filepath: the path to which the file was stored.
    """

    if not os.path.isdir(directory):
        os.makedirs(directory)

    filepath = _make_local_path(os.path.join(directory, filename))

    from h2o.job import H2OJob
    from h2o.utils.typechecks import assert_is_type
    from h2o.frame import H2OFrame
    from h2o import api
    assert_is_type(data, H2OFrame)
    assert_is_type(filepath, str)
    assert_is_type(force, bool)
    assert_is_type(parts, int)
    H2OJob(api("POST /3/Frames/%s/export" % (data.frame_id), data={"path": filepath, "num_parts": parts, "force": force}),
           "Export File").poll()

    return filepath


def store_pandas_pickle(data, directory, filename, compression=None):
    """
    Saves data to Disk as Pickle file.

    :param DataFrame data: The data to be saved
    :param string dir: dir where the data file should be saved
    :param string filename: Name that the data file should be saved with
    :return string filepath: the path to which the file was stored.
    """

    if not compression:
        compression_postfix = ""
    else:
        compression_postfix = "." + compression

    if not os.path.exists(directory):
        os.makedirs(directory)

    filepath = os.path.join(directory, filename + ".pkl")

    data.to_pickle(path=os.path.join(directory, filename + ".pkl" + compression_postfix),
                   compression=compression)

    return filepath


def store_pandas_json(data, directory, filename, orient="table", compression=None):
    """
    Saves data to Disk as JSON file.

    :param DataFrame data: The data to be saved
    :param string dir: dir where the data file should be saved
    :param string filename: Name that the data file should be saved with
    :return string filepath: the path to which the file was stored.
    """

    if compression is None:
        compression_postfix = ""
    else:
        compression_postfix = "." + compression

    if not os.path.exists(directory):
        os.makedirs(directory)

    filepath = os.path.join(directory, filename + ".json")

    data.to_json(path_or_buf=filepath,
                 orient=orient)

    return filepath


def store_dict_json(data, directory, filename):
    """
    Saves data to Disk as JSON file.

    :param dict data: The data to be saved
    :param string dir: dir where the data file should be saved
    :param string filename: Name that the data file should be saved with
    :return string filepath: the path to which the file was stored.
    """

    if not os.path.isdir(directory):
        os.makedirs(directory)

    filepath = os.path.join(directory, filename + ".json")
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    return filepath


def _make_local_path(path_name):
    if path_name[0] == ".":
        path_name = os.path.join(os.getcwd(), path_name)
        path_name = os.path.abspath(path_name)
    return path_name
