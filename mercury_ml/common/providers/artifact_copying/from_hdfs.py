import os
import subprocess

def copy_from_hdfs_to_disk(source_dir, target_dir, filename, overwrite=False, delete_source=False):
    """
    Moves a file from HDFS to Disk

    :param string source_dir: Path of file to be copied
    :param string target_dir: Path that file is to be copied to
    :param string filename: Name of file to be copied
    :param bool overwrite: If true, overwrite if file already exists in target_dir
    :param bool delete_source: If true, delete file from source_dir after copying
    :return:
    """

    # TODO change this to use the Python "hdfs" class
    subprocess.call(["hdfs", "dfs", "-copyToLocal", os.path.join(source_dir,
                                                                 filename), os.path.join(target_dir, filename)])
