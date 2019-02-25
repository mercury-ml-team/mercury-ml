import shutil
import os
import subprocess
import json

def copy_from_disk_to_disk(source_dir, target_dir, filename, overwrite=False, delete_source=False):
    """
    Moves a file from one location of Disk to another location on Disk

    :param string source_dir: Path of file to be copied
    :param string target_dir: Path that file is to be copied to
    :param string filename: Name of file to be copied
    :param bool overwrite: If true, overwrite if file already exists in target_dir
    :param bool delete_source: If true, delete file from source_dir after copying
    :return:
    """

    source_dir = _make_local_path(source_dir)
    target_dir = _make_local_path(target_dir)

    source_path = os.path.join(source_dir, filename)
    target_path = os.path.join(target_dir, filename)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    if overwrite == False:
        if os.path.exists(target_path):
            pass
            #raise FileExistsError("File already exists, but overwrite was not activated")
        else:
            shutil.copyfile(source_path,target_path)
    else:
        shutil.copyfile(source_path,target_path)

    if delete_source == True:
        # check if copy was successfull:
        if os.path.exists(os.path.join(target_dir, filename)):
            os.remove(os.path.join(source_dir, filename))
        else:
            raise ChildProcessError(
                "Delete Source was activated but copying file to new location failed")


def copy_from_disk_to_hdfs(source_dir, target_dir, filename, overwrite=False, delete_source=False):
    """
    Moves a file from Disk to HDFS

    :param string source_dir: Path of file to be copied
    :param string target_dir: Path that file is to be copied to
    :param string filename: Name of file to be copied
    :param bool overwrite: If true, overwrite if file already exists in target_dir
    :param bool delete_source: If true, delete file from source_dir after copying
    :return:
    """

    # check if directory exists
    exists = subprocess.call(["hadoop", "fs", "-test", "-d", target_dir])

    if exists == 0:  # 0 equals exists
        subprocess.call(["hadoop", "fs", "-copyFromLocal", os.path.join(
            source_dir, filename), os.path.join(target_dir, filename)])
    else:  # create directory first
        subprocess.call(["hadoop", "fs", "-mkdir", "-p", target_dir])
        subprocess.call(["hadoop", "fs", "-copyFromLocal", os.path.join(source_dir, filename),
                         os.path.join(target_dir, filename)])

def copy_from_disk_to_mongo(source_dir, target_dir, filename, database_name, collection_name,
                            mongo_client_params, document_key_separator="/", reuse_existing=True,
                            overwrite=True, delete_source=False):

    source_path = os.path.join(source_dir, filename)

    with open(source_path, "r") as f:
        data = json.load(f)

    from mercury_ml.common.providers.artifact_storage.mongo import store_dict_on_mongo
    document_id, document_key = _get_document_id_and_key_from_path(target_dir)

    store_dict_on_mongo(data=data,
                        document_id=document_id,
                        database_name=database_name,
                        collection_name=collection_name,
                        mongo_client_params=mongo_client_params,
                        document_key=document_key,
                        document_key_separator=document_key_separator,
                        reuse_existing=reuse_existing,
                        overwrite=overwrite)

    if delete_source == True:
        os.remove(os.path.join(source_dir, filename))

def _get_document_id_and_key_from_path(path):
    document_key="/".join(path.strip("/").split('/')[1:])
    document_id="/".join(path.strip("/").split('/')[:1])

    return document_id, document_key

def copy_from_disk_to_s3(source_dir, target_dir, filename, overwrite=False, delete_source=False, s3_session_params=None,
                         reuse_existing=True):
    """
    Moves a file from Disk to S3

    :param string source_dir: Path of file to be copied
    :param string target_dir: Path that file is to be copied to
    :param string filename: Name of file to be copied
    :param bool overwrite: If true, overwrite if file already exists in target_dir
    :param bool delete_source: If true, delete file from source_dir after copying
    :return:
    """
    if not s3_session_params:
        s3_session_params={}

    import boto3
    if not reuse_existing:
        # s3 = boto3.resource("s3")
        session = boto3.Session(**s3_session_params)
        s3 = session.resource("s3")
    else:
        from mercury_ml.common.providers.artifact_copying import S3Singleton
        s3 = S3Singleton(**s3_session_params).s3

    s3_bucket_name, s3_path = target_dir.split("/", 1)
    s3_key = s3_path + "/" + filename

    source_path = os.path.join(source_dir, filename)
    if overwrite or not _s3_key_exists(s3, s3_bucket_name, s3_key):
        s3.Object(s3_bucket_name, s3_key).put(Body=open(source_path, "rb"))

    if delete_source:
        os.remove(source_path)

def copy_from_disk_to_gcs(source_dir, target_dir, filename, overwrite=False, delete_source=False):
    """
    Moves a file from Disk to GCS (Google Cloud Storage)

    :param string source_dir: Path of file to be copied
    :param string target_dir: Path that file is to be copied to
    :param string filename: Name of file to be copied
    :param bool overwrite: If true, overwrite if file already exists in target_dir
    :param bool delete_source: If true, delete file from source_dir after copying
    :return:
    """

    from google.cloud import storage
    storage_client = storage.Client()

    gcs_bucket_name, gcs_path = target_dir.split("/", 1)

    bucket = storage_client.get_bucket(gcs_bucket_name)

    blob = bucket.blob(gcs_path + "/" + filename)
    blob.upload_from_filename(source_dir + "/" + filename )

    if delete_source:
        os.remove(source_dir + "/" + filename)



def _make_local_path(path_name):
    if path_name[0] == ".":
        path_name = os.path.join(os.getcwd(), path_name)
        path_name = os.path.abspath(path_name)
    return path_name

def _s3_key_exists(s3, s3_bucket, s3_key):
    content = s3.head_object(Bucket=s3_bucket, Key=s3_key)
    if content.get('ResponseMetadata', None) is not None:
        return True
    else:
        return False