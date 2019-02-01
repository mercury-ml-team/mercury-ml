import os

def copy_from_gcs_to_disk(source_dir, target_dir, filename, overwrite=False, delete_source=False):
    """
    Moves a file from GCS to Disk

    :param string source_dir: Path of file to be copied
    :param string target_dir: Path that file is to be copied to
    :param string filename: Name of file to be copied
    :param bool overwrite: If true, overwrite if file already exists in target_dir
    :param bool delete_source: If true, delete file from source_dir after copying
    :return:
    """

    if not overwrite and os.path.isfile(target_dir + "/" + filename):
        pass
    else:
        from google.cloud import storage
        storage_client = storage.Client()

        gcs_bucket_name, gcs_path = target_dir.split("/", 1)

        bucket = storage_client.get_bucket(gcs_bucket_name)

        blob = bucket.blob(gcs_path + "/" + filename)
        blob.download_to_filename(source_dir + "/" + filename)

        if delete_source:
            blob.delete()

