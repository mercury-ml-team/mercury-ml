import os

def copy_from_s3_to_disk(source_dir, target_dir, filename, overwrite=False, delete_source=False, s3_session_params=None,
                         reuse_existing=True):
    """
    Moves a file from S3 to Disk

    :param string source_dir: Path of file to be copied
    :param string target_dir: Path that file is to be copied to
    :param string filename: Name of file to be copied
    :param bool overwrite: If true, overwrite if file already exists in target_dir
    :param bool delete_source: If true, delete file from source_dir after copying
    :param dict s3_session_params: a dictionary of parameters used to initialise a new S3 session
    :param bool reuse_existing_s3_session: If set to True an existing S3 session will be reused instead of opening a new one
    :return:
    """

    if not s3_session_params:
        s3_session_params={}

    if not overwrite and os.path.isfile(target_dir + "/" + filename):
        pass
    else:
        import boto3
        if not reuse_existing:
            session = boto3.Session(**s3_session_params)
            s3 = session.resource("s3")
        else:
            from mercury_ml.common.providers.artifact_copying import S3Singleton
            s3 = S3Singleton(**s3_session_params).s3

        s3_bucket_name, s3_path = source_dir.split("/", 1)

        s3.meta.client.download_file(s3_bucket_name,
                                     s3_path + "/" + filename,
                                     target_dir + "/" + filename)

        if delete_source:
            s3.delete_objects(Bucket=s3_bucket_name, Delete={'Objects': [s3_path + "/" + filename]})

