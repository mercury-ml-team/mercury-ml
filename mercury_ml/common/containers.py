"""
Simple IoC containers that provide direct access to various providers
"""


class SourceReaders:
    """IoC container for SourceStore providers."""

    from mercury_ml.common.providers.source_reading import disk, hive

    read_disk_pandas = disk.read_pandas_data_set
    read_disk_keras_single_input_iterator = disk.read_keras_single_input_image_iterator_data_set
    read_disk_keras_multi_label_iterator = disk.read_keras_multi_label_image_iterator_data_set
    read_hive_spark = hive.read_spark_data_set

class CustomMetrics:
    """IoC container for CustomMetric providers."""

    from mercury_ml.common.providers.metric_evaluation.numpy import metrics as numpy_metrics

    evaluate_numpy_auc = numpy_metrics.evaluate_macro_auc
    evaluate_numpy_micro_auc = numpy_metrics.evaluate_micro_auc
    evaluate_numpy_macro_auc = numpy_metrics.evaluate_macro_auc


class CustomLabelMetrics:
    """IoC container for CustomLabelMetric providers."""
    
    from mercury_ml.common.providers.metric_evaluation.numpy import label_metrics as numpy_label_metrics

    evaluate_numpy_auc = numpy_label_metrics.evaluate_auc
    evaluate_numpy_confusion_matrix = numpy_label_metrics.evaluate_confusion_matrix
    evaluate_numpy_accuracy = numpy_label_metrics.evaluate_accuracy
    evaluate_numpy_multi_label_confusion_matrix = numpy_label_metrics.evaluate_multi_label_confusion_matrix


class LocalArtifactStorers:
    from mercury_ml.common.providers.artifact_storage import local, mongo

    store_pandas_pickle = local.store_pandas_pickle
    store_pandas_json = local.store_pandas_json
    store_dict_json = local.store_dict_json
    store_h2o_frame = local.store_h2o_frame
    store_dict_on_mongo = mongo.store_dict_on_mongo

class ArtifactCopiers:
    """IoC container of FileMover providers for moving saved models from local to remote store"""

    from mercury_ml.common.providers.artifact_copying import from_disk, from_gcs, from_s3, from_hdfs

    copy_from_disk_to_disk = from_disk.copy_from_disk_to_disk

    copy_from_disk_to_hdfs = from_disk.copy_from_disk_to_hdfs
    copy_from_disk_to_s3 = from_disk.copy_from_disk_to_s3
    copy_from_disk_to_gcs = from_disk.copy_from_disk_to_gcs
    copy_from_disk_to_mongo = from_disk.copy_from_disk_to_mongo

    copy_from_s3_to_disk = from_s3.copy_from_s3_to_disk
    copy_from_hdfs_to_disk = from_hdfs.copy_from_hdfs_to_disk
    copy_from_gcs_to_disk = from_gcs.copy_from_gcs_to_disk



