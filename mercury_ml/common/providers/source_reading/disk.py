#TODO possibly split into two modules: data_set readers and data_bunch readers

import pandas as pd

from mercury_ml.common.providers.data_set import DataSet
from mercury_ml.common.providers.data_wrappers.pandas import PandasDataWrapper
import os

def read_pandas_data_set(path, input_format, full_data_columns, index_columns, features_columns, targets_columns):
    """
    Reads a pandas dataset from a local source and creates a DataSet consisting of PandasDataWrappers for full_data
    index, features and targets

    :param string path: The local path from which the Pandas DataFrame should be read.
    :param string input_format: The format (e.g. .csv) of the input data.
    :param list full_data_columns: The full list of columns that should be read from the input file.
    :param list index_columns: A subset of full_data_columns. The columns corresponding to the unique index.
    :param list features_columns: A subset of full_data_columns. The columns corresponding to the features.
    :param list targets_columns: A subset of full_data_columns. The columns corresponding to the targets.
    :return: DataSet consiting of PandasDataWrappers
    """

    if input_format == ".pkl":
        df = pd.read_pickle(path)[full_data_columns]
    elif input_format == ".csv":
        df = pd.read_csv(path, usecols=full_data_columns)
    elif input_format == ".json":
        df = pd.read_json(path)[full_data_columns]
    else:
        raise NotImplementedError("Extension '{}' has not yet been implemented")

    return DataSet(
                {
                    "full_data": PandasDataWrapper(df, full_data_columns),
                    "index": PandasDataWrapper(df[index_columns], index_columns),
                    "features": PandasDataWrapper(df[features_columns], features_columns),
                    "targets": PandasDataWrapper(df[targets_columns], targets_columns)
                }
    )


def read_keras_single_input_image_iterator_data_set(generator_params, iterator_params):
    """
    Reads a Keras SingleInputDirectoryInterator from a local source and creates a DataSet consisting of full_dta, targets and index

    :param dict generator_params: parameters to initialise the SingleInputImageDataGenerator
    :param dict iterator_params: parameters to pass to the SingleInputImageDataGenerator.flow_from_directory in order to get the SingleInputDirectoryInterator

    :return: DataSet consiting of KerasIteratorFeaturesDataWrapper, KerasIteratorTargetsDataWrapper and KerasIteratorIndexDataWrapper
    """

    from mercury_ml.keras.providers.image_generators.single_input import SingleInputImageDataGenerator
    from mercury_ml.common.providers.data_wrappers.keras import KerasIteratorFeaturesDataWrapper, \
        KerasIteratorTargetsDataWrapper, KerasIteratorIndexDataWrapper

    generator = SingleInputImageDataGenerator(**generator_params)
    iterator = generator.flow_from_directory(**iterator_params)

    return DataSet(
                {
                    "features": KerasIteratorFeaturesDataWrapper(iterator, None),
                    "targets": KerasIteratorTargetsDataWrapper(iterator, None),
                    "index": KerasIteratorIndexDataWrapper(iterator, None)
                }
            )


def read_keras_multi_label_image_iterator_data_set(
        label_df_path, index_column, label_columns, generator_params, iterator_params, label_df_read_params=None):
    """
    Reads a Keras MultiLabelDirectoryInterator from a local source and creates a DataSet consisting of full_dta, targets and index

    :param string label_df_path: A local path where a DataFrame that has the multi-label information can be found
    :param list index_column: The name of the column that make up the unique index
    :param list label_columns: The names of the columns that contain the label information
    :param dict generator_params: parameters to initialise the MultiLabelImageDataGenerator
    :param dict iterator_params: parameters to pass to the MultiLabelImageDataGenerator.flow_from_directory in order to get the MultiLabelDirectoryInterator
    :param dict DataFrame label_df_read_params: The parameters according to which the label DataFrame should be read
    :return:
    """

    from mercury_ml.keras.providers.image_generators.multi_label import MultiLabelImageDataGenerator
    from mercury_ml.common.providers.data_wrappers.keras import KerasIteratorFeaturesDataWrapper, \
        KerasIteratorTargetsDataWrapper, KerasIteratorIndexDataWrapper

    extension = os.path.splitext(label_df_path)[1]
    if not label_df_read_params:
        label_df_read_params = {}
    if extension == ".pkl":
        label_df = pd.read_pickle(label_df_path, **label_df_read_params)
    elif extension == ".csv":
        label_df = pd.read_csv(label_df_path, **label_df_read_params)
    elif extension == ".json":
        label_df = pd.read_json(label_df_path, **label_df_read_params)
    else:
        raise NotImplementedError("Extension '{}' has not yet been implemented")

    columns_to_select = [index_column] + label_columns
    # label_df must consist entirely of label columns. Therefore we set an explicit index. This will be used to
    # identify the image that the labels belong to
    label_df = label_df[columns_to_select].set_index(index_column)

    generator = MultiLabelImageDataGenerator(label_df, **generator_params)
    iterator = generator.flow_from_directory(**iterator_params)

    return DataSet(
                {
                    "features": KerasIteratorFeaturesDataWrapper(iterator, None),
                    "targets": KerasIteratorTargetsDataWrapper(iterator, None),
                    "index": KerasIteratorIndexDataWrapper(iterator, None)
                }
            )