import pandas as pd
import numpy as np

class KerasIteratorDataWrapper():
    """
    A base class for DataWrappers that have Keras iterators as their underlying
    """

    def __init__(self, underlying, field_names):
        self.underlying = underlying
        self.field_names = field_names


class KerasIteratorIndexDataWrapper(KerasIteratorDataWrapper):
    """
    A DataWrapper with a Keras DirectoryIterator as its underlying. This particular wrapper is used to access index information (in this case the filenames)
    """

    def to_numpy(self):
        """
        Creates a new NumpyDataWrapper based on this KerasIteratorIndexDataWrapper and returns it as a new object

        :return: a new NumpyDataWrapper
        """

        from mercury_ml.common.providers.data_wrappers.numpy import NumpyDataWrapper
        index, index_fields = self._get_numpy_index()
        return NumpyDataWrapper(index, index_fields)

    def to_pandas(self):
        """
        Creates a new PandasDataWrapper based on this KerasIteratorIndexDataWrapper and returns it as a new object

        :return: a new PandasDataWrapper
        """

        from mercury_ml.common.providers.data_wrappers.pandas import PandasDataWrapper
        index, index_fields = self._get_numpy_index()
        return PandasDataWrapper(pd.DataFrame(index, columns=index_fields), index_fields)

    def to_keras_directory_iterator(self):
        return self

    def _get_numpy_index(self):
        """
        Converts self.underlying to a numpy array

        :return: index (the new numpy array), index_fields (a list of of headings for the index, in this case just ["filenames"])
        """

        index = np.array(self.underlying.get_filenames())
        index_fields = ["filenames"]

        return index, index_fields

class KerasIteratorTargetsDataWrapper(KerasIteratorDataWrapper):
    """
    A DataWrapper with a Keras DirectoryIterator as its underlying. This particular wrapper is used to access targets information
    """

    def to_numpy(self):
        """
        Creates a new NumpyDataWrapper based on this KerasIteratorTargetsDataWrapper and returns it as a new object

        :return: a new NumpyDataWrapper
        """

        from mercury_ml.common.providers.data_wrappers.numpy import NumpyDataWrapper
        targets, targets_fields = self._get_numpy_targets()
        return NumpyDataWrapper(targets, targets_fields)

    def to_pandas(self):
        """
        Creates a new PandasDataWrapper based on this KerasIteratorTargetsDataWrapper and returns it as a new object

        :return: a new PandasDataWrapper
        """

        from mercury_ml.common.providers.data_wrappers.pandas import PandasDataWrapper
        targets, targets_fields = self._get_numpy_targets()
        return PandasDataWrapper(pd.DataFrame(targets, columns=targets_fields), targets_fields)

    def to_keras_directory_iterator(self):
        return self

    def _get_numpy_targets(self):
        """
        Converts self.underlying to a numpy array

        :return: targets (the new numpy array), targets_fields (a list of of headings for the targets)
        """

        self.underlying.shuffle = False
        targets = self.underlying.get_target_dummies()
        targets_fields = self.underlying.get_labels_dummies()

        return targets, targets_fields


class KerasIteratorFeaturesDataWrapper(KerasIteratorDataWrapper):
    """
    A DataWrapper with a Keras DirectoryIterator as its underlying. This particular wrapper is used to access features information
    """

    def to_numpy(self):
        """
        Creates a new NumpyDataWrapper based on this KerasIteratorFeaturesDataWrapper and returns it as a new object

        :return: a new NumpyDataWrapper
        """

        from mercury_ml.common.providers.data_wrappers.numpy import NumpyDataWrapper
        features, features_fields = self._get_numpy_features()
        return NumpyDataWrapper(features, features_fields)

    def to_pandas(self):
        """
        Creates a new PandasDataWrapper based on this KerasIteratorFeaturesDataWrapper and returns it as a new object

        :return: a new PandasDataWrapper
        """

        from mercury_ml.common.providers.data_wrappers.pandas import PandasDataWrapper
        features, features_fields = self._get_numpy_features()
        return PandasDataWrapper(pd.DataFrame(features, columns=features_fields), features_fields)

    def to_keras_directory_iterator(self):
        return self

    def _get_numpy_features(self):
        """
        Converts self.underlying to a numpy array

        :return: features (the new numpy array), None (since field_names are not relevant here)
        """

        self.underlying.batch_size = self.underlying.n
        self.underlying.shuffle = False #TODO check if this is sufficient!

        features, _ = self.underlying.next()

        return features, None