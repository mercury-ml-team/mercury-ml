"""ImageGenerator and DirectoryIterator classes used for models with multiple (attribute-type) labels"""

from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
import os
import numpy as np
import pandas as pd
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array, array_to_img


class MultiLabelImageDataGenerator(ImageDataGenerator):
    """
    An image data generator that is able to return multiple labels, based on a LabelMetaStore
    """
    def __init__(self, label_df, *args, **kwargs):
        self.label_df = label_df
        super(MultiLabelImageDataGenerator, self).__init__(*args, **kwargs)

    def flow_from_directory(self, *args, **kwargs):
        kwargs["image_data_generator"] = self
        kwargs["label_df"] = self.label_df
        return MultiLabelDirectoryInterator(*args, **kwargs)


class MultiLabelDirectoryInterator(DirectoryIterator):
    def __init__(self, *args, **kwargs):
        self.label_df = kwargs.pop("label_df")
        super(MultiLabelDirectoryInterator, self).__init__(*args, **kwargs)

    def get_target_dummies(self):
        """
        :return: array, the targets associated with this generator, in one-hot-encoded (dummy) format
        """

        if self.shuffle:
            raise AssertionError(
                "The 'shuffle' parameter should always be switched to 'False' for a generator for which the array of "
                "targets is retrieved!")
        else:
            filenames = [self._split_filenames_vector(filename) for filename in self.filenames]
            targets_df = self.label_df.loc[filenames]
            return targets_df.values

    def get_labels_dummies(self):
        return list(self.label_df.columns)

    def get_labels_classes(self):
        return list(self.label_df.columns)


    def get_target_classes(self):
        """
        :return: array, the targets associated with this generator, in dense format
        """
        # for multi-label models there is no single "class", therefore an array of "-1" values is returned
        return np.full(len(self.classes), -1)

    def get_filenames(self):
        """
        Gets the unique filenames associated with each of the source images in a generator

        :param generator: DirectoryIterator, a generator used to feed the model with training images
        :return: An array of strings
        """

        if self.shuffle:
            raise AssertionError( """The 'shuffle' parameter should always be switched to 'False' for a generator for 
            which the array of filenames is retrieved!""")

        return self.filenames

    def _split_filenames_vector(self, filename):
        _, fname_without_folder = os.path.split(filename)
        fname_name_only, _ = os.path.splitext(fname_without_folder)
        return fname_name_only

    def _load_label_array(self, fname_name_only):
        return self.label_df.loc[fname_name_only].values

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) +
                           self.image_shape, dtype=K.floatx())
        grayscale = self.color_mode == "grayscale"

        # custom parameter
        batch_y = np.zeros(
            (len(index_array), self.label_df.shape[1]), dtype=K.floatx())

        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = load_img(os.path.join(self.directory, fname),
                           grayscale=grayscale,
                           target_size=self.target_size,
                           interpolation=self.interpolation)
            x = img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)

            # custom fetch
            path, fname_without_folder = os.path.split(fname)
            fname_name_only, extension = os.path.splitext(fname_without_folder)
            y = self._load_label_array(fname_name_only)

            batch_x[i] = x
            batch_y[i] = y

        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = "{prefix}_{index}_{hash}.{format}".format(prefix=self.save_prefix,
                                                                  index=j,
                                                                  hash=np.random.randint(1e7),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

        return batch_x, batch_y
