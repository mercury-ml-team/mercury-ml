"""
ImageGenerator and DirectoryIterator classes used for models with multiple inputs
"""

from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
import pandas as pd


class MultiInputImageDataGenerator(ImageDataGenerator):
    """
    In image data generator that returns multiple inputs
    """
    def __init__(self, *args, **kwargs):
        super(MultiInputImageDataGenerator, self).__init__(*args, **kwargs)

    def flow_from_directory(self, *args, **kwargs):
        kwargs["image_data_generator"] = self
        return MultiInputDirectoryInterator(*args, **kwargs)


class MultiInputDirectoryInterator(DirectoryIterator):
    def __init__(self, *args, **kwargs):
        self.number_of_input_streams = kwargs.pop("number_of_input_streams")
        super(MultiInputDirectoryInterator, self).__init__(*args, **kwargs)

    # override function
    def _get_batches_of_transformed_samples(self, index_array):
        next_batch = super(MultiInputDirectoryInterator,
                           self)._get_batches_of_transformed_samples(index_array)
        next_x = next_batch[0]
        next_y = next_batch[1]

        list_next_x = []
        for i in range(self.number_of_input_streams):
            list_next_x = list_next_x + [next_x]

        return list_next_x, next_y

    def get_labels_dummies(self):
        classes_lookup = {v: k for k, v in self.class_indices.items()}
        return [v for k,v in sorted(classes_lookup.items())] #returns a list, sorted on the class number. #TODO test this

    def get_labels_classes(self):
        return ["class"]

    def get_target_dummies(self):
        """
        :return: array, the targets associated with this generator, in one-hot-encoded (dummy) format
        """
        if self.shuffle:
            raise AssertionError(
                "The 'shuffle' parameter should always be switched to 'False' for a generator for which the array of targets is retrieved!")
        else:
            df = pd.get_dummies(pd.Series(self.classes)).to_dense()
            return df.values

    def get_target_classes(self):
        """
        :return: array, the targets associated with this generator, in dense format
        """
        if self.shuffle:
            raise AssertionError(
                "The 'shuffle' parameter should always be switched to 'False' for a generator for which the array of targets is retrieved!")

        else:
            return self.classes

    def get_filenames(self):
        """
        Gets the unique filenames associated with each of the source images in a generator

        :param generator: DirectoryIterator, a generator used to feed the model with training images
        :return: An array of strings
        """

        if self.shuffle:
            raise AssertionError(
                """The 'shuffle' parameter should always be switched to 'False' for a generator for which the array of filenames is retrieved!""")

        return self.filenames