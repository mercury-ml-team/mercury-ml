"""ImageGenerator and DirectoryIterator classes used for models with single inputs"""

from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
import pandas as pd


class SingleInputImageDataGenerator(ImageDataGenerator):
    """
    Wraps the vanilla Keras ImageDataGenerator into a slightly altered class definition
    """
    def __init__(self, *args, **kwargs):
        super(SingleInputImageDataGenerator, self).__init__(*args, **kwargs)

    def flow_from_directory(self, *args, **kwargs):
        kwargs["image_data_generator"] = self
        return SingleInputDirectoryInterator(*args, **kwargs)


class SingleInputDirectoryInterator(DirectoryIterator):
    def __init__(self, *args, **kwargs):
        super(SingleInputDirectoryInterator, self).__init__(*args, **kwargs)

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
            raise AssertionError("""The 'shuffle' parameter should always be switched to 'False' for a generator for which the array of filenames is retrieved!""")

        return self.filenames


