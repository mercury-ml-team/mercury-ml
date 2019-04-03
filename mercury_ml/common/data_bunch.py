class DataBunch:
    """
    A class that groups together DataSets. A typical DataBunch would consist of "train", "valid" and "test" DataSets.
    """

    def __init__(self, data_sets_dict=None):
        if data_sets_dict:
            self.add_data_sets(data_sets_dict)

    def add_data_sets(self, data_sets_dict):
        """
        Adds datasets to this DataBunch. For example

        data_bunch = DataBunch()
        data_bunch.add_data_sets(
            {
                "train": train_data_set,
                "valid": valid_data_set,
                "test": test_data_set
            }
        )

        :param dict data_sets_dict: Of the form {{"data_set_name": data_set},...}
        :return:
        """

        for data_set_name, data_set in data_sets_dict.items():
            self.add_data_set(data_set_name, data_set)

    def add_data_set(self, data_set_name, data_set):
        """
        Adds a single DataSet to this DataBunch.

        :param string data_set_name: name of the DataSet.
        :param DataSet data_set: the DataSet to add.
        :return:
        """

        setattr(self, data_set_name, data_set)


    def transform(self, data_set_names, params, transform_then_slice=None):
        """
        Transforms all the underlying DataSets contained in the list data_set_names and return a new DataBunch with the
        transformed datasets

        :param list data_set_names: Names of the DataSets that should be transformed
        :param dict params: Parameters according to which the DataSets should be transformed
        :param bool transform_then_slice: Passed to DataSet.transform. Determines whether each DataWrapper within the DataSet should
        be transformed independently, or whether only "full_data" should be transformed and the subset the derived from there
        :return: new DataBunch
        """

        output_data_bunch = DataBunch()
        for data_set_name in data_set_names:
            input_data_set = getattr(self, data_set_name)
            output_data_bunch.add_data_set(
                data_set_name=data_set_name,
                data_set=input_data_set.transform(transform_then_slice=transform_then_slice,
                                                  transformation_params=params))

        return output_data_bunch

