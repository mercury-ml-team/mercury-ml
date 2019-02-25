class DataSet:
    """
    A class that groups together DataWrappers. A typical DataSets would consist of "full_data", "index" , "features" and "targets" DataWrappers.
    """

    def __init__(self, data_wrappers_dict=None):
        if data_wrappers_dict:
            self.add_data_wrappers(data_wrappers_dict)

    def add_data_wrapper(self, data_wrapper_name, data_wrapper):
        """
        Adds a single DataWrapper to this DataSet
        :param string data_wrapper_name: Name of the DataWrapper
        :param DataWrapper data_wrapper: The DataWrapper to add
        :return:
        """
        setattr(self, data_wrapper_name, data_wrapper)

    def add_data_wrappers(self, data_wrappers_dict):
        """
        Adds DataWrappers to this DataSet. For example

        data_set = DataSet()
        data_bunch.add_data_sets(
            {
                "full_data": full_data_data_wrapper,
                "index": index_data_wrapper,
                "features": features_data_wrapper
                "targets": targets_data_wrapper
            }
        )

        :param dict data_sets_dict: Of the form {{"data_wrapper_name": data_wrapper},...}
        :return:
        """

        for data_wrapper_name, data_wrapper in data_wrappers_dict.items():
            self.add_data_wrapper(data_wrapper_name, data_wrapper)

    def add_data_wrapper_via_concatenate(self, left_data_wrapper_name, right_data_wrapper_name, new_data_wrapper_name):
        """
        Concatenates two DataWrappers and add the result as a new DataWrapper

        :param DataWrapper left_data_wrapper_name: The DataWrapper to concatenate to the left.
        :param DataWrapper right_data_wrapper_name: The DataWrapper to concatenate to the right.
        :param string new_data_wrapper_name: The name of the new DataWrapper.
        :return:
        """

        left_data_wrapper = getattr(self, left_data_wrapper_name)
        right_data_wrapper = getattr(self, right_data_wrapper_name)
        self.add_data_wrapper(new_data_wrapper_name, left_data_wrapper.concatenate(right_data_wrapper))

    def transform(self, transformation_params, transform_then_slice=False):
        """
        Transforms of of the underlying DataWrappers to a new type, adds them to a new DataSet and returns that DataSet.
        :param dict transformation_params: The parameters to be used in the DataWrapper transformations
        :param bool transform_then_slice: Determines whether each DataWrapper within the DataSet should be transformed
        independently (False), or whether only "full_data" should be transformed and the subset the derived from there (True)
        :return:
        """

        if transform_then_slice:
            return self._transform_full_data_and_make_slices(**transformation_params)
        else:
            return self._transform(**transformation_params)

    def _transform(self, transform_to, data_wrapper_params=None):
        """
        Transforms each DataWrapper independently
        """

        if data_wrapper_params is None:
            data_wrapper_params =  {k:{} for k in list(self.__dict__.keys())} #TODO: does this work?

        new_data_set = DataSet()
        for data_wrapper_name, params in data_wrapper_params.items():
            current_data_wrapper = getattr(self, data_wrapper_name)
            transform = getattr(current_data_wrapper, "to_" + transform_to)
            new_data_set.add_data_wrapper(data_wrapper_name, transform(**params))

        return new_data_set

    def _transform_full_data_and_make_slices(self, transform_to, full_data_wrapper_params=None, data_wrapper_names=None):
        """
        Transforms only the underlying full data. Populates features, targets and index by slicing the transformed data
        instead of transforming each separately
        """

        current_full_data_wrapper = self.full_data
        transform = getattr(current_full_data_wrapper, "to_" + transform_to)
        new_full_data_wrapper = transform(**full_data_wrapper_params)

        new_data_set = DataSet({"full_data": new_full_data_wrapper})

        if data_wrapper_names is None:
            data_wrapper_names = list(self.__dict__.keys()).remove("full_data")

        for data_wrapper_name, data_wrapper_value in self.__dict__.items():
            if data_wrapper_name in data_wrapper_names:
                field_names = getattr(self, data_wrapper_name).field_names
                new_data_set.add_data_wrappers(
                    {data_wrapper_name: new_full_data_wrapper.__class__(new_full_data_wrapper.slice_on_column_names(field_names), field_names)})

        return new_data_set