import json
import os

def save_keras_hdf5(model, filename, local_dir, extension=None):
    """
    Saves a Keras model in .h5 format

    :param model: A Keras model
    :param string local_dir: Local directory where the model is to be saved
    :param string filename: Filename with which the model is to be saved
    :param string extension: Extension to the filename with which the model is to be saved
    :return: The filepath where the model is saved
    """

    if not extension:
        extension = ".h5"

    _make_dirs(local_dir)
    filename = filename + extension
    path = os.path.join(local_dir + "/" + filename)
    model.save(path)

    return path

def save_tensorflow_serving_predict_signature_def(model, local_dir, filename, input_name="input", output_name="output",
                                                  extension=".zip", model_number="1", do_save_labels_txt=False,
                                                  labels_list=None, temp_base_dir=None):
    """
    Saves a Kers model in the format expected by TensorFlow serving (put various objects in a folder structure and
    zips it)

    :param model: A Keras model
    :param string local_dir: Local directory where the model is to be saved
    :param string filename: Filename with which the model is to be saved
    :param input_name: The name of the input tensors
    :param output_name: The name of the output tensors
    :param string extension: Extension to the filename with which the model is to be saved
    :param string model_number: The model version number
    :param bool do_save_labels_txt: In true a txt file with the model labels will be included in with the saved model files
    :param list labels_list: The labels to be saved
    :param string temp_base_dir: Temporary local directory where the model objects are to be saved. They will be compressed and moved
    to local_dir thereafter
    :return: The filepath where the model is saved
    """

    from keras import backend as K
    from tensorflow.python.saved_model import builder as saved_model_builder
    from tensorflow.python.saved_model.signature_def_utils import predict_signature_def
    from tensorflow.python.saved_model import tag_constants

    if not temp_base_dir:
        temp_base_dir = os.path.join(os.getcwd(), "_tmp_model", filename)

    if not os.path.isdir(temp_base_dir):
        os.makedirs(temp_base_dir)

    # Set the learning phase to Test since the model is already trained.
    K.set_learning_phase(0)

    temp_dir = os.path.join(temp_base_dir, model_number)

    builder = saved_model_builder.SavedModelBuilder(temp_dir)

    # Create prediction signature to be used by TensorFlow Serving Predict API
    signature = predict_signature_def(inputs={input_name: model.input},
                                      outputs={output_name: model.output})

    sess = K.get_session()
    # with K.get_session() as sess:
    # Save the meta graph and the variables
    builder.add_meta_graph_and_variables(sess=sess, tags=[tag_constants.SERVING],
                                         signature_def_map={"predict": signature, "serving_default": signature})

    builder.save()

    if do_save_labels_txt:
        save_labels_txt(filename="labels", local_dir=temp_dir, labels_list=labels_list)

    # create zip file
    import shutil
    shutil.make_archive(os.path.join(local_dir, filename), extension[1:], temp_base_dir)

    # remove temporary folder
    shutil.rmtree(temp_base_dir, ignore_errors=True)

    return os.path.join(local_dir, filename + extension)



def save_tensorflow_graph(model, filename, local_dir, num_outputs = 1, quantize = False, extension=None):
    """
    Saves a Keras model as a TensorFlow graph

    :param model: A Keras model
    :param string local_dir: Local directory where the model is to be saved
    :param string filename: Filename with which the model is to be saved
    :param int num_outputs: The number of outputs in the model
    :param bool quantize: If set to true will try to quantize the graph (could improve performance)
    :param string extension: Extension to the filename with which the model is to be saved
    :return: The filepath where the model is saved
    """

    if not extension:
        extension = ".pb"

    _make_dirs(local_dir)

    filename = filename + extension
    path = os.path.join(local_dir + "/" + filename)
    
    tf_graph = _get_tensorflow_graph(model, num_outputs, quantize)

    from tensorflow.python.framework import graph_io
    graph_io.write_graph(tf_graph, local_dir, filename, as_text=False)

    return path

def save_tensorrt_pbtxt_config(model, filename, local_dir, model_base_name, extension=None):
    """
    Derives a pbtxt config from a Keras model and saves it

    :param model: A Keras model
    :param string local_dir: Local directory where the model is to be saved
    :param string filename: Filename with which the model is to be saved
    :param model_base_name: The base name of the model, to be used in the pbtxt config
    :param string extension: Extension to the filename with which the model is to be saved
    :return: The filepath where the model is saved
    """

    if not extension:
        extension = ".pbtxt"

    _make_dirs(local_dir)

    filename = filename + extension
    path = os.path.join(local_dir + "/" + filename)
    
    config = _get_tensorrt_pbtxt_config_storage_model_format(model, model_base_name)

    with open(path, "w") as f:
        f.write(config)

    return path

def save_tensorrt_json_config(model, filename, local_dir, extension=None):
    """
    Derives a json config from a Keras model and saves it

    :param model: A Keras model
    :param string local_dir: Local directory where the model is to be saved
    :param string filename: Filename with which the model is to be saved
    :param string extension: Extension to the filename with which the model is to be saved
    :return: The filepath where the model is saved
    """

    if not extension:
        extension = ".json"

    _make_dirs(local_dir)
    filename = filename + extension
    path = os.path.join(local_dir + "/" + filename)
    
    config = _get_tensorrt_json_config_storage_model_format(model, filename)
    with open(path, "w") as f:
        _ = json.dump(config, f, sort_keys=True, indent=2)

    return path


def save_labels_txt(filename, local_dir, labels_list, model=None, extension=None):
    """

    :param string local_dir: Local directory where the model is to be saved
    :param string filename: Filename with which the model is to be saved
    :param labels_list:
    :param model:
    :param string extension: Extension to the filename with which the model is to be saved
    :return: The filepath where the model is saved
    """

    if not extension:
        extension = ".txt"

    _make_dirs(local_dir)

    filename = filename + extension
    path = os.path.join(local_dir + "/" + filename)
    
    with open(path, "w") as f:
        for item in labels_list:
            f.write("%s\n" % item)

    return path



def _get_tensorflow_graph(model, num_outputs = 1, quantize = False):
    output_node_prefix = "output_node" #TODO check if this can be replaced with just using the Keras output name?
    import tensorflow as tf
    from keras import backend as K

    K.set_learning_phase(0)

    pred = [None] * num_outputs
    pred_node_names = [None] * num_outputs
    for i in range(num_outputs):
        pred_node_names[i] = output_node_prefix + str(i)
    pred[i] = tf.identity(model.outputs[i], name=pred_node_names[i])
    print('output nodes names are: ', pred_node_names)

    sess = K.get_session()

    from tensorflow.python.framework import graph_util

    if quantize:
        from tensorflow.tools.graph_transforms import TransformGraph
        transforms = ["quantize_weights", "quantize_nodes"]
        transformed_graph_def = TransformGraph(sess.graph.as_graph_def(), [], pred_node_names, transforms)
        constant_graph = graph_util.convert_variables_to_constants(sess, transformed_graph_def, pred_node_names)
    else:
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)

    return constant_graph

def _get_tensorrt_pbtxt_config_storage_model_format(model, model_base_name):
    input_name = model.input.name.split(":")[0]
    output_name = model.output.name.split(":")[0]
    model_type = "tensorflow_graphdef"
    input_dims = str(list(model.layers[0].input_shape[1:]))
    output_dims = str(list(model.layers[-1].output_shape[1:]))
    config = \
        'name: "{}"\n'.format(model_base_name) + \
        'platform: "{}"\n'.format(model_type) + \
        'max_batch_size: 128\n' + \
        'input [\n' + \
        '   {\n' + \
        '       name: "{}"\n'.format(input_name) + \
        '       data_type: TYPE_FP32\n' + \
        '       model_format: model_format_NHWC\n' + \
        '       dims: {}\n'.format(input_dims) + \
        '   }\n' + \
        ']\n' + \
        'output [\n' + \
        '   {\n' + \
        '       name: "{}"\n'.format(output_name) + \
        '       data_type: TYPE_FP32\n' + \
        '       dims: {}\n'.format(output_dims) + \
        '       label_filename: "labels.txt"\n' + \
        '   }\n' + \
        ']\n'

    return config


def _get_tensorrt_json_config_storage_model_format(model, filename):
    input_name = model.input.name.split(":")[0]
    output_name = model.output.name.split(":")[0]
    model_type = "tensorflow_graphdef"
    input_dims = list(model.layers[0].input_shape[1:])
    output_dims = list(model.layers[-1].output_shape[1:])
    config = {
                "name": filename,
                "platform": model_type,
                "max_batch_size": 128,
                "input": [
                   {
                       "name": input_name,
                       "data_type": "TYPE_FP32",
                       "model_format": "model_format_NHWC",
                       "dims": input_dims
                   }
                ],
                "output": [
                   {
                       "name": output_name,
                       "data_type": "TYPE_FP32",
                       "dims": output_dims,
                       "label_filename": "labels.txt"
                   }
                ]
            }

    return config


def _make_dirs(dir):
    import os
    if not os.path.exists(dir):
        os.makedirs(dir)
