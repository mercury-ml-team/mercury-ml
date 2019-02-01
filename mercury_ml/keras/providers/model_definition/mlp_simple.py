from keras.layers import Dense
from keras.models import Sequential

def define_model(dense_activation, final_activation, nb_classes, nb_features, seed=1234):
    """
    A small example function for defining a simple (mlp) model. This just serves to illustrate what a "define_model"
    function might look like. It is expected that users will create their own "define_model" functions to use together with
    the other mercury-ml functions

    :param string dense_activation: The activation function to use in the dense layer.
    :param string final_activation: The activation function to use in the final layer.
    :param int nb_classes: The number of classes.
    :param int nb_features: The number of features.
    :param int seed: The random seed.
    :return:
    """

    model = Sequential()
    model.add(Dense(24, activation=dense_activation, input_shape=(nb_features,)))
    model.add(Dense(nb_classes, activation=final_activation))

    return model
