from keras.initializers import glorot_normal
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import Sequential

def define_model(input_size, dropout_rate, final_activation, nb_classes, seed=1234):
    """
    A small example function for defining a simple (convolutional) model. This just serves to illustrate what a "define_model"
    function might look like. It is expected that users will create their own "define_model" functions to use together with
    the other mercury-ml functions

    :param list input_size: the shape of the expected input data.
    :param double dropout_rate: The dropout rate.
    :param string final_activation: The activation function to use in the final layer.
    :param int nb_classes: The number of classes.
    :param int seed: The random seed.
    :return: A (defined) Keras model.
    """

    dense_initializer = glorot_normal(seed=seed)
    conv_initializer = glorot_normal(seed=seed)

    model = Sequential()
    model.add(Conv2D(4, kernel_size=(5, 5), strides=(2, 2),
                     kernel_initializer=conv_initializer,
                     input_shape=(input_size[0],
                                  input_size[1], 3)))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(2, activation="relu",
                    kernel_initializer=dense_initializer))
    model.add(Dropout(rate=dropout_rate, seed=seed))

    model.add(Dense(nb_classes, activation=final_activation))

    return model
