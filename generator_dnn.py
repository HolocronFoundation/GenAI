
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models


def build_dnn_model(input_size, output_size, hidden_layers=4, default_layer_size=16, final_activation='tanh', optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5), batch_normalization=False):
    # layers is either an integer, or an array of nodes per layer

    # Checks that hidden_layers is a proper input
    if isinstance(hidden_layers, int):
        hidden_layers = [default_layer_size] * hidden_layers
    elif not isinstance(hidden_layers, list):
        raise TypeError(
            'build_dnn_model takes either an integer or a list as an input, you provided an input of type: ' + type(hidden_layers))

    model = models.Sequential()

    # Converts hidden_layers into a network
    for i, layer in enumerate(hidden_layers):
        if i == 0:
            model.add(layers.Dense(layer, input_dim=input_size,
                                   kernel_initializer=keras.initializers.RandomNormal(stddev=0.02)))  # note the kernel
        else:
            model.add(layers.Dense(layer))
        if batch_normalization:
            model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(.1))
    model.add(layers.Dense(output_size, activation=final_activation))

    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    return model

    # ~~~ Current unmarked things - if hiddenlayer is a list,
