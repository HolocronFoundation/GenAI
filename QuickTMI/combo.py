
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)


def build_gan_model(input_shape, layer_config=None, default_layer_size=16,
                    outputs=1, random_size=100, final_activation='sigmoid',
                    optimizers=None):

    if layer_config is None:
        layer_config = {"generator": 4, "discriminator": 4}
    if optimizers is None:
        optimizers = [keras.optimizers.Adam(lr=0.0002, beta_1=0.5),
                      keras.optimizers.Adam(lr=0.0002, beta_1=0.5),
                      keras.optimizers.Adam(lr=0.0002, beta_1=0.5)]

    generator = build_dnn_model(random_size, input_shape, layer_config["generator"],
                                default_layer_size, optimizer=optimizers[0],
                                batch_normalization=True)
    discriminator = build_dnn_model(input_shape, outputs, layer_config["discriminator"],
                                    default_layer_size, final_activation=final_activation,
                                    optimizer=optimizers[1])
    return [generator, discriminator, gan_compiler(random_size, generator, discriminator, optimizer=optimizers[2]), random_size]


def gan_compiler(input_size, generator, discriminator, optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy'):
    discriminator.trainable = False
    gan_input = layers.Input(shape=(input_size,))
    gan_output = discriminator(generator(gan_input))
    gan = keras.Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss=loss, optimizer=optimizer)
    return gan


def MNIST_gan_Test():
    model = build_gan_model(784, [256, 512, 1024], [1024, 512, 256])
    train_gan(model, loadMNISTData(), {"width":10, "height":10}, epochs=100)


def loadMNISTData():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_train = x_train.reshape(60000, 784)  # Learn more about this reshape
    return x_train


def plot_generated_images(gan, epoch, image, examples=100, dim=(10, 10), figsize=(10, 10)):
    generator = gan[0]
    noise = np.random.normal(0, 1, size=[examples, gan[3]])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(
        examples, image["width"], image["height"])
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image_epoch_%d.png' % epoch)


def train_gan(gan, training_x, image, epochs=1, batch_size=128, display_function=plot_generated_images):
    # [[Turn gan into dictionary instead of array
    batch_count = int(training_x.shape[0] / batch_size)

    for epoch in range(epochs):
        print("Epoch: " + str(epoch))
        for i in tqdm(range(batch_count)):
            print()
            noise = np.random.normal(0, 1, size=[batch_size, gan[3]])
            batch = training_x[np.random.randint(
                0, training_x.shape[0], size=batch_size)]

            generated = gan[0].predict(noise)
            x = np.concatenate([batch, generated])

            y_dis = np.zeros(2 * batch_size)
            y_dis[:batch_size] = .9

            gan[1].trainable = True
            gan[1].train_on_batch(x, y_dis)

            noise = np.random.normal(0, 1, size=[batch_size, gan[3]])
            y_gen = np.ones(batch_size)
            gan[1].trainable = False
            gan[2].train_on_batch(noise, y_gen)

        display_function(gan, epoch, image)  # TODO: Changes needed HERE!!!


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
            model.add(layers.Dense(layer, input_dim=input_size, kernel_initializer=keras.initializers.RandomNormal(
                stddev=0.02)))  # note the kernel
        else:
            model.add(layers.Dense(layer))
        if batch_normalization:
            model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(.1))
    model.add(layers.Dense(output_size, activation=final_activation))

    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    return model

    # ~~~ Current unmarked things - if hiddenlayer is a list, it must be integers.
