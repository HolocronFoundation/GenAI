
import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# TODO: Do something better than this
USE_GPU = True
if not USE_GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

config = tf.ConfigProto(log_device_placement=False)
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
    return {
        "networks": {
            "generator": generator,
            "discriminator": discriminator,
            "gan": gan_compiler(random_size, generator, discriminator, optimizer=optimizers[2])
        },
        "seed_size": random_size,
        "epoch_current": 0 # TODO: Check this more
    }


def gan_compiler(input_size, generator, discriminator, optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy'):
    discriminator.trainable = False
    gan_input = layers.Input(shape=(input_size,))
    gan_output = discriminator(generator(gan_input))
    gan = keras.Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss=loss, optimizer=optimizer)
    return gan


def MNIST_gan_Test():
    model = build_gan_model(784, [256, 512, 1024], [1024, 512, 256])
    train_gan(model, loadMNISTData(), {"width":10, "height":10}, epoch_total=100)


def loadMNISTData():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_train = x_train.reshape(60000, 784)  # Learn more about this reshape
    return x_train


def plot_generated_images(gan, image, examples=100, dim=(10, 10), figsize=(10, 10)):
    generator = gan["networks"]["generator"]
    noise = np.random.normal(0, 1, size=[examples, gan["seed_size"]])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(
        examples, image["width"], image["height"])
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image_epoch_%d.png' % gan["epoch_current"])


def train_gan(gan, training_x, image, epoch_total=1, batch_size=128, display_function=plot_generated_images):
    # TODO: Make mods?

    batch_count = int(training_x.shape[0] / batch_size)

    for epoch in range(gan["epoch_current"], epoch_total):
        print("Epoch: " + str(epoch))
        for i in tqdm(range(batch_count)):
            # Generates random noise for the seeds used by the generator
            seed_noise = np.random.normal(0, 1, size=[batch_size, gan["seed_size"]])

            # Creates a batch of size batch_size, pulled from a random selection
            # of the training data
            batch = training_x[np.random.randint(
                0, training_x.shape[0], size=batch_size)]

            # Generates images from the noise
            generated = gan["networks"]["generator"].predict(seed_noise)

            # Creates a training batch composed of half generated images, and
            # half real images
            discriminator_training_x = np.concatenate((batch, generated))

            # Creates a corresponding "real/not real" set of data for training,
            # where .9 is assigned to real items and 0 to fake
            discriminator_training_y = np.zeros(2 * batch_size)
            discriminator_training_y[:batch_size] = .9

            # Trains the discriminator on it's own
            gan["networks"]["discriminator"].trainable = True
            gan["networks"]["discriminator"].train_on_batch(discriminator_training_x, discriminator_training_y)
            gan["networks"]["discriminator"].trainable = False

            # Generates new random seed noise to train the generator
            seed_noise = np.random.normal(0, 1, size=[batch_size, gan["seed_size"]])
            # Sets the target values, to be 1 (the generator wants to convinve
            # the discriminator the forgeries are real)
            desired_discriminator_outcome = np.ones(batch_size)
            gan["networks"]["gan"].train_on_batch(seed_noise, desired_discriminator_outcome)
        display_function(gan, image)  # TODO: Changes needed HERE!!!
        gan["epoch_current"] += 1


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
