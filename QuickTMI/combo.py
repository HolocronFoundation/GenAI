import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tqdm import tqdm

#from tensorflow.keras.models import Model, Sequential
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras import initializers

def buildGANModel(inputShape, generatorLayers=4, discriminatorLayers=4, defaultLayerSize=16, randomSize=100):
    generator = buildDNNModel(randomSize, inputShape, generatorLayers, defaultLayerSize)
    discriminator = buildDNNModel(inputShape, 1, discriminatorLayers, defaultLayerSize) #Change hardcoded output?
    return [generator, discriminator, GANCompiler([randomSize], generator, discriminator)]

def GANCompiler(inputSize, generator, discriminator, optimizer=tf.train.AdadeltaOptimizer()):
    discriminator.trainable = False
    GAN_input = layers.Input(shape=(inputSize,))
    GAN_output = discriminator(generator(GAN_input))
    GAN = keras.Model(inputs=GAN_input, outputs=GAN_output)
    GAN.compile(loss='binary_crossentropy', optimizer=optimizer)
    return GAN

def MNIST_GAN_Test():
    models = buildGANModel(784, [256, 512, 1024], [1024, 512, 256])
    trainGAN(models, loadMNISTTemp(), 100)

def loadMNISTTemp():
    (xTrain, yTrain), (x_test, y_test) = keras.datasets.mnist.load_data()
    xTrain = (xTrain.astype(np.float32) - 127.5)/127.5
    xTrain = xTrain.reshape(60000, 784)
    return [xTrain, yTrain]

def trainGAN(GAN, trainingData, epochs=1, batchSize=128):
    xTrain = trainingData[0]
    yTrain = trainingData[1]

    batchCount = int(xTrain.shape[0] / batchSize)

    for epoch in range(epochs):
        print("Epoch: " + str(epoch))
        for i in tqdm(range(batchCount)):
            noise = np.random.normal(0, 1, size=[batchSize, 100]) #REMOVE HARDCODED 100
            batch = xTrain[np.random.randint(0, xTrain.shape[0], size=batchSize)]

            generated = GAN[0].predict(noise)
            X = np.concatenate([batch, generated])

            yDis = np.zeros(2*batchSize)
            yDis[:batchSize] = .9

            GAN[1].trainable = True
            GAN[1].train_on_batch(X, yDis)

            noise = np.random.normal(0, 1, size=[batchSize, 100]) #REMOVE HARDCODED 100
            yGen = np.ones(batchSize)
            GAN[1].trainable = False

            GAN[2].train_on_batch(noise, yGen)
        
        plotGeneratedImages(epoch, GAN[0])

def plotGeneratedImages(epoch, generator, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, 100]) #REMOVE HARDCODED 100
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image_epoch_%d.png' % epoch)

def buildDNNModel(inputSize, outputSize, hiddenLayers=4, defaultLayerSize=16, finalActivation='tanh', optimizer=tf.train.AdadeltaOptimizer()):
    # layers is either an integer, or an array of nodes per layer

    #Checks that hiddenLayers is a proper input
    if isinstance(hiddenLayers, int):
        hiddenLayers = [defaultLayerSize]*hiddenLayers
    elif not isinstance(hiddenLayers, list):
        raise TypeError('buildDNNModel takes either an integer or a list as an input, you provided an input of type: ' + type(hiddenLayers))
    
    model = models.Sequential()

    #Converts hiddenLayers into a network
    for i in range(len(hiddenLayers)):
        if i == 0:
            model.add(layers.Dense(hiddenLayers[i], input_dim=inputSize, kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))) #note the kernel
        else:
            model.add(layers.Dense(hiddenLayers[i]))
        model.add(layers.LeakyReLU())
        #model.add(layers.BatchNormalization())
        model.add(layers.Dropout(.1))
    model.add(layers.Dense(outputSize, activation=finalActivation))

    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    return model

    #~~~ Current unmarked things - if hiddenlayer is a list, it must be integers.

####

# The dimension of our random noise vector.
random_dim = 100

def train(epochs=1, batch_size=128):
    # Get the training and testing data
    x_train = loadMNISTTemp()[0]
    y_train = loadMNISTTemp()[1]
    # Split the training data into batches of size 128
    batch_count = int(x_train.shape[0] / batch_size)

    # Build our GAN netowrk
    adam = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    generator = buildDNNModel(random_dim, 784, [256, 512, 1024], optimizer=adam)
    discriminator = buildDNNModel(784, 1, [1024, 512, 256], finalActivation='sigmoid', optimizer=adam)
    #gan = get_gan_network(discriminator, random_dim, generator, adam) CHANGED
    gan = GANCompiler(random_dim, generator, discriminator, adam)

    for e in range(1, epochs+1):
        print('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in tqdm(range(batch_count)):
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

            # Generate fake MNIST images
            generated_images = generator.predict(noise)
            X = np.concatenate([image_batch, generated_images])

            # Labels for generated and real data
            y_dis = np.zeros(2*batch_size)
            # One-sided label smoothing
            y_dis[:batch_size] = 0.9

            # Train discriminator
            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)

            # Train generator
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y_gen)

        plotGeneratedImages(e, generator)

if __name__ == '__main__':
    train(400, 128)
