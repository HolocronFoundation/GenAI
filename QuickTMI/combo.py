import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tqdm import tqdm
from copy import copy

#from tensorflow.keras.models import Model, Sequential
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras import initializers

def buildGANModel(inputShape, generatorLayers=4, discriminatorLayers=4, defaultLayerSize=16, outputs=1, randomSize=100, finalActivation='sigmoid',
                  optimizers=[keras.optimizers.Adam(lr=0.0002, beta_1=0.5),keras.optimizers.Adam(lr=0.0002, beta_1=0.5), keras.optimizers.Adam(lr=0.0002, beta_1=0.5)]):
    
    generator = buildDNNModel(randomSize, inputShape, generatorLayers, defaultLayerSize, optimizer=optimizers[0])
    discriminator = buildDNNModel(inputShape, outputs, discriminatorLayers, defaultLayerSize, finalActivation=finalActivation, optimizer=optimizers[1])
    return [generator, discriminator, GANCompiler(randomSize, generator, discriminator, optimizer=optimizers[2]), randomSize]

def GANCompiler(inputSize, generator, discriminator, optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy'):
    discriminator.trainable = False
    GAN_input = layers.Input(shape=(inputSize,))
    GAN_output = discriminator(generator(GAN_input))
    GAN = keras.Model(inputs=GAN_input, outputs=GAN_output)
    GAN.compile(loss=loss, optimizer=optimizer)
    return GAN

def MNIST_GAN_Test():
    models = buildGANModel(784, [256, 512, 1024], [1024, 512, 256])
    trainGAN(models, loadMNISTData(), epochs=100)

def loadMNISTData():
    (xTrain, yTrain), (x_test, y_test) = keras.datasets.mnist.load_data()
    xTrain = (xTrain.astype(np.float32) - 127.5)/127.5
    xTrain = xTrain.reshape(60000, 784) #Learn more about this reshape
    return [xTrain, yTrain]

def trainGAN(GAN, trainingData, epochs=1, batchSize=128):
    xTrain = trainingData[0]
    yTrain = trainingData[1]

    batchCount = int(xTrain.shape[0] / batchSize)

    for epoch in range(epochs):
        print("Epoch: " + str(epoch))
        for i in tqdm(range(batchCount)):
            print()
            noise = np.random.normal(0, 1, size=[batchSize, GAN[3]])
            batch = xTrain[np.random.randint(0, xTrain.shape[0], size=batchSize)]

            generated = GAN[0].predict(noise)
            X = np.concatenate([batch, generated])

            yDis = np.zeros(2*batchSize)
            yDis[:batchSize] = .9

            GAN[1].trainable = True
            GAN[1].train_on_batch(X, yDis)

            noise = np.random.normal(0, 1, size=[batchSize, GAN[3]])
            yGen = np.ones(batchSize)
            GAN[1].trainable = False
            GAN[2].train_on_batch(noise, yGen)
        
        plotGeneratedImages(epoch, GAN[0], (28, 28))

def plotGeneratedImages(epoch, generator, imageDimension, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, dim[0]*dim[1]])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, imageDimension[0], imageDimension[1])
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image_epoch_%d.png' % epoch)

def buildDNNModel(inputSize, outputSize, hiddenLayers=4, defaultLayerSize=16, finalActivation='tanh', optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5)):
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
        model.add(layers.Dropout(.1))
    model.add(layers.Dense(outputSize, activation=finalActivation))

    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    return model

    #~~~ Current unmarked things - if hiddenlayer is a list, it must be integers.
