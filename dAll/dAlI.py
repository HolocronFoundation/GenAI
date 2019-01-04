import sys, os
import numpy as np
from PIL import Image

sys.path.append('C:/Users/Samuel Troper/Documents/GitHub/QuickTMI/QuickTMI')

import combo

currentDir = 'C:/Users/Samuel Troper/Documents/GitHub/QuickTMI/dAll'

outputDir ='D:/dAlI Output'

largestWidth = 0
largestHeight = 0

def JPGExtension(fileName):
    if ".jpg" in fileName.lower():
        return True
    return False

def PNGExtension(fileName):
    if ".png" in fileName.lower():
        return True
    return False

def normalizeInput(rawImage):
    normalizedImage = []
    for i in range(len(rawImage)):
        if i is 0:
            normalizedImage.append((rawImage[i]/largestWidth)*2-1)
        elif i is 1:
            normalizedImage.append((rawImage[i]/largestHeight)*2-1)
        else:
            normalizedImage.append((rawImage[i]/255)*2-1)
    return normalizedImage

def prepInput(inputDirectory="C:/Users/Samuel Troper/Pictures/Im-DB",
              newWidth=100, newHeight=100):
    directoryList = list(filter(JPGExtension, os.listdir(inputDirectory)))
    directoryList.extend(list(filter(PNGExtension, os.listdir(inputDirectory))))
    images = []
    iteration = 0
    global largestWidth
    global largestHeight
    for image in directoryList:
        #if iteration > 100: #REMOVE 4 REAL
            #break
        print(iteration)
        print(image)
        iteration += 1
        currentImage = Image.open(inputDirectory + "/" + image)
        width = currentImage.size[0]
        height = currentImage.size[1]
        if width > largestWidth:
            largestWidth = width
        if height > largestHeight:
            largestHeight = height
        currentImageInfo = [width, height]
        currentImage = currentImage.resize((newWidth,newHeight))
        for pixel in list(currentImage.convert('RGB').getdata()):
            currentImageInfo.extend(pixel)
        images.append(currentImageInfo)
    normalizedImages = list(map(normalizeInput, images))
    #print("Ok")
    #for i in range(10):
    #    displayImage(normalizedImages[i], newWidth, newHeight)
    return normalizedImages

def displayImage(image, newWidth, newHeight, epoch, iteration):
    temp = Image.new('RGB', (newWidth, newHeight))
    temp.putdata(linearToRGB(image[2:]))
    adjustedWidth = int(largestWidth*(image[0]+1)/2)
    adjustedHeight = int(largestHeight*(image[1]+1)/2)
    if adjustedWidth > 0 and adjustedHeight > 0:
        filename = outputDir+'/epoch' + str(epoch) + '/' + str(iteration) + '.png'
        temp = temp.resize((adjustedWidth, adjustedHeight))
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        #temp.show()
        temp.save(filename)

def linearToRGB(image):
    newImage = []
    currentPixel = []
    for i in range(len(image)):
        currentPixel.append(int((image[i]+1)*255/2))
        if i%3 is 2:
            newImage.append(tuple(currentPixel))
            currentPixel = []
    return tuple(newImage)

def generateAndSave(GAN, epoch, imageDimension=(88,88), number=5):
    generator = GAN[0]
    if epoch % 500 == 0:
        filename = outputDir+'/epoch' + str(epoch) + '/GAN[0].h5'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        GAN[0].save(filename)
        filename = outputDir+'/epoch' + str(epoch) + '/GAN[1].h5'
        GAN[1].save(filename)
        filename = outputDir+'/epoch' + str(epoch) + '/GAN[2].h5'
        GAN[2].save(filename)
    for i in range(number):
        noise = np.random.normal(0, 1, size=[1, GAN[3]])
        generatedImage = generator.predict(noise)[0]
        displayImage(generatedImage, imageDimension[0], imageDimension[1], epoch, i)
    

def test_GAN(newWidth=88, newHeight=88):
    models = combo.buildGANModel(newWidth*newHeight*3+2, [128, 256, 512, 1024, 2048, 4096], [1024, 512, 256])
    data = prepInput(newWidth=newWidth, newHeight=newHeight)
    data = np.array(data)
    combo.trainGAN(models, data, epochs=10000, batchSize=64, displayFunction=generateAndSave)

test_GAN()

