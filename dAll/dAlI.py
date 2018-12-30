import sys, os
from PIL import Image

sys.path.append('C:/Users/Samuel Troper/Desktop/QuickTMI/QuickTMI')

import combo

def JPGExtension(fileName):
    if ".jpg" in fileName.lower():
        return True
    return False

def normalizeInput(rawImage, largestWidth, largestHeight):
    normalizedImage = []
    for i in range(len(rawImage)):
        if i is 0:
            normalizedImage.append(rawImage[i]/largestWidth)
        elif i is 1:
            normalizedImage.append(rawImage[i]/largestHeight)
        else:
            normalizedImage.append(rawImage[i]/255)
    return normalizedImage

def prepInput(inputDirectory="C:/Users/Samuel Troper/Desktop/dAll/R_Images",
              newWidth = 100, newHeight=100):
    #Currently will only import JPG
    directoryJPGList = list(filter(JPGExtension, os.listdir(inputDirectory)))
    images = []
    largestWidth = 0
    largestHeight = 0
    iteration = 0
    for image in directoryJPGList:
        if iteration > 10: #REMOVE 4 REAL
            break
        print(iteration)
        iteration += 1
        currentImage = Image.open(inputDirectory + "/" + image)
        print(currentImage.size)
        width = currentImage.size[0]
        print(width)
        height = currentImage.size[1]
        print(height)
        if width > largestWidth:
            largestWidth = width
        if height > largestHeight:
            largestHeight = height
        currentImageInfo = [width, height]
        currentImage = currentImage.resize((newWidth,newHeight))
        for pixel in list(currentImage.getdata()):
            currentImageInfo.extend(pixel)
        images.append(currentImageInfo)
    print(largestWidth)
    print(largestHeight)
    dkk = [largestWidth]*len(images)
    print(dkk)
    dkj = [largestHeight]*len(images)
    print(dkj)
    normalizedImages = list(map(normalizeInput, images,
                                dkk, dkj))
    return normalizedImages
    #print("Ok")
    #for i in range(10):
        #displayImage(normalizedImages[i],
                     #largestWidth, largestHeight,
                     #newWidth, newHeight)

def displayImage(image, largestWidth, largestHeight, newWidth, newHeight):
    temp = Image.new('RGB', (newWidth, newHeight))
    temp.putdata(linearToRGB(image[2:]))
    temp = temp.resize((int(largestWidth*image[0]), int(largestHeight*image[1])))
    temp.show()

def displayImage2(image, largestWidth, largestHeight):
    display = Image.new('RGB', (int(largestWidth*image[0]),
                                int(largestHeight*image[1])))
    display.putdata(linearToRGB(image[2:]))
    display.show()

def linearToRGB(image):
    newImage = []
    currentPixel = []
    for i in range(len(image)):
        currentPixel.append(int(image[i]*255))
        if i%3 is 2:
            newImage.append(tuple(currentPixel))
            currentPixel = []
    return tuple(newImage)

def generateAndSave(GAN, epoch, number=3):
    generator = GAN[0]
    noise = np.random.normal(0, 1, size=[number, GAN[3]])
    generated_image = generator.predict(noise)
    
    

def test_GAN(newWidth=100, newHeight=100):
    models = buildGANModel(newWidth*newHeight, [256, 512, 1024, 2048], [1024, 512, 256], epochs=100)
    data = prepInput(newWidth = newWidth, newHeight=newHeight)
    trainGAN(models, data, epochs=10, displayFunction=generateAndSave)

test_GAN()

