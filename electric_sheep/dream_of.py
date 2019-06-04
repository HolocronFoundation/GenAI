
import sys
import os
import numpy as np
from PIL import Image
sys.path.append('C:/Users/Samuel Troper/Documents/GitHub/QuickTMI/QuickTMI') # TODO: Change this path
import combo

output_dir ='D:/dAlI Output' # TODO: Change this

largest_width = 0
largestHeight = 0

def check_extension(file_name, extension):
    if extension in file_name.lower():
        return True
    return False

def normalize_input(raw_image):
    normalized_image = []
    for i, value in enumerate(raw_image):
        if i == 0:
            normalized_image.append((value/largest_width)*2-1)
        elif i == 1:
            normalized_image.append((value/largestHeight)*2-1)
        else:
            normalized_image.append((value/255)*2-1)
    return normalized_image

def prep_input(new_width, new_height, input_directory):
    directory_list = list(filter(lambda x: check_extension(x, ".jpg"), input_directory))
    directory_list.extend(list(filter(lambda x: check_extension(x, ".png"), input_directory)))
    images = []
    iteration = 0
    global largest_width
    global largestHeight
    for image in directory_list:
        #if iteration > 100: #REMOVE 4 REAL
            #break
        print(iteration)
        print(image)
        iteration += 1
        current_image = Image.open(input_directory + "/" + image)
        width = current_image.size[0]
        height = current_image.size[1]
        if width > largest_width:
            largest_width = width
        if height > largestHeight:
            largestHeight = height
        current_image_info = [width, height]
        current_image = current_image.resize((new_width, new_height))
        for pixel in list(current_image.convert('RGB').getdata()):
            current_image_info.extend(pixel)
        images.append(current_image_info)
    normalized_images = list(map(normalize_input, images))
    #print("Ok")
    #for i in range(10):
    #    display_image(normalized_images[i], new_width, new_height)
    return normalized_images

def display_image(image, new_width, new_height, epoch, iteration):
    temp = Image.new('RGB', (new_width, new_height))
    temp.putdata(linearToRGB(image[2:]))
    adjusted_width = int(largest_width*(image[0]+1)/2)
    adjusted_height = int(largestHeight*(image[1]+1)/2)
    if adjusted_width > 0 and adjusted_height > 0:
        filename = output_dir +'/epoch' + str(epoch) + '/' + str(iteration) + '.png'
        temp = temp.resize((adjusted_width, adjusted_height))
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        #temp.show()
        temp.save(filename)

def linearToRGB(image):
    new_image = []
    current_pixel = []
    for i, value in enumerate(image):
        current_pixel.append(int((value+1)*255/2))
        if i%3 == 2:
            new_image.append(tuple(current_pixel))
            current_pixel = []
    return tuple(new_image)

def generateAndSave(GAN, epoch, imageDimension=(100,100), number=5, save=True, saveInterval=10000): # TODO: Creat a save in place option TODO: Cleanup some of these defaults
    generator = GAN[0]
    if epoch % saveInterval == 0 and save and epoch != 0:
        filename = output_dir +'/epoch' + str(epoch) + '/GAN[0].h5'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        GAN[0].save(filename)
        filename = output_dir +'/epoch' + str(epoch) + '/GAN[1].h5'
        GAN[1].save(filename)
        filename = output_dir +'/epoch' + str(epoch) + '/GAN[2].h5'
        GAN[2].save(filename)
    for i in range(number):
        noise = np.random.normal(0, 1, size=[1, GAN[3]])
        generated_image = generator.predict(noise)[0]
        display_image(generated_image, imageDimension[0], imageDimension[1], epoch, i)


def test_GAN(output_image={"width":100,"height":100}): # TODO: Add BW option
    models = combo.buildGANModel(output_image["width"]*output_image["height"]*3+2, [128, 256, 512, 1024, 2048, 4096], [256, 256, 128, 16, 8, 4, 2])
    data = prep_input(output_image["width"], output_image["height"], "C:/Users/Samuel Troper/Pictures/Im-DB") # TODO: Change this directory
    data = np.array(data)
    combo.trainGAN(models, data, epochs=100000, batchSize=128, displayFunction=generateAndSave)

test_GAN()

