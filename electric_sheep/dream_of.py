
import sys
import os
import numpy as np
from PIL import Image
sys.path.append('/main-d/Projects/QuickTMI/QuickTMI')
import combo

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
    directory_list = list(filter(lambda x: check_extension(x, ".jpg"), os.listdir(input_directory)))
    directory_list.extend(list(filter(lambda x: check_extension(x, ".png"), os.listdir(input_directory))))
    images = []
    global largest_width
    global largestHeight
    for i,image in enumerate(directory_list):
        #if iteration > 100: #REMOVE 4 REAL
            #break
        if i%2400 == 0:
            print("Processing image " + str(i) + " of " + str(len(directory_list)))
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

def display_image(image, epoch, iteration, output_dir):
    temp = Image.new('RGB', (image["width"], image["height"]))
    temp.putdata(linearToRGB(image["generated"][2:]))
    adjusted_width = int(largest_width*(image["generated"][0]+1)/2)
    adjusted_height = int(largestHeight*(image["generated"][1]+1)/2)
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

def generate_and_save(GAN, epoch, image, number=5, save=None):
    if save is None:
        save = {
            "on": True,
            "interval": 10000,
            "in_place": True,
            "directory": '/media/troper/Troper_Work-DB/dreams_of/output'
        }
    generator = GAN[0]
    if epoch % save["interval"] == 0 and save["on"] and epoch != 0:
        if save["in_place"]:
            filename = save["directory"] + '/GAN[0].h5'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            GAN[0].save(filename)
            filename = save["directory"] + '/GAN[1].h5'
            GAN[1].save(filename)
            filename = save["directory"] + '/GAN[2].h5'
            GAN[2].save(filename)
        else:
            filename = save["directory"] +'/epoch' + str(epoch) + '/GAN[0].h5'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            GAN[0].save(filename)
            filename = save["directory"] +'/epoch' + str(epoch) + '/GAN[1].h5'
            GAN[1].save(filename)
            filename = save["directory"] +'/epoch' + str(epoch) + '/GAN[2].h5'
            GAN[2].save(filename)
    for i in range(number):
        noise = np.random.normal(0, 1, size=[1, GAN[3]])
        image["generated"] = generator.predict(noise)[0]
        display_image(image, epoch, i, save["directory"])


def test_GAN(image={"width":100,"height":100}): # TODO: Add BW option
    models = combo.build_gan_model(image["width"]*image["height"]*3+2, {"generator":[128, 256, 512, 1024, 2048, 4096], "discriminator":[256, 256, 128, 16, 8, 4, 2]})
    data = prep_input(image["width"], image["height"], "/media/troper/Troper_Work-DB/dreams_of/electric_sheep")
    data = np.array(data)
    combo.train_gan(models, data, image, epochs=100000, batch_size=128, display_function=generate_and_save)

test_GAN()

