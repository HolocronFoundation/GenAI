
import sys
import os
import json
import atexit
import numpy as np
from PIL import Image
from tensorflow.keras import models
sys.path.append('/home/Projects/QuickTMI/QuickTMI')
import combo

# TODO: Add more model summary statistics when saved
# TODO: Create more optionality - allow swapping in various generators and discriminators in order to create a more flexible network

IMAGE = {"width":100, "height":100}
WORKING_ROOT = "/media/troper/Troper_Work-DB/dreams_of/"
INPUT_DIR = "electric_sheep/"
OUTPUT_DIR = "output/"
EPOCHS = 100000
BATCH_SIZE = 16
MODEL_NAMES = ["generator", "discriminator", "gan"]
GENERATOR_LAYERS = [128, 256, 512, 1024, 2048, 4096]
DISCRIMINATOR_LAYERS = [128, 16, 8, 4, 2]
SAVE = {
    "on": True,
    "interval": 100,
    "in_place": True,
    "directory": WORKING_ROOT + OUTPUT_DIR
}
TEST_IMAGE_COUNT = 10
LARGEST_HEIGHT = 0
LARGEST_WIDTH = 0

def check_extension(file_name, extension):
    if extension in file_name.lower():
        return True
    return False

def normalize_input(raw_image):
    normalized_image = []
    for i, value in enumerate(raw_image):
        if i == 0:
            normalized_image.append((value/LARGEST_WIDTH)*2-1)
        elif i == 1:
            normalized_image.append((value/LARGEST_HEIGHT)*2-1)
        else:
            normalized_image.append((value/255)*2-1)
    return normalized_image

def prep_input(new_width, new_height, input_directory):
    iters = 24
    directory_list = list(filter(lambda x: check_extension(x, ".jpg"), os.listdir(input_directory)))
    directory_list.extend(list(filter(lambda x: check_extension(x, ".png"), os.listdir(input_directory))))
    images = np.empty((int(len(directory_list)/iters)+1, new_width*new_height*3+2), np.float64)
    global LARGEST_WIDTH
    global LARGEST_HEIGHT
    np_index = 0
    for i, image in enumerate(directory_list):
        if i%2400 == 0:
            print("Processing image " + str(i) + " of " + str(len(directory_list)))
        if i%iters == 0:# TODO: Right now this loads 4.1% of the data. Ideally, we'd like to shuffle and shard the data and use all of it. This should be based upon system RAM size. We probably want to limit it to 1/2 of RAM to allow plenty of extra space.
            current_image = Image.open(input_directory + "/" + image)
            width = current_image.size[0]
            height = current_image.size[1]
            if width > LARGEST_WIDTH:
                LARGEST_WIDTH = width
            if height > LARGEST_HEIGHT:
                LARGEST_HEIGHT = height
            current_image_info = [width, height]
            current_image = current_image.resize((new_width, new_height))
            for pixel in list(current_image.convert('RGB').getdata()):
                current_image_info.extend(pixel)
            images[np_index] = normalize_input(current_image_info)
            np_index += 1
    return images

def display_image(image, epoch, iteration, output_dir):
    temp = Image.new('RGB', (image["width"], image["height"]))
    temp.putdata(linearToRGB(image["generated"][2:]))
    adjusted_width = int(LARGEST_WIDTH*(image["generated"][0]+1)/2)
    adjusted_height = int(LARGEST_HEIGHT*(image["generated"][1]+1)/2)
    if adjusted_width > 0 and adjusted_height > 0:
        filename = output_dir + 'epoch' + str(epoch) + '/' + str(iteration) + '.png'
        temp = temp.resize((adjusted_width, adjusted_height))
        os.makedirs(os.path.dirname(filename), exist_ok=True)
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

def generate_and_save(gan, image, test_image_count=TEST_IMAGE_COUNT, save_info=None):
    if save_info is None:
        save_info = SAVE
    save(gan, save_info, False)
    for i in range(test_image_count):
        noise = np.random.normal(0, 1, size=[1, gan["seed_size"]])
        image["generated"] = gan["networks"]["generator"].predict(noise)[0]
        display_image(image, gan["epoch_current"], i, save_info["directory"])

def save(gan, save_info, is_exit):
    if save_info["on"] and gan["epoch_current"] != 0 and (gan["epoch_current"] % save_info["interval"] == 0 or is_exit):
        in_place_mod = ""
        if not save_info["in_place"]:
            in_place_mod = 'epoch' + str(gan["epoch_current"]) + '/'
        for key in gan["networks"]:
            filename = save_info["directory"] + in_place_mod + key + '.h5'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            gan["networks"][key].save(filename)
        reduced_dict = {}
        for key in gan:
            if key != "networks":
                reduced_dict[key] = gan[key]
        other_file_json = json.dumps(reduced_dict)
        filename = save_info["directory"] + in_place_mod + 'other.tmi'
        with open(filename, "w") as other_file:
            other_file.write(other_file_json)

def test_gan(image=None, load=None): # TODO: Add BW optionAdd
    if image is None:
        image = IMAGE
    gan = {}
    if load is None:
        gan = combo.build_gan_model(image["width"]*image["height"]*3+2, {"generator": GENERATOR_LAYERS, "discriminator": DISCRIMINATOR_LAYERS})
    else:
        other_info = None
        with open(load["directory"] + "other.tmi") as other_info_file:
            other_info = json.load(other_info_file)
        gan["networks"] = {}
        for network in load["names"]:
            gan["networks"][network] = models.load_model(load["directory"] + network + ".h5")
        for key in other_info:
            gan[key] = other_info[key]
    images = prep_input(image["width"], image["height"], WORKING_ROOT + INPUT_DIR)
    # TODO: This could be a good spot to implement sharding - train a few epochs with each set, then swap out for more sets
    try:
        combo.train_gan(gan, images, image, epoch_total=EPOCHS, batch_size=BATCH_SIZE, display_function=generate_and_save)
    except (KeyboardInterrupt, SystemExit):
        save(gan, SAVE, True)
        raise

def load_and_restore(model_names=None, load_dir=WORKING_ROOT + OUTPUT_DIR, image=None):
    if model_names is None:
        model_names = MODEL_NAMES
    if image is None:
        image = IMAGE
    load = {
        "directory": load_dir,
        "names": model_names
    }
    test_gan(image, load=load)

if __name__ == "__main__":
    #test_gan()
    load_and_restore()
