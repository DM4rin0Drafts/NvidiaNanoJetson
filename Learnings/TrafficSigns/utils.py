import os
from tensorflow.python.client import device_lib
import tensorflow as tf

from Convolutional.networks import build_lenet5, train_model_with_generator, build_convolutional_model
from preprocessing.preprocessing import lazy_load_and_augment_batches, lazy_load_test_batches
import os
import platform


def set_cpu():
	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def print_devices():
	print("Available devices: ", device_lib.list_local_devices())


def fix_gpu():
	config = tf.compat.v1.ConfigProto()
	config.gpu_options.allow_growth = True
	session = tf.compat.v1.InteractiveSession(config=config)


def limiting_gpu():
	gpu = tf.config.experimental.list_logical_devices('GPU')
	tf.config.experimental.set_memory_growth(gpu[0], True)


def hyperparameter_search(search_ranges:dict, params:dict, layer_list:list, path_to_train:str, path_to_test=None) -> list:
    result_list, final_losses, final_accuracies = [], [], []
    best_loss, best_accuracy = 9999999, 0
    best_run_loss, best_run_acc = None, None
    for batch_size in search_ranges["batch_size"]:
        params["batch_size"] = batch_size
        for epochs in search_ranges["epochs"]:
            params["epochs"] = epochs
            for learning_rate in search_ranges["learning_rate"]:
                for frac in [1.0, 0.9, 0.8, 0.7, 0.5, 0.3]:
                    params["learning_rate"] = learning_rate
                    model_own = build_convolutional_model(layer_list=layer_list, params=params)
                    print(f"Training model for config {params}")
                    train_generator = lazy_load_and_augment_batches(path_to_train,
                                                                    batch_size=params["batch_size"],
                                                                    subset='training',
                                                                    dataset_fraction=frac,
                                                                    validation_split=params["validation_split"])
                    validation_generator = lazy_load_test_batches(path_to_test, batch_size=params["batch_size"])

                    history = train_model_with_generator(train_generator, validation_generator, model_own, params, "saved_models/model" + str(frac), "saved_models/model_acc" + str(frac) + ".txt")
                    print("finished")


    print(final_losses)
    print(f"Best loss: {best_loss}")
    print(f"Best run (loss): {best_run_loss}")
    print(final_accuracies)
    print(f"Best accuracy: {best_run_acc}")
    print(f"Best run (acc): {best_accuracy}")
    return result_list


def get_operating_system():
    return platform.system()
        

def up_directory(file):
    return os.path.dirname(file)


def get_path(path):  # TODO: add to search path
    operating_system = get_operating_system()
    if operating_system == "Windows":
        dir = up_directory(up_directory(os.path.realpath(__file__)))
        #path = dir + "\\Data\\GTSRB\\Test\\" + path
        path = dir + "\\Data\\Test_images_chopped\\" + path
        return path
    else:
        dir = up_directory(os.path.abspath(__file__))
        return os.path.join(dir, '..', path)

import  numpy as np
import cv2
from datetime import datetime
from PIL import Image
def sliding_window(image, size, step):
    shape = image.shape
    n, m = shape[0], shape[1]

    for i in range(0, n, step):
        for j in range(0, m, step):
            if i + size[0] < n and j + size[1] < m:
                new_image = Image.fromarray(image[i:i + size[0], j:j + size[1], :])

                dateTimeObj = datetime.now()
                filename = dateTimeObj.strftime("%d-%b-%Y_%H-%M-%S-%f") + ".jpeg"

                new_image.save("Data/SearchTrafficSigns/SeparateImages/" + filename)

    a = 0


def separate_data(path, width=100, height=100, pixel_step=15):
    files = os.listdir(path)
    for f in files:
        file_path = path + "/" + f
        picture = cv2.imread(file_path)
        sliding_window(picture, (height, width), pixel_step)



class_names = {
    0: '30 kmh',
    1: 'stop',
    2: 'keine_geschwindigkeitsbegrenzung',
    3: 'naechste_links',
    4: 'naechste_rechts',
}