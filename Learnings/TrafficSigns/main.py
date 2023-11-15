import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

from preprocessing.preprocessing import lazy_load_and_augment_batches, lazy_load_test_batches, augmentations_basic_noise
from Convolutional.networks import  train_model_with_generator, build_convolutional_model, evaluate
from DeepLearning.learning import hyperparameter_search
from utils import print_devices
import numpy as np
from tensorflow.keras import layers
import os

np.random.seed(1)
tf.random.set_seed(2)

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# Hidden Layer of the Convolutional Network
layer_list = [layers.Conv2D(256, kernel_size=(5, 5), padding="same", activation="relu"),
                  layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                  layers.Conv2D(512, kernel_size=(5, 5), padding="same", activation="relu"),
                  layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                  layers.Flatten(),
                  layers.Dropout(0.3)]

params = {"batch_size": 16,
              "epochs": 35,
              "validation_split": 0.0,
              "input_shape": (150, 150, 1),
              "num_classes": 5,
              "loss": 'categorical_crossentropy',
              "optimizer": 'adam',
              "metrics": 'accuracy',
              "learning_rate": 0.00005
              }

def main():
    path_to_train = r"./Data/Train"
    path_to_test = r"./Data/Test"


    train_generator = lazy_load_and_augment_batches(path_to_train,
                                                    batch_size=params["batch_size"],
                                                    subset='training',
                                                    validation_split=params["validation_split"])
    validation_generator = lazy_load_and_augment_batches(path_to_test,
                                                         batch_size=params["batch_size"])

    if mode == "hyper":
        search_ranges = {"batch_size": [16],
                         "epochs": [5],
                         "learning_rate": [0.00005]
                         }
        hyperparameter_list = hyperparameter_search(search_ranges, params, layer_list, path_to_train, path_to_test)

    if mode == "fraction_experiment":
        frac_gen = lazy_load_and_augment_batches(
            path_to_train,
            dataset_fraction=1.0,
            validation_split=params["validation_split"],
            batch_size=params["batch_size"],
            subset='training',
            augmentation_list=None,
        )

        learned_models = []
        for frac, epochs in zip([0.7], [42]):
            print("frac: {}, epochs: {}".format(frac, epochs))
            frac_gen = lazy_load_and_augment_batches(
                path_to_train,
                dataset_fraction=frac,
                validation_split=params["validation_split"],
                batch_size=params["batch_size"],
                subset='training',
                augmentation_list=augmentations_basic_noise,
            )
            test_gen = lazy_load_test_batches(path_to_test, batch_size=params["batch_size"])
            frac_model = build_convolutional_model(layer_list=layer_list, params=params)
            params['epochs'] = epochs
            params["experiment_title"] = "data_percent_{}".format(str(frac * 100))
            history = train_model_with_generator(frac_gen, test_gen, frac_model, params, "saved_models/model" + str(frac), "saved_models/model_val" + str(frac) + ".txt")
            learned_models.append(history)


    if mode == 'evaluate':
        for i in range(10):
            path = r"Data/SearchTrafficSigns/TrainIJCNN2013"
            model_path = "saved_models/model"
            generator = lazy_load_test_batches(path)
            evaluate(model_path, generator)



from utils import separate_data
def train_localization():
    separate_data("Data/SearchTrafficSigns/Images")




if __name__ == '__main__':
    #mode = "evaluate"
    #print("Running in {} mode".format(mode))
    #print_devices()
    #main()
    train_localization()
