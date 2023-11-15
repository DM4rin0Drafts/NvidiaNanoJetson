import matplotlib.pyplot as plt
from Convolutional.networks import build_convolutional_model
from tensorflow.keras import layers

from .preprocessing import (FillSpaces, GaussianNoise, RandomBrightness,
                            RandomFlipHorizontal, RandomFlipVertical,
                            RandomRotation, RandomShear, RandomShiftHorizontal,
                            RandomShiftVertical, RandomZoom, SaltPepperNoise,
                            lazy_load_and_augment_batches, show_generator_samples)

train_params = {
    "batch_size": 32,
    "epochs": 2,
    "target_size": (150, 150),
    "input_shape": (150, 150, 1),
    "num_classes": 43,
    "loss": 'categorical_crossentropy',
    "optimizer": 'adam',
    "metrics": ['accuracy'],
}

augmentations = [
    FillSpaces("gray"),
    RandomRotation(),
    RandomShiftHorizontal(),
    RandomShiftVertical(),
    RandomBrightness(),
    RandomShear(),
    RandomZoom(),
    RandomFlipHorizontal(),
    RandomFlipVertical(),
    GaussianNoise(),
    SaltPepperNoise(),
]

batches_generator_1 = lazy_load_and_augment_batches(
    'data/GTSRB/Train/', 
    dataset_fraction = 0.02,
    augmentation_list = None
)

show_generator_samples(batches_generator_1)

batches_generator_2 = lazy_load_and_augment_batches(
    'data/GTSRB/Train/', 
    dataset_fraction = 0.02,
    batch_size = train_params["batch_size"],
    target_size = train_params["target_size"],
    augmentation_list = augmentations
)

show_generator_samples(batches_generator_2)

layer_list_simple = [
    layers.Conv2D(8, kernel_size=4, padding="same", activation="relu"),
    layers.MaxPooling2D(pool_size=5, strides=1),
    layers.Conv2D(16, kernel_size=4, padding="same", activation="relu"),
    layers.MaxPooling2D(pool_size=5, strides=1),
    layers.Flatten(),
    layers.Dropout(0.2)
]

model = build_convolutional_model(layer_list_simple, train_params)
model.summary()

history = model.fit(batches_generator_2, epochs = train_params["epochs"])

plt.plot(history.history["accuracy"])
plt.title("Simple model - Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.show()
