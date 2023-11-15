from .preprocessing import (FillSpaces, GaussianNoise, RandomBrightness, RandomContrast,
                            RandomFlipHorizontal, RandomFlipVertical,
                            RandomRotation, RandomShear, RandomShiftHorizontal,
                            RandomShiftVertical, RandomZoom, SaltPepperNoise,
                            augmentations_basic, augmentations_basic_noise, augmentations_basic_noise_flip,
                            lazy_load_and_augment_batches, show_generator_samples)

augmentations = [
    FillSpaces("black"),
    RandomRotation(),
    RandomShiftHorizontal(),
    RandomShiftVertical(),
    RandomBrightness(),
    RandomShear(),
    RandomZoom(),
    #RandomFlipHorizontal(),
    #RandomFlipVertical(),
    RandomContrast(),
    GaussianNoise(),
    SaltPepperNoise(),
]

batches_generator = lazy_load_and_augment_batches(
    'data/GTSRB/Train/', 
    dataset_fraction = 0.02,
    target_size = (150, 150),
    color_mode='rgb',
    augmentation_list = augmentations_basic_noise
)

show_generator_samples(batches_generator)