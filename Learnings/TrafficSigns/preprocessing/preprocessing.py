"""Functions and classes for loading and augmenting an image dataset.

# Contents of this module

  - lazy_load_and_augment_batches() function
  - Augmentation interface
  - Augmentations, e.g. 'RandomZoom'
  - Predefined augmentation lists
  - show_generator_samples() function

# Load and augment data

    To lazily load batches of images and augment them on the fly, 
    use the function 'lazy_load_and_augment_batches' and provide 
    a list of Augmentation objects as an argument. The function 
    returns a generator of batches that can be directly inserted 
    into the 'fit' function of our model.

    ```python

    augmentations = [
        RandomZoom(),
        RandomBrightness(),
        GaussianNoise()
    ]

    batches_generator = lazy_load_and_augment_batches(
        train_directory,
        dataset_fraction = 0.5,
        augmentations_list = augmentations
    ) 

    model = build_and_compile_model()
    model.fit(batches_generator, epochs = 100)
    ```

    For instructions on how to use a validation split on the 
    reduced dataset, see the documentation of the 
    'lazy_load_and_augment_batches' function.

    Internally, an instance of [keras_preprocessing.image.ImageDataGenerator] 
    will be created using constructor arguments that are retrieved 
    from the specified Augmentation objects. Argument values 
    retrieved from augmentations that are placed later in the 
    provided list may override argument values retrieved from 
    earlier augmentations. Different callbacks for custom 
    augmentations will however be chained together, rather than 
    overridden. 
 
# Implement image augmentations
 
    To implement a new image augmentation, subclass the 
    Augmentation class and implement the 'get_dict' and/or the 
    'get_callback' method. 

    The 'get_dict' method, if implemented, shall return a 
    dictionary containing values for arguments of the constructor 
    of keras_preprocessing.image.ImageDataGenerator. The following 
    are the valid dictionary entries together with their default 
    values. 

    ```
    rotation_range = 0, 
    width_shift_range = 0, 
    height_shift_range = 0, 
    brightness_range = None, 
    shear_range = 0, 
    zoom_range = 0, 
    channel_shift_range = 0, 
    fill_mode = 'nearest', 
    cval = 0, 
    horizontal_flip = False, 
    vertical_flip = False, 
    ```
    
    The get_callback method, if implemented, shall return a callback 
    that takes a rescaled and augmented image and returns a further 
    augmented image. From the documentation of the 
    ImageDataGenerator class:
 
  > function that will be applied on each input.
  > The function will run after the image is resized and augmented.
  > The function should take one argument: one image (Numpy tensor 
  > with rank 3), and should output a Numpy tensor with the same 
  > shape.

# TODO

[ ] Occluding patches augmentation

"""

import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator


def lazy_load_and_augment_batches(train_directory,
                                  dataset_fraction=1.0,
                                  validation_split=0.0,
                                  subset=None,
                                  batch_size=32,
                                  target_size=(150, 150),
                                  color_mode='grayscale',
                                  augmentation_list=None):
    """
    Yields batches of resized and augmented images and corresponding 
    class encodings.

    # Arguments
        train_directory: String.
            Path to the training directory. The training directory 
            must contain one subdirectory for each data class.
        dataset_fraction: Float between 0.0 and 1.0 (default: 1.0).
            Use only a fraction of the dataset. All data classes 
            are affected equally. 
        validation_split: Float between 0.0 and 1.0 (default: 0.0).
            Fraction of the used part of the dataset that is 
            reserved for validation. If the parameter 'subset' is 
            None, the validation split will be ignored. Set the 
            'subset' parameter to either 'training' or 'validation' 
            to use the validation split.
        subset: String or None (default: None).
            To use a validation split, set the 'subset' parameter to 
            either 'training' or 'validation'. A validation split 
            reserves a fraction of the used part of the dataset for 
            validation. If the subset is 'training', the returned 
            batches will only contain images from the non-reserved 
            part of the used part of the dataset. If the subset is 
            'validation', only the reserved images will be used.
            Multiple calls of this function with the same dataset, 
            'dataset_fraction', 'validation_split', and 'subset' 
            values will be consistent, i.e. they will reserve the
            same images for validation. 
        batch_size: Int (default: 32).
        target_size: Tuple of Ints (height, width) (default: (150, 150)).
            Each input image will be resized to the specified size.
        color_mode: String (default: 'grayscale').
        augmentation_list: = List of Augmentation objects (default: None).
            A list of Augmentation objects that represent augmentations 
            that are applied to each input image. Each augmentation must 
            implement the Augmentation interface. Augmentation parameters 
            retrieved from augmentations that are placed later in the 
            provided list may override argument values retrieved from 
            earlier augmentations. Different callbacks for custom 
            augmentations will however be chained together, rather than 
            overridden.

    # Returns
        An iterator yielding tuples (x, y), where x is a NumPy array 
            containing a batch of augmented images with shape 
            (batch_size, *target_size, channels) and y is a NumPy 
            array of the corresponding labels. 

    # Use only fraction of dataset
        To use only a fraction of the dataset, set the 
        'dataset_fraction' parameter to a value between 0.0 and 1.0.
        All data classes will be affected equally. 

    # Use validation split
        The used portion of the dataset can be divided further into 
        a 'training' and a 'validation' subset by setting the 
        'validation_split' parameter to a value between 0.0 and 1.0 
        and the 'subset' parameter to either 'training' or 
        'validation'. See the usage examples below. 

    # Examples
        ```python

        augmentations = [
            RandomZoom(),
            RandomBrightness(),
            GaussianNoise()
        ]

        # Without validation split:

        batches_generator = lazy_load_and_augment_batches(
            train_directory,
            dataset_fraction = 0.5,
            augmentations_list = augmentations
        )

        model = build_and_compile_model()
        model.fit(batches_generator, epochs = 100)

        # With validation split:

        dataset_fraction = 0.3
        validation_split = 0.5 

        train_generator = lazy_load_and_augment_batches(
            train_directory,
            dataset_fraction = dataset_fraction,
            validation_split = validation_split,
            subset = 'training'
            augmentations_list = augmentations
        )

        validation_generator = lazy_load_and_augment_batches(
            train_directory,
            dataset_fraction = dataset_fraction,
            validation_split = validation_split,
            subset = 'validation'
            augmentations_list = augmentations
        )

        model = build_and_compile_model()
        model.fit(batches_generator, 
            epochs = 100, 
            validation_data = validation_generator,
        )

        ```
     
    """
    config = _augmentation_config(augmentation_list)
    (p_split, p_subset) = _pseudo_validation_params(
        dataset_fraction, validation_split, subset)
    config['validation_split'] = p_split
    # Default values, listed here for easy access
    additional_settings = { 
        'data_format' : 'channels_last', 
        'interpolation_order' : 1, 
        'dtype' : 'float32'}
    config.update(additional_settings)
    datagen = ImageDataGenerator(**config)
    return datagen.flow_from_directory(
        train_directory, 
        target_size = target_size, # Non-default.
        color_mode = color_mode, # Non-default.
        batch_size = batch_size, # Non-default.
        subset = p_subset, # Non-default.
        save_to_dir = None,
        save_prefix = '',
        save_format = 'png')

def lazy_load_test_batches(path, batch_size=32,
                                  target_size=(150, 150),
                                  color_mode='grayscale',
                                  augmentation_list=None):
    config = _augmentation_config(augmentation_list)

    additional_settings = {
        'data_format'        : 'channels_last',
        'interpolation_order': 1,
        'dtype'              : 'float32' }

    config.update(additional_settings)
    datagen = ImageDataGenerator(**config)
    return datagen.flow_from_directory(
            path,
            target_size=target_size,  # Non-default.
            color_mode=color_mode,  # Non-default.
            batch_size=batch_size,  # Non-default.
            save_to_dir=None,
            save_prefix='',
            save_format='ppm',
            shuffle=False)


def _augmentation_config(augmentation_list):
    if augmentation_list is None:
        return {}
    config = {}
    callbacks = []
    for aug in augmentation_list:
        aug_dict = aug.get_dict()
        if aug_dict is not None:
            _remove_invalid_items(aug_dict, aug)
            config.update(aug_dict)
        aug_callback = aug.get_callback()
        if aug_callback is not None:
            callbacks.append(aug_callback)
    if len(callbacks) != 0:
        config['preprocessing_function'] = _PreprocessingFunction(callbacks)
    return config


def _pseudo_validation_params(dataset_fraction, validation_split, subset):
    pseudo_validation_split = None
    pseudo_subset = None
    if subset is None:
        pseudo_validation_split = 1.0 - dataset_fraction
        pseudo_subset = 'training'
    elif subset == 'training':
        pseudo_validation_split = 1.0 - (dataset_fraction - dataset_fraction * validation_split)
        pseudo_subset = 'training'
    elif subset == 'validation':
        pseudo_validation_split = dataset_fraction * validation_split
        pseudo_subset = 'validation'
    return (pseudo_validation_split, pseudo_subset)


def _remove_invalid_items(augmentation_dict, source_object):
    for key in list(augmentation_dict.keys()):
        if not key in Augmentation.allowed_augmentation_keys:
            # Modifying the dict here is safe because we are 
            # iterating over a list with references to the 
            # original keys, rather than iterating over the 
            # dict directly
            del augmentation_dict[key]
            warning = " ".join([
                f"WARNING: Removing invalid key '{key}' from the dictionary",
                f"returned by '{type(source_object).__name__}.get_dict()'"])
            print(warning)


class _PreprocessingFunction():
    def __init__(self, callback_list) -> None:
        self.callback_list = callback_list

    def __call__(self, image):
        for callback in self.callback_list:
            image = callback(image)
        return image

class Augmentation:

    allowed_augmentation_keys = [
        'rotation_range',
        'width_shift_range',
        'height_shift_range',
        'brightness_range',
        'shear_range',
        'zoom_range',
        'channel_shift_range', # TODO: create corresponding augmentation or use in brightness augmentation
        'fill_mode',
        'cval',
        'horizontal_flip',
        'vertical_flip',
    ]

    def get_dict(self):
        pass

    def get_callback(self):
        pass 

class FillSpaces(Augmentation):

    allowed_modes = ["nearest", "reflect", "wrap", "black", "gray", "white"]

    def __init__(self, mode = "nearest") -> None:
        if mode in ["nearest", "reflect", "wrap"]:
            self.fill_mode = mode
        else:
            mode_valid = False
            for (color, cval) in zip(["black", "gray", "white"], [0, 128, 255]):
                if mode == color:
                    mode_valid = True
                    self.fill_mode = "constant"
                    self.cval = cval
                    break
            if not mode_valid:
                raise Exception(f"'mode' must be one of {FillSpaces.allowed_modes}")

    def get_dict(self):
        config = {'fill_mode' : self.fill_mode}
        if self.fill_mode == "constant":
            config['cval'] = self.cval
        return config

class RandomRotation(Augmentation):

    def __init__(self, max_degrees = 10) -> None:
        """
        # Arguments
            max_degrees: Int (default: 10)
                Sets the rotation angle range to
                [-max_degrees, max_degrees].
        """
        self.max_degrees = max_degrees

    def get_dict(self):
        return {'rotation_range' : self.max_degrees}

class RandomShiftHorizontal(Augmentation):

    def __init__(self, max_pixels = 10, max_width_frac = None) -> None:
        """Randomly shifts the image horizontally.

        The maximum shift value can either be specified as a number
        of pixels or as a fraction of the width of the image. By
        default a maximum shift value of 10 pixels is used.

        # Arguments
            max_pixels: Int (default: 10)
                Maximum number of pixels for random shifts. This parameter
                is ignored if the parameter 'max_width_frac' is set.
            max_width_frac: Float between 0.0 and 1.0 inclusive.
                Maximum shift distance, specified as a fraction of the
                width of the image.
        """
        if max_width_frac is not None:
            if max_width_frac <= 1.0:
                self.width_shift_range = max_width_frac
            else:
                raise Exception("'max_width_frac' must be between 0 and 1")
        else:
            self.width_shift_range = max_pixels

    def get_dict(self):
        return {'width_shift_range' : self.width_shift_range}

class RandomShiftVertical(Augmentation):

    def __init__(self, max_pixels = 10, max_height_frac = None) -> None:
        """Randomly shifts the image vertically.

        The maximum shift value can either be specified as a number
        of pixels or as a fraction of the height of the image. By
        default a maximum shift value of 10 pixels is used.

        # Arguments
            max_pixels: Int (default: 10)
                Maximum number of pixels for random shifts. This parameter
                is ignored if the parameter 'max_height_frac' is set.
            max_height_frac: Float between 0.0 and 1.0 inclusive.
                Maximum shift distance, specified as a fraction of the
                height of the image.
        """
        if max_height_frac is not None:
            if max_height_frac <= 1.0:
                self.height_shift_range = max_height_frac
            else:
                raise Exception("'max_height_frac' must be between 0 and 1")
        else:
            self.height_shift_range = max_pixels

    def get_dict(self):
        return {'height_shift_range' : self.height_shift_range}

class RandomBrightness(Augmentation):

    def __init__(self, brightness_range = (0.7, 1.3)) -> None:
        """
        # Arguments
            brightness_range: Tuple of two Floats (default: (0.7, 1.3))
                The range of floats from which a brightness factor 
                will be picked for augmentation. The image array will 
                be multiplied with the brightness factor and then 
                clipped to the valid range. A factor of 0.0 will result 
                in a black image and a factor of 1.0 will not change 
                the image. No restrictions apply for the factor.
        """
        self.brightness_range = brightness_range

    def get_dict(self):
        return {'brightness_range' : self.brightness_range}

class RandomShear(Augmentation):

    def __init__(self, shear_range = 10.0) -> None:
        """
        # Arguments
            shear_range: Float. Shear angle in degrees
        """
        self.shear_range = shear_range

    def get_dict(self):
        return {'shear_range' : self.shear_range}

class RandomZoom(Augmentation):

    def __init__(self, zoom_range = (0.8, 1.2)) -> None:
        """
        # Arguments
            zoom_range: Tuple of Floats (default: (0.8, 1.2))
                Range from which to pick a zoom factor. A factor
                of 1.0 does not change the image, smaller factors
                zoom out, larger factors zoom in.
        """
        self.zoom_range = zoom_range

    def get_dict(self):
        return {'zoom_range' : self.zoom_range}

class RandomFlipHorizontal(Augmentation):

    def __init__(self) -> None:
        super().__init__()

    def get_dict(self):
        return {'horizontal_flip' : True}

class RandomFlipVertical(Augmentation):

    def __init__(self) -> None:
        super().__init__()

    def get_dict(self):
        return {'vertical_flip' : True}

class RandomContrast(Augmentation):

    def __init__(self, contrast_range = (0.6, 1.4)) -> None:
        """Randomly changes the contrast of an image.  

        The original image must be an 8-bit RGB or grayscale image.

        # Arguments
            contrast_range: Tuple of two Floats (default: (0.6, 1.4))
                When augmenting an image, a contrast factor is randomly 
                chosen from the specified range. A factor of 1 does not 
                change the image, smaller factors decrease the contrast, 
                and larger factors increase the contrast. A factor of 0 
                results in a gray image.
        """
        self.range_start = contrast_range[0]
        self.range_stop = contrast_range[1]

    def augment(self, image):
        # TODO: new code should use a Generator instance for random number generation
        f = np.random.uniform(low = self.range_start, high = self.range_stop)
        return np.clip(
            (1 - f) * np.mean(image) + f * image, 
            a_min = 0, 
            a_max = 255
        )

    def get_callback(self):
        return self.augment

class GaussianNoise(Augmentation):

    def __init__(self, mean = 0.0, std_dev = 20) -> None:
        """Adds gaussian noise to images.

        The original image must be an 8-bit RGB or grayscale image.
        """
        self.mean = mean
        self.std_dev = std_dev

    def get_callback(self):
        # TODO: new code should use a Generator instance for random number generation
        return lambda image : np.clip(
            image + np.random.normal(loc = self.mean, scale = self.std_dev, size = image.shape), 
            a_min = 0, 
            a_max = 255
        )

class SaltPepperNoise(Augmentation):

    def __init__(self, strength = 0.1) -> None:
        """Adds salt and pepper noise to images.

        This augmentation assumes that the channel axis is the 
        last axis of the numpy array that represents the image 
        to be augmented. The original image must be an 8-bit 
        RGB or grayscale image.
        
        # Arguments
            strength: Float between 0 and 1 inclusive (default: 0.1)
                The probability for manipulating a pixel. If a 
                pixel is manipulated it's color will be set to 
                white or black with equal probability.
        """
        self.p_salt = strength / 2
        self.p_none = 1 - 2 * self.p_salt

    def augment(self, image):
        noise_shape = image.shape[:-1]
        # TODO: new code should use a Generator instance for random number generation
        noise = np.random.choice([0, -255, 255], p = [self.p_none, self.p_salt, self.p_salt], size = noise_shape)
        rolled_image = np.moveaxis(image, -1, 0)
        rolled_result = np.clip(rolled_image + noise, a_min = 0, a_max = 255)
        return np.moveaxis(rolled_result, 0, -1)

    def get_callback(self):
        return self.augment

augmentations_basic = [
    FillSpaces("nearest"),
    RandomRotation(),
    RandomShiftHorizontal(),
    RandomShiftVertical(),
    RandomBrightness(),
    RandomShear(),
    RandomZoom(),
    RandomContrast()]

augmentations_basic_noise = [
    FillSpaces("nearest"),
    RandomRotation(),
    RandomShiftHorizontal(),
    RandomShiftVertical(),
    RandomBrightness(),
    RandomShear(),
    RandomZoom(),
    RandomContrast(),
    GaussianNoise(),
    SaltPepperNoise()]

augmentations_basic_noise_flip = [
    FillSpaces("nearest"),
    RandomRotation(),
    RandomShiftHorizontal(),
    RandomShiftVertical(),
    RandomBrightness(),
    RandomShear(),
    RandomZoom(),
    RandomFlipHorizontal(),
    RandomContrast(),
    GaussianNoise(),
    SaltPepperNoise()]

def show_generator_samples(data_generator):
    """Visualizes images from the given batches generator.
    
    This function gets the next batch from the given generator 
    and plots the first 12 images from the batch. It assumes 
    that the channel axis is the last axis of an image array.

    # Arguments
        data_generator: Iterator returning batches
            An Iterator that returns batches. Each batch must 
            be a tuple containing an array of images as it's 
            first element. The tuple may also contain other 
            data, e.g. an array of corresponding class labels  
    """
    batch = next(data_generator)
    images = batch[0]
    if len(images) > 12:
        images = images[:12]
    for (index, image) in enumerate(images):
        plt.subplot(3, 4, index + 1)
        # We turn the axis off for all images except the first. 
        # We include the axis of the first image because they 
        # show the size of the images, which could be of interest
        if index == 0:
            height = image.shape[0]
            width = image.shape[1]
            plt.yticks([0, height - 1], [0, height])
            plt.xticks([0, width - 1], [0, width])
            plt.gca().xaxis.tick_top()
        else:
            plt.axis("off")
        # If the image is grayscale, i.e. has only one channel, 
        # we need to manually set the color map to grayscale.
        # If the image is RGB, we need to cast the values to ints 
        # because for RGB data plt.imshow ignores the vmin and vmax 
        # parameters and determines the value range based on the 
        # data type. The allowed value range is [0.0, 1.0] for 
        # floats and [0, 255] for ints
        if image.shape[-1] == 1:
            plt.imshow(image, vmin = 0, vmax = 255)
            plt.set_cmap("gray")
        else:    
            plt.imshow(image.astype(int))
    plt.show()
