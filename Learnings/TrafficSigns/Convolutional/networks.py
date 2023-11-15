import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras import backend as K

# Data
x_train = np.empty(shape=(10,))
y_train = np.empty(shape=(10,))
x_test = np.empty(shape=(10,))
y_test = np.empty(shape=(10,))


highest_acc = 0.0
smallest_loss = np.inf
# LeNet5
def build_lenet5(params):
    lenet5 = Sequential([])

    lenet5.add(layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding="same",
                             input_shape=(224, 224, 3)))
    lenet5.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    lenet5.add(layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
    lenet5.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    lenet5.add(layers.Flatten())
    lenet5.add(layers.Dense(120, activation='tanh'))
    lenet5.add(layers.Dense(84, activation='tanh'))
    lenet5.add(layers.Dense(30, activation='softmax'))

    lenet5.compile(optimizer=params['optimizer'], loss=params['loss'], metrics=params['metrics'])

    return lenet5


# In LeNet sind die Eingabegrößen anders als von netz erwartet, filter dimensions passen nicht. Wollen wir das trotzdem nutzen?

# Parameter of the Convolutional Network
parameter = {'num_classes': 1,
             'input_shape': (150, 150, 1),
             'loss': 'categorial_crossentropy',
             'optimizer': 'adam',
             'metrics': 'accuracy',
             'batch_size': 10,
             'epochs': 100,
             'validation_split': 0,
             'learning_rate': 0.001
             }


def build_convolutional_model(layer_list, params=parameter):
    # Architecture
    model = Sequential([])

    # Add Input Layer
    model.add(keras.Input(shape=params['input_shape']))

    # Add Hidden Layer
    for layer in layer_list:
        model.add(layer)

    # Add Output Layer
    model.add(layers.Dense(params['num_classes'], activation="softmax"))

    # Compile model
    model.compile(loss=params['loss'], optimizer=params['optimizer'], metrics=params['metrics'])

    # Set the custom learning rate
    K.set_value(model.optimizer.learning_rate, params['learning_rate'])

    return model


def train_model(train_generator, validation_generator, model, params):
    # Train model without generator
    model.fit(train_generator, validation_data=validation_generator, epochs=params['epochs'])

    print("finished with learning")


def train_model_with_generator(train_generator, test_generator, model, params, save_model_path, save_evaluation_path):

    for i in range(params['epochs']):
        print("Train: ")
        model.fit(train_generator, batch_size=params['batch_size'], epochs=1)
        model_evaluation(test_generator, model, params, i, save_model_path, save_evaluation_path)
    print("finished with learning")

    return model


def model_evaluation(test_generator, model, params, i, save_model_path, acc_path):
    print("Test: ")
    y_pred = model.predict(test_generator, batch_size=params['batch_size'])
    s = model.evaluate(test_generator, batch_size=params['batch_size'])
    scores = np.zeros((1, 3))
    scores[0, 0] = i
    scores[0, 1] = s[0]
    scores[0, 2] = s[1]
    y_pred = np.argmax(y_pred, axis=1)

    global highest_acc, smallest_loss
    print(highest_acc, s[1])
    if s[1] >= highest_acc and s[0] <= smallest_loss:
        highest_acc = s[1]
        smallest_loss = s[0]
        save_model(model, save_model_path)

        fileDir = os.path.dirname(os.path.realpath('__file__'))
        filename = os.path.join(fileDir, acc_path)
        with open(filename, 'a') as f:
            f.write('Test Accuracy: {}\n'.format(s[1]))


def evaluate(model_path, generator):
    model = load_model(model_path)

    y_pred = model.predict(generator)
    y_pred = np.argmax(y_pred, axis=1)
    print("Prediction: ", y_pred)


# Save trained model
def save_model(model, path):
    model.save(path)


# Load trained model
def load_model(path):
    model = keras.models.load_model(path)
    return model
