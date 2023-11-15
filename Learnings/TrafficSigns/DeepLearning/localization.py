import numpy as np
from PIL import Image
import cv2
from keras.applications.mobilenetv2 import preprocess_input
from Convolutional.networks import load_model


def load_image(path):
	pass

def sliding_window(image):
	pass

def extract_image(image, x_cur, y_cur, size=(75, 75)):
	pass


def precision(TP, FP):
	return TP / TP + FP


def recall(TP, FN):
	return TP / TP + FN


def loss():
	pass  # maybe softmax


def predict_input(full_image, model, image_size=(100, 100, 3)):
	image_height, image_width, _ = full_image.shape

	image = cv2.resize(full_image, image_size)
	feat_scaled = preprocess_input(np.array(image, dtype=np.float32))

	region = model.predict(x=np.array([feat_scaled]))[0]

	x0 = int(region[0] * image_width / image_size[0])
	y0 = int(region[1] * image_height / image_size[1])

	x1 = int((region[0] + region[2]) * image_width / image_size[0])
	y1 = int((region[1] + region[3]) * image_height / image_size[1])

	cv2.rectangle(full_image, (x0, y0), (x1, y1), (0, 0, 255), 1)
	cv2.imshow("image", full_image)

import time
import matplotlib.pyplot as plt
from PIL import Image

def show_image(img):
	plt.imshow(img)
	plt.show()

if __name__ == "__main__":
	path = r"../Data/SearchTrafficSigns/Images/00011.ppm"
	model_path = "../saved_models/model"

	# model = load_model(model_path)
	img = np.asarray(Image.open(path))
	img_size = img.shape
	image = cv2.resize(img, img_size[:2])
	feat_scaled = preprocess_input(np.array(image, dtype=np.float32))
	predict_input(img, model, img_size)
