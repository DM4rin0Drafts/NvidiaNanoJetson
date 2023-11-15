from collections import deque
from ..Primitives.camera import CameraImage
from ..Primitives.NeuralNetwork import CNN

import numpy as np
from ..utils.argumentparser import gazebo
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import imageio
import cv2


class ReplayBuffer(object):
    def __init__(self, buffer_size) -> None:
        self.count = 0
        self.buffer_size = buffer_size
        self.buffer = deque()

    def add(self, value):
        if self.count < self.buffer_size:
            self.buffer.append(value)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(value)

    def clear(self):
        self.buffer.clear()
        self.count = 0


class ImageSeparator(CameraImage, ReplayBuffer):
    def __init__(self, gazebo_=gazebo, n_images=(2, 4), show_img=True, image_width=640, image_height=480, fps=30):
        CameraImage.__init__(self, gazebo_, image_width, image_height, fps, show_img)

        buffer_size = 0
        for n in n_images:
            buffer_size += n*n
            
        ReplayBuffer.__init__(self, buffer_size)

        self.n_images = n_images        # number of images that should be spitted
        self.img = None

    def set_current_frame(self, img):
        self.img = img
    
    def seperator(self):
        x, y = 0, 0
        for number in self.n_images:
            h = int(self.image_height / number)
            w = int(self.image_width / number)

            for i in range(number):
                for j in range(number):
                    x = i * h
                    y = j * w
                    
                    ext_img = self.extract_image(x, y, w, h)
                    
                    if isinstance(ext_img, np.ndarray):
                        self.add(ext_img)

        #if not self.sub_images:
        #    self.refactor_images()

    def refactor_images(self): #TODO
        # This code is only for n_images (2, 4)
        # Rearange most important sub images
        last_img = self.sub_images[-self.n_images[-1]:]
        self.sub_images = self.sub_images[:-self.n_images[-1]]

        self.sub_images[4:4] = last_img
    
    def extract_image(self, x, y, w, h):
        if isinstance(self.img, np.ndarray):
            #print("max row: {}, max column: {}".format(x+h, y+w))
            # if y+w < self.image_height and x+h < self.image_width:
            return self.img[x:x+h, y:y+w, :]  
        else:
            return None
        
    def save_seperated_image(self):
        if self.buffer:
            for _ in range(self.count):
                element = self.buffer.popleft()
            
                if isinstance(element, np.ndarray):
                    date = datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p_%f")
                    filename = str(f"filename_{date}" + ".jpg")
                
                    cv2.imwrite(filename, element)
            self.clear()


class TrafficSignSearch(CNN, ImageSeparator):
    def __init__(self, path, n_images=(2, 4), image_width=320, image_height=240, fps=30) -> None:
        CNN.__init__(path)
        ImageSeparator.__init__(n_images=(2, 4), image_width=320, image_height=240, fps=30)

    def search_traffic_signs(self):
        
        self.predicitons = []
        if not self.sub_images:
            for img in self.sub_images:
                y_pred = self.evaluate(img)

                if y_pred[0] == 1:
                    self.predicitons.append(y_pred)

