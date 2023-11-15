#!/usr/bin/env python

import threading
import rclpy
from .Primitives.camera import CameraConnector
from .Primitives.ImageProcessing import ImageSeparator
from .utils.argumentparser import gazebo, collect_data
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2


def destroy(node):
    node.destroy_node()


def run(camera):
    sep = ImageSeparator(gazebo)

    while True:
        if gazebo and collect_data:
            # collecting data in gazebo
            # print("Collecting Data in Gazebo")
            # camera.save_csi_image()

            print("Seperate image? (y/n): ")
            save_img = input()
            sep.set_current_frame(np.asarray(camera.img))
            if save_img == "y" or save_img == "Y":
                sep.seperator()
                sep.save_seperated_image()

            time.sleep(1)

        elif gazebo and not collect_data:
            # running code in simulation
            print("Running Code in Gazebo")

            # image.show()
            time.sleep(10)

            pass

        if not gazebo and collect_data:
            # collect data in on jetbot waveshare robot
            print("Collect data on jetbot robot")

            print("Seperate image? (y/n): ")
            save_img = input()
            sep.set_current_frame(np.asarray(camera.img))
            if save_img == "y" or save_img == "Y":
                sep.seperator()
                sep.save_seperated_image()
            time.sleep(1)

        elif not gazebo and not collect_data:
            # running code in real world
            print("Running Vision package on jetbot robot")
            pass


def main(args=None):
    rclpy.init(args=args)

    # Create the node
    camera = CameraConnector(gazebo, collect_data, True)
    if gazebo:
        t = threading.Thread(target=rclpy.spin, args=(camera,))
    else:
        t = threading.Thread(target=camera.show_CSI_camera, args=(True,))

    t.start()
    time.sleep(3)
    run(camera)

    # destroy(test)
    # Shutdown the ROS client library for Python
    # rclpy.shutdown()


if __name__ == '__main__':
    main()
