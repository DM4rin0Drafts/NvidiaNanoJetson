#!/usr/bin/env python

import threading
import numpy as np
import rclpy
from .Primitives.Def_Class_ImageProcessor import ImageProcessor
from .utils.argumentparser import gazebo
import time
from .utils.utils import get_full_path
from .Primitives.LaneDetector import LaneDetector

def main(args=None):
    rclpy.init(args=args)

    img_processor = ImageProcessor(gazebo)
    t = threading.Thread(target=rclpy.spin, args=(img_processor,))
    t.start()

    model_path = get_full_path("jetbot_ws/src/vision/vision/utils/model/LaneModel/")
    lane = LaneDetector(model_path)
    
    while(True):
        img = img_processor.img
        
        if isinstance(img_processor.img, np.ndarray):
            angle = lane.image2angle(img)
            print("Angle: ", angle)

    #img_processor.initModel()

    #while True:
    #    ang = img_processor.image2angle()
    #    img_processor.publishAngle(ang)
    #    print(np.rad2deg(ang))

    # rclpy.spin(img_processor)

    # img_processor.destroy_node()
    # rclpy.shutdown()

    return


if __name__ == '__main__':
    main()
