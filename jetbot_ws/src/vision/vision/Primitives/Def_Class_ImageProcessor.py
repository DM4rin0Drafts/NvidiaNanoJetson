from std_msgs.msg import Float64
from sensor_msgs.msg import Image
from rclpy.node import Node
import rclpy

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from cv_bridge import CvBridge


class ImageProcessor(Node):
    """
    Class providing the functions to process camera image and to output a (steering) angle to wheel velocity controller
    """

    def __init__(self, gazebo):
        super().__init__("img_processor", namespace="jetbot")
        #self.angle_pub = self.create_publisher(Float64, "/jetbot/angle", qos_profile=10)
        self.image_sub = self.create_subscription(Image, "/jetbot/camera/image_raw", self._getImage, qos_profile=10)

        self.gazebo = gazebo
        self.model = None
        self.model_turnRight = None
        self.model_turnLeft = None
        self.device = None
        self.mean = None
        self.std = None

        self.br = CvBridge()
        self.img = None

        return

    def publishAngle(self, angle):
        angle_msg = Float64()
        angle_msg.data = float(angle)
        self.angle_pub.publish(angle_msg)
        return

    def _getImage(self, img_msg):
        if self.gazebo:
            self.img = self.br.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        else:
            self.img = self.br.imgmsg_to_cv2(img_msg)

    def initModel(self):
        # Initialize ResNet
        # self.model = torchvision.models.resnet18(pretrained=False)
        # self.model.fc = torch.nn.Linear(512, 1)

        # ToDo: Initialisiere beide Unternetze self.model_turnRight und self.model_turnLeft

        # Initialize SqueezeNet
        self.model = torchvision.models.squeezenet1_1(pretrained=False)
        self.model.classifier[1] = torch.nn.Conv2d(512, 1, kernel_size=1)
        self.model.num_classes = 1

        if not self.gazebo:
            # following lines (commented out) in case of gpu (real Jetbot)
            self.model.load_state_dict(
                torch.load('/home/jetbot/Robotikseminar/jetbot_ws/src/vision/vision/utils/steering_model_angle_gpu.pth'))
            self.device = torch.device('cuda')
            self.model = self.model.to(self.device)
            self.model = self.model.eval().half()
            self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
            self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()

        else:
            # following lines in case of cpu (simulation)
            self.model.load_state_dict(
                torch.load('/home/jetbot/Robotikseminar/jetbot_ws/src/vision/vision/utils/squeezenet_V1_CPU.pth',
                           map_location='cpu'))  # additional parameter map_location only needed for PC with cpu
            self.device = torch.device('cpu')  # cuda instead of cpu at jetbot
            self.model = self.model.to(self.device)
            self.model = self.model.eval()
            self.mean = torch.Tensor([0.485, 0.456, 0.406]).cpu()
            self.std = torch.Tensor([0.229, 0.224, 0.225]).cpu()
        return

    def _preprocess(self, image: np.ndarray):
        if not self.gazebo:
            # in case of gpu (real Jetbot)
            image = transforms.functional.to_tensor(image).to(self.device).half()
        else:
            # in case of cpu (simulation)
            image = transforms.functional.to_tensor(image).to(self.device)

        image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return image[None, ...]

    def calcAngle(self, image):
        preprocessed_img = self._preprocess(image)
        # ToDo: if-Abfrage über self.sign mit für Wahl des Netzes zur Ausgabe des Winkels
        angle = self.model(preprocessed_img).detach().float().cpu().numpy().flatten()
        angle = np.deg2rad(angle-90)
        return angle

    def updateImage(self):
        rclpy.spin_once(self, timeout_sec=None)
        return

    def image2angle(self):
        self.updateImage()
        angle = self.calcAngle(self.img)
        return angle
