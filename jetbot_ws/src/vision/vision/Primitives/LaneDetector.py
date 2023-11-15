import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import numpy as np

class LaneDetector(object):
    def __init__(self, path, gazebo=False):
        self.gazebo = gazebo
        
        if self.gazebo:
            self.path = path + 'steering_model_angle_cpu.pth'
        else:
            self.path = path + 'squeezenet_V3_inv_GPU_conv.pth'
        self.model = self.load_model()

        self.model_turnRight = None
        self.model_turnLeft = None

    def load_model(self):
        # Initialize ResNet
        # self.model = torchvision.models.resnet18(pretrained=False)
        # self.model.fc = torch.nn.Linear(512, 1)

        # ToDo: Initialisiere beide Unternetze self.model_turnRight und self.model_turnLeft

        # Initialize SqueezeNet
        model = torchvision.models.squeezenet1_1(pretrained=False)
        model.classifier[1] = torch.nn.Conv2d(512, 1, kernel_size=1)
        model.num_classes = 1
        

        if not self.gazebo:
            # following lines (commented out) in case of gpu (real Jetbot)
            model.load_state_dict(
                torch.load(self.path))
            self.device = torch.device('cuda')
            model = model.to(self.device)
            model = model.eval().half()
            self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
            self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()
            

        else:
            # following lines in case of cpu (simulation)
            model.load_state_dict(
                torch.load(self.path,
                           map_location='cpu'))  # additional parameter map_location only needed for PC with cpu
            self.device = torch.device('cpu')  # cuda instead of cpu at jetbot
            model = model.to(self.device)
            model = model.eval()
            self.mean = torch.Tensor([0.485, 0.456, 0.406]).cpu()
            self.std = torch.Tensor([0.229, 0.224, 0.225]).cpu()
        print(self.mean, self.std)

        return model

    def _preprocess(self, image: np.ndarray):
        # print("img: ", type(image))
        # print("std: {}, mean: {}", type(self.std), type(self.mean))
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

    def image2angle(self, img):
        angle = self.calcAngle(img)
        return angle
