from sensor_msgs.msg import Image
from rclpy.node import Node

import cv2
from cv_bridge import CvBridge
from datetime import datetime
from rclpy.qos import QoSProfile


class CameraImage(Node):
    def __init__(self, gazebo, image_width, image_height, fps, show_img, node_name="camera") -> None:
        super().__init__(node_name, namespace="jetbot")
        self.image_width = image_width
        self.image_height = image_height
        self.fps = fps
        self.show_img = show_img

        self.gazebo = gazebo
        self.br = CvBridge()
        self.frame = None

        if not self.gazebo:
            self.img_pub = self.create_publisher(Image, "/jetbot/camera/image_raw", qos_profile=10)

    def ros_camera_pipline(self, msg):
        self.frame = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if self.show_img:
            cv2.imshow("camera", self.frame)
            cv2.waitKey(1)

    def jetbot_camera_pipline(self, capture_width=1280, capture_height=720):
        return (
                "nvarguscamerasrc sensor-id=%d !"
                "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
                "nvvidconv flip-method=%d ! "
                "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
                "videoconvert ! "
                "video/x-raw, format=(string)BGR ! appsink"
                % (
                    0,
                    capture_width,
                    capture_height,
                    self.fps,
                    0,
                    self.image_width,
                    self.image_height,
                )
        )

    def show_CSI_camera(self, show_img=True):
        window_title = "CSI Camera"
        video_capture = cv2.VideoCapture(self.jetbot_camera_pipline(), cv2.CAP_GSTREAMER)
        if video_capture.isOpened():
            try:
                window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
                while True:
                    _, self.frame = video_capture.read()

                    if not self.gazebo:
                        # transform frame into ImgMsg and publish it as message
                        img_msg = self.br.cv2_to_imgmsg(self.frame, encoding="passthrough")
                        self.img_pub.publish(img_msg)
                    if show_img:
                        if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                            cv2.imshow(window_title, self.frame)
                        else:
                            break
                        keyCode = cv2.waitKey(10) & 0xFF
                        if keyCode == 27 or keyCode == ord('q'):
                            break
            finally:
                video_capture.release()
                cv2.destroyAllWindows()
        else:
            print("Error: Unable to open camera")

    def save_csi_image(self, file_type='.bmp'):
        date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p_%f")
        filename = f"filename_{date}" + file_type

        print("Save image? (y/n): ")
        save_img = input()
        if save_img == "y" or save_img == "Y":
            cv2.imwrite(filename, self.frame)


class CameraConnector(CameraImage):
    def __init__(self, gazebo, collect_data, show_img=True, image_width=224, image_height=224, fps=10, topic='camera/image_raw'):
        super().__init__(gazebo, image_width, image_height, fps, show_img)
        #Node.__init__(self, node_name='image_subscriber', namespace='jetbot')
        #CameraImage.__init__(self, image_width, image_height, fps)

        self.topic = topic
        self.collect_data = collect_data

        self.subscriber = None
        self.video_capture = None
        self.msg = None

        if self.gazebo:
            self.subscriber = self.create_subscription(Image, self.topic,
                                                       self.get_ros_camera_image,
                                                       QoSProfile(depth=10))

    def get_ros_camera_image(self, msg):
        #self.get_logger().info('Receiving video img')
        self.ros_camera_pipline(msg)
        print("Frame height: ", msg.height)
        print("Frame width: ", msg.width)
        #print("\n")
