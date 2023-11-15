from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


class Publisher(Node):

    def __init__(self, node_name, msg_type, topic_name):
        super().__init__(node_name, namespace="jetbot")
        self.publisher_ = self.create_publisher(msg_type, topic_name, qos_profile=10)


class TwistPublisher(Publisher):

    def __init__(self, node_name):
        super().__init__(node_name, Twist, "/jetbot/cmd_vel")

    def publishTwist(self, lin_vel, ang_vel):
        twist = Twist()

        twist.linear.x = float(lin_vel)
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = float(ang_vel)

        self.publisher_.publish(twist)


class Subscriber(Node):

    def __init__(self, node_name, msg_type, topic_name, callback_function):
        super().__init__(node_name, namespace='jetbot')
        self.subscriber_ = self.create_subscription(msg_type, topic_name, callback_function, qos_profile=10)


class OdometrySubscriber(Subscriber):

    def __init__(self, node_name):
        super().__init__(node_name, Odometry, "/jetbot/odom", self.odom_msg_received)
        self.lin_vel = 0.0
        self.ang_vel = 0.0

    def odom_msg_received(self, odom_msg_):
        self.lin_vel = odom_msg_.twist.twist.linear.x
        self.ang_vel = odom_msg_.twist.twist.angular.z
