import time

from rclpy.node import Node
from std_msgs.msg import Float64
from std_msgs.msg import String
from geometry_msgs.msg import Twist

import rclpy
import numpy as np
import time as t

rpm_max = 200
wheel_diameter = 0.065
wheel_separation = 0.12
# ToDo: Choose safety factor for maximum feasible speed
safety_factor = 0.22
v_max = safety_factor * rpm_max / 60 * np.pi * wheel_diameter


# v_left and v_right within the range [-v_max, v_max]
def calcLinAngVel(v_left, v_right):
    lin_vel = (v_left + v_right) / 2
    ang_vel = (v_right - v_left) / wheel_separation
    return lin_vel, ang_vel


class Controller(Node):
    def __init__(self):
        super().__init__("controller", namespace="jetbot")
        self.angle_sub = self.create_subscription(Float64, "/jetbot/angle", self._getAngle, qos_profile=10)

        # ToDo: Subscriber an Topic mit Schilderkennung evtl. anpassen
        self.sign_sub = self.create_subscription(String, "/jetbot/sign", self._getSign, qos_profile=10)
        self.sign = None  # None bedeutet: Es wurde kein Schild registriert.
        self.signPriority = None

        self.speed = 1

        self.timer_start = None
        self.isTimer = False

        self.twist_pub = self.create_publisher(Twist, "/jetbot/cmd_vel", qos_profile=10)

        self.angle = None
        self.angle_last = 0.0
        self.angle_history = []
        self.angle_sum = 0.0
        return

    def _getAngle(self, angle_msg):
        self.angle = angle_msg.data
        return

    def updateAngle(self):
        rclpy.spin_once(self, timeout_sec=None)
        return

    def _getSign(self, sign_msg):
        pass  # remove this line later
        # ToDo: write update method to update self.sign

    def publishTwist(self, lin_vel, ang_vel):
        twist = Twist()

        twist.linear.x = float(lin_vel)
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = float(ang_vel)

        self.twist_pub.publish(twist)
        return

    # set timer with specific duration or check whether timer is active
    def compareTimer(self, duration):
        if not self.isTimer:
            self.timer_start = t.time()
            self.isTimer = True
        if (t.time() - self.timer_start) >= duration:
            self.isTimer = False
        return

    # input speed (as an element of the set [-1,1]) given as fraction of maximum speed
    # angle < 0 if jetbot to the right of the lane, otherwise angle > 0
    def calcControllerOutput(self, angle):
        # ToDo: Wähle abhängig von speed andere Reglerparameter?
        # PID Controller parameters
        k_p = 0.12
        k_i = 0.0
        k_d = 0.18

        self.angle_history.append(angle)
        while len(self.angle_history) > 10:
            self.angle_history.pop(0)

        self.angle_sum += angle
        pid = k_p * angle + k_i * self.angle_sum + k_d * (angle - self.angle_last)
        # Modified PID controller
        # pid = k_p * angle + k_i * sum(self.angle_history) + k_d * (angle - self.angle_last)
        self.angle_last = angle

        # Speed reduction
        # ToDo: Wähle abhängig von speed ein anderen Wert für brake_gain
        brake_gain = 0.00
        brake = brake_gain * np.mean(np.abs(self.angle_history))

        speed = self.speed * v_max  # transform from set [-1,1] to set [-v_max,v_max]
        v_left = max(min(speed - brake + pid, v_max), -v_max)
        v_right = max(min(speed - brake - pid, v_max), -v_max)

        lin_vel, ang_vel = calcLinAngVel(v_left, v_right)

        return lin_vel, ang_vel

    # meant for lane following
    def closedLoop(self):
        self.updateAngle()
        lin_vel, ang_vel = self.calcControllerOutput(self.angle)
        self.publishTwist(lin_vel, ang_vel)

    # ToDo: design open loop controller for turning left or right
    # meant for turning right/left due to sign
    def openLoop(self):
        """
        stoppen
        rotieren um 90°
        losfahren
        """
        self.publishTwist(0, 0)
        t.sleep(3)
        if self.sign == 'Rechts Abbiegen':
            self.publishTwist(0, -np.pi / 2)
        elif self.sign == 'Links Abbiegen':
            self.publishTwist(0, np.pi / 2)
        t.sleep(1)
        self.publishTwist(0, 0)
        t.sleep(1)
        # self.publishTwist(0.5, 0)

    def startController(self):
        # ToDo: update self.sign
        # vary speed depending on sign
        if self.sign == 'Stop':
            self.publishTwist(0, 0)
            t.sleep(2)
            self.sign = None
        elif self.sign == 'Achtung':
            self.speed = 0.5
            self.compareTimer(2)
            if not self.isTimer:
                self.speed = 1
                self.sign = None
        elif self.sign == '30':
            self.speed = 0.3
            self.sign = None
        elif self.sign == 'Aufhebung':
            self.speed = 1
            self.sign = None

        if self.sign == 'Rechts Abbiegen' or self.sign == 'Links Abbiegen':
            self.openLoop()
            self.sign = None
        else:
            self.closedLoop()
