import time
import numpy as np
import rclpy

from .Def_Class_PID_Controller import PID_Controller, calc_error_value
from .Def_Class_PubSub import TwistPublisher, OdometrySubscriber
from .Trajectories import calc_trajectory


class Robot:
    """
    Class providing the structure of the Jetbot and functions for odometry, trajectory planning and control
    """

    def __init__(self, time_step=1. / 60.):
        self.a_max = 0.1  # in m/s²
        self.time_step = time_step  # 1/FPS

        self.lin_vel = 0.
        self.ang_vel = 0.
        self.trajectory = {}

        self.vel_pub = TwistPublisher("twist_controller")  # evtl. change node name into "controller"
        self.vel_sub = OdometrySubscriber("odometry_getter")

        self.lin_vel_controller = PID_Controller(1., 0.0, 0.0, self.time_step)
        self.ang_vel_controller = PID_Controller(1., 0.80, 0.0, self.time_step)

    def updateVelocities(self):
        rclpy.spin_once(self.vel_sub, timeout_sec=None)
        self.lin_vel = self.vel_sub.lin_vel
        self.ang_vel = self.vel_sub.ang_vel
        return

    def changeVelocities(self, goal_lin_vel, goal_ang_vel):
        self.vel_pub.publishTwist(goal_lin_vel, goal_ang_vel)
        # print("Des. Lin. Vel.: ", goal_lin_vel, " | Des. Ang. Vel.: ", goal_ang_vel)
        # self.updateVelocities()
        # print("Act. Lin. Vel.: ", self.lin_vel, " | Act. Ang. Vel.: ", self.ang_vel)
        return

    def controlVelocities_singleIteration(self, desired_lin_vel, desired_ang_vel):
        self.updateVelocities()
        print("Act. Lin. Vel.: ", self.lin_vel, " | Act. Ang. Vel.: ", self.ang_vel)

        e_lin_vel = calc_error_value(desired_lin_vel, self.lin_vel)
        e_ang_vel = calc_error_value(desired_ang_vel, self.ang_vel)

        cmd_lin_vel = self.lin_vel_controller.calc_controller_output(e_lin_vel)
        cmd_ang_vel = self.lin_vel_controller.calc_controller_output(e_ang_vel)

        self.changeVelocities(cmd_lin_vel, cmd_ang_vel)
        return

    def controlVelocities(self, traj):
        for ii in range(len(traj["lin_vel"])):
            self.controlVelocities_singleIteration(traj["lin_vel"][ii], traj["ang_vel"][ii])
            time.sleep(self.time_step)
        return

    # Methode zum Testen der Regler anhand einer Sprungantwort
    def controlVelocities_stepResponse(self, desired_lin_vel, desired_ang_vel):
        while True:
            self.controlVelocities_singleIteration(desired_lin_vel, desired_ang_vel)
            time.sleep(self.time_step)
        return

    def drive_straight(self, v_goal, traj_type):
        if self.lin_vel == v_goal:
            return

        self.trajectory.update(
            calc_trajectory("straight", traj_type, self.lin_vel, v_goal, self.a_max, self.time_step))

        for ii in range(len(self.trajectory["lin_vel"])):
            self.changeVelocities(self.trajectory["lin_vel"][ii], self.trajectory["ang_vel"][ii])
            time.sleep(self.time_step)
        return

    def drive_curve(self, v_goal, traj_type, curve_radius=0.125, curve_angle=np.pi / 2):
        self.trajectory.update(
            calc_trajectory("curve", traj_type, self.lin_vel, v_goal, self.a_max, self.time_step, curve_radius,
                            curve_angle))

        for ii in range(len(self.trajectory["lin_vel"])):
            self.vel_pub.publishTwist(self.trajectory["lin_vel"][ii], self.trajectory["ang_vel"][ii])
            # print("Lin. Vel.: ", self.lin_vel, " | Ang. Vel.: ", self.ang_vel)
            time.sleep(self.time_step)
        return
