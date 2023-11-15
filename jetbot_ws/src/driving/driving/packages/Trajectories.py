import numpy as np


def calc_trajectory(path_type, traj_type, v_start, v_goal, a_max, dt, curve_radius=0.125, curve_angle=np.pi / 2):
    if path_type == "straight":

        if traj_type == "trapezoidal":
            traj = Straight_Trapez(v_start, v_goal, dt, a_max)

    elif path_type == "curve":

        if traj_type == "trapezoidal":
            traj = Curve_Trapez(v_start, v_goal, curve_radius, curve_angle, dt, a_max)

        # elif ... for alternative trajectories

    # else/ elif path_type == "curve": ... for curves

    return traj


def Straight_Trapez(v_start, v_goal, dt, a_max):
    v = v_start
    omega = 0
    traj = {"lin_vel": [], "ang_vel": []}

    if v_start < v_goal:
        while v < v_goal:
            v = v + a_max * dt
            traj["lin_vel"].append(v)
            traj["ang_vel"].append(omega)
            # print(v)

    if v_start > v_goal:
        while v > v_goal:
            v = v - a_max * dt
            traj["lin_vel"].append(v)
            traj["ang_vel"].append(omega)

    return traj


def Curve_Trapez(v_start, v_goal, curve_radius, curve_angle, dt, a_max):
    # curve_radius > 0: left curve; curve_radius < 0: right curve
    traj = {"lin_vel": [], "ang_vel": []}
    if curve_radius != 0:
        v = v_start
        omega = 0
        omega_goal = v_goal / curve_radius
        theta = 0  # necessary?
        angle_limit = 0.75*curve_angle
        if v_start < v_goal:
            while v < v_goal:
                v = v + a_max * dt
                traj["lin_vel"].append(v)
                traj["ang_vel"].append(omega)

        if v_start > v_goal:
            while v > v_goal:
                v = v - a_max * dt
                traj["lin_vel"].append(v)
                traj["ang_vel"].append(omega)

        if omega_goal > 0:
            while omega < omega_goal and theta < curve_angle:
                omega = omega + a_max * dt  # new variable for angular acceleration?
                theta = theta + abs(omega) * dt
                traj["lin_vel"].append(v)
                traj["ang_vel"].append(omega)

        elif omega_goal < 0:
            while omega > omega_goal and theta < curve_angle:
                omega = omega - a_max * dt
                theta = theta + abs(omega) * dt
                traj["lin_vel"].append(v)
                traj["ang_vel"].append(omega)

        if theta < angle_limit:
            while theta < angle_limit:
                theta = theta + abs(omega) * dt
                traj["lin_vel"].append(v)
                traj["ang_vel"].append(omega)
        if theta >= angle_limit:
            while omega > 0:
                omega = omega - a_max * dt
                theta = theta + abs(omega) * dt
                traj["lin_vel"].append(v)
                traj["ang_vel"].append(omega)
                print("THETA======= " + str(theta))
            while omega < 0:
                omega = omega + a_max * dt
                theta = theta + abs(omega) * dt
                traj["lin_vel"].append(v)
                traj["ang_vel"].append(omega)
                print("THETA======= " + str(theta))

    else:
        v = 0
        omega = 0
        traj["lin_vel"].append(v)
        traj["ang_vel"].append(omega)

    return traj
