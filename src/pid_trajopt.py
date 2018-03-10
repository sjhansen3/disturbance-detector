#!/usr/bin/env python
"""
Adapted from Andrea Bajcsy
https://github.com/abajcsy/iact_control/
"""
import roslib

roslib.load_manifest('kinova_demo')

import rospy
import math
import pid
import tf
import sys, select, os
import thread
import argparse
import actionlib
import time
import ros_utils

import kinova_msgs.msg
import geometry_msgs.msg
import std_msgs.msg
import sensor_msgs.msg
import kinova_msgs.srv


import numpy as np
from numpy import array

prefix = 'j2s7s300_driver'

epsilon = 0.10
MAX_CMD_TORQUE = 40.0
INTERACTION_TORQUE_THRESHOLD = 8.0


class PIDController(object):
    """
    This class represents a node that moves the Jaco with PID control.
    The joint velocities are computed as:

        V = -K_p(e) - K_d(e_dot) - K_i*Integral(e)
    where:
        e = (target_joint configuration) - (current joint configuration)
        e_dot = derivative of error
        K_p = accounts for present values of position error
        K_i = accounts for past values of error, accumulates error over time
        K_d = accounts for possible future trends of error, based on current rate of change

    Subscribes to:
        /j2s7s300_driver/out/joint_angles	- Jaco sensed joint angles
        /j2s7s300_driver/out/joint_torques	- Jaco sensed joint torques

    Publishes to:
        /j2s7s300_driver/in/joint_velocity	- Jaco commanded joint velocities

    Required parameters:
        p_gain, i_gain, d_gain    - gain terms for the PID controller
        sim_flag 				  - flag for if in simulation or not
    """

    def __init__(self, robot):
        """
        Setup of the ROS node. Publishing computed torques happens at 100Hz.
        """

        self.robot = robot
        self.executing = False
        self.stepping = False
        self.reached_start = False
        self.reached_goal = False
        self.last_dof = None
        self.target_index = 0
        self.step_size = 1

        # ----- Controller Setup ----- #

        # stores maximum COMMANDED joint torques
        self.max_cmd = MAX_CMD_TORQUE * np.eye(7)
        # stores current COMMANDED joint torques
        self.cmd = np.eye(7)
        # stores current joint MEASURED joint torques
        self.joint_torques = np.zeros((7, 1))

        # P, I, D gains
        p_gain = 50.0
        i_gain = 0.0
        d_gain = 20.0
        self.P = p_gain * np.eye(7)
        self.I = i_gain * np.eye(7)
        self.D = d_gain * np.eye(7)
        self.controller = pid.PID(self.P, self.I, self.D, 0, 0)

        # ---- ROS Setup ---- #

        rospy.init_node("pid_trajopt", anonymous=True)

        # create joint-velocity publisher
        self.vel_pub = rospy.Publisher(prefix + '/in/joint_velocity',
                                       kinova_msgs.msg.JointVelocity,
                                       queue_size=1)

        # create subscriber to joint_angles
        rospy.Subscriber(prefix + '/out/joint_angles',
                         kinova_msgs.msg.JointAngles,
                         self.joint_angles_callback, queue_size=1)
        # create subscriber to joint_torques
        rospy.Subscriber(prefix + '/out/joint_torques',
                         kinova_msgs.msg.JointTorque,
                         self.joint_torques_callback, queue_size=1)

    def execute_loop(self):
        r = rospy.Rate(100)

        while not rospy.is_shutdown() and not (self.reached_goal and
                                               self.reached_start):
            self.vel_pub.publish(ros_utils.cmd_to_JointVelocityMsg(self.cmd))
            r.sleep()

    def execute_trajectory(self, traj, duration=10.):

        self.start_admittance_mode()

        trajectory = self.fix_joint_angles(traj)
        self.trajectory = trajectory

        # ---- Trajectory Setup ---- #

        # total time for trajectory
        self.trajectory_time = duration

        self.start = trajectory[0].reshape((7, 1))
        self.goal = trajectory[-1].reshape((7, 1))

        self.target_pos = trajectory[0].reshape((7, 1))

        # track if you have gotten to start/goal of path
        self.reached_start = False
        self.reached_goal = False
        self.executing = True

        # keeps running time since beginning of path
        self.path_start_T = time.time()

        self.execute_loop()

        # end admittance control mode
        self.stop_admittance_mode()
        self.executing = False

    def step_trajectory(self, traj, starting_index=0, step_size=1):
        self.start_stepping(traj, starting_index)
        self.step_size = step_size
        while True:
            cmd = raw_input().lower()
            if 'n' in cmd:
                self.step_next()
            if 'p' in cmd:
                self.step_prev()
            if 'q' in cmd:
                break
        self.stop_stepping()
        return self.target_index

    def start_stepping(self, traj, starting_index):
        trajectory = self.fix_joint_angles(traj)
        self.trajectory = trajectory
        self.target_index = starting_index
        self.target_pos = trajectory[self.target_index].reshape((7, 1))
        self.goal = self.target_pos
        self.reached_start = True
        self.reached_goal = False
        self.stepping = True
        self.execute_loop()

    def stop_stepping(self):
        self.stepping = False

    def step_next(self):
        self.target_index += self.step_size
        if self.target_index > len(self.trajectory) - 1:
            self.target_index = len(self.trajectory) - 1
        self.target_pos = self.trajectory[self.target_index].reshape((7, 1))
        self.goal = self.target_pos
        self.reached_goal = False
        self.execute_loop()

    def step_prev(self):
        self.target_index -= self.step_size
        if self.target_index == 0:
            self.target_index = 0
        self.target_pos = self.trajectory[self.target_index].reshape((7, 1))
        self.goal = self.target_pos
        self.reached_goal = False
        self.execute_loop()

    def grav_comp(self):
        self.start_admittance_mode()
        raw_input("Press enter to exit gravity compensation mode.")
        self.stop_admittance_mode()
        return self.last_dof

    def start_admittance_mode(self):
        """
        Switches Jaco to admittance-control mode using ROS services
        """

        service_address = prefix + '/in/start_force_control'
        rospy.wait_for_service(service_address)
        try:
            startForceControl = rospy.ServiceProxy(
                service_address,
                kinova_msgs.srv.Start
            )
            startForceControl()
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e
            return None

    def stop_admittance_mode(self):
        """
        Switches Jaco to position-control mode using ROS services
        """

        service_address = prefix + '/in/stop_force_control'
        rospy.wait_for_service(service_address)
        try:
            stopForceControl = rospy.ServiceProxy(
                service_address,
                kinova_msgs.srv.Stop
            )
            stopForceControl()
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e
            return None

    def PID_control(self, pos):
        """
        Return a control torque based on PID control
        """
        error = -((self.target_pos - pos + math.pi) % (2 * math.pi) - math.pi)
        return -self.controller.update_PID(error)

    def joint_torques_callback(self, msg):
        """
        Reads the latest torque sensed by the robot and records it for
        plotting & analysis
        """
        return

    def joint_angles_callback(self, msg):
        """
        Reads the latest position of the robot and publishes an
        appropriate torque command to move the robot to the target
        """
        # read the current joint angles from the robot
        curr_pos = np.array(
            [msg.joint1, msg.joint2, msg.joint3, msg.joint4, msg.joint5,
             msg.joint6, msg.joint7]).reshape((7, 1))

        # convert to radians
        curr_pos = curr_pos * (math.pi / 180.0)

        # update the OpenRAVE simulation
        # self.planner.update_curr_pos(curr_pos)
        self.update_robot(curr_pos)
        self.last_dof = curr_pos

        # update target position to move to depending on:
        # - if moving to START of desired trajectory or
        # - if moving ALONG desired trajectory
        self.update_target_pos(curr_pos)

        # update cmd from PID based on current position
        self.cmd = self.PID_control(curr_pos)
        #print "target pos: ", self.target_pos
        #print "robot commend: " , self.cmd

        # check if each angular torque is within set limits
        for i in range(7):
            if self.cmd[i][i] > self.max_cmd[i][i]:
                self.cmd[i][i] = self.max_cmd[i][i]
            if self.cmd[i][i] < -self.max_cmd[i][i]:
                self.cmd[i][i] = -self.max_cmd[i][i]

    def update_target_pos(self, curr_pos):
        """
        Takes the current position of the robot. Determines what the next
        target position to move to should be depending on:
        - if robot is moving to start of desired trajectory or
        - if robot is moving along the desired trajectory
        """
        if self.stepping:
            dist_from_goal = -((curr_pos - self.goal + math.pi) %
                               (2 * math.pi) - math.pi)
            if np.all(np.abs(dist_from_goal) < epsilon):
                self.reached_goal = True
            return

        if not self.executing:
            self.target_pos = curr_pos
            return

        # check if the arm is at the start of the path to execute
        if not self.reached_start:

            dist_from_start = -(
                    (curr_pos - self.start + math.pi) % (2 * math.pi) - math.pi)
            dist_from_start = np.fabs(dist_from_start)
            # print "d to start: ", np.linalg.norm(dist_from_start)

            # check if every joint is close enough to start configuration
            close_to_start = [dist_from_start[i] < epsilon for i in range(7)]

            # if all joints are close enough, robot is at start
            is_at_start = np.all(close_to_start)

            if is_at_start:
                self.reached_start = True
                self.path_start_T = time.time()
            else:
                self.target_pos = self.start.reshape((7, 1))
        else:
            t = time.time() - self.path_start_T

            self.target_pos = self.interpolate_trajectory(t)

            if not self.reached_goal:

                dist_from_goal = -((curr_pos - self.goal + math.pi) %
                                   (2 * math.pi) - math.pi)
                if np.all(np.abs(dist_from_goal) < epsilon):
                    self.reached_goal = True

            else:
                self.stop_admittance_mode()

    def interpolate_trajectory(self, time):
        n = len(self.trajectory)
        step = self.trajectory_time / (n - 1)

        if n == 1:
            print self.trajectory

        if time >= self.trajectory_time:
            target_pos = self.trajectory[-1]
        else:
            index = int(time / step)
            delta = self.trajectory[index + 1] - self.trajectory[index]
            diff = delta * (time - index * step) / step
            target_pos = diff + self.trajectory[index]
        return np.array(target_pos).reshape((7, 1))

    def fix_joint_angles(self, trajectory):
        trajectory = trajectory.copy()
        # for dof in trajectory:
        #     dof[2] -= np.pi
        return trajectory

    def update_robot(self, dof):
        if self.robot:
            dof = dof.copy().reshape((7))
            dof[2] += np.pi
            self.robot.SetDOFValues(np.append(dof, [0, 0, 0]))


if __name__ == '__main__':
	optimal_waypoints = np.asarray([[1.4026159729033887, 3.4399007841406206, 3.139112717978348, 0.7585550309264283,-1.6469238518197384, 4.490225951725771,
	5.026357563409073], [1.6336541209880149, 3.337647918563857, 3.0353076609441674, 0.8149932045443874, -1.6987336509953348, 4.420514392488953, 5.069412470747306],
	[1.864692270488171, 3.2353950547527504, 2.931502611280963, 0.8714313787092338, -1.750543448658653, 4.35080283378039, 5.112467379706972],
	[2.095730422157649, 3.133142194307773, 2.8276975882114, 0.92786955403402, -1.8023532420375856, 4.281091276358607, 5.1555222932162845],
	[2.32676857649305, 3.030889339299526, 2.723892643030677, 0.9843077313648195, -1.854163025126749, 4.211379721273314, 5.198577216823983],
	[2.55780920310372, 2.9286383188765726, 2.6200904628701345, 1.040746362983751, -1.9059728544515546, 4.141668165443907, 5.241632161346775],
	[2.7888513592901836, 2.826388939984241, 2.516290411723663, 1.0971852211941622, -1.957782701967882, 4.071956611999344, 5.284687149789819],
	[3.019908498954687, 2.724153438712805, 2.4125067048921967, 1.153624346411072, -2.009594315353172, 4.002240894603939, 5.327742284904397],
	[3.2509932204842267, 2.621937512074268, 2.308748417463299, 1.2100573709862938, -2.0614084875963603, 3.9325118621781257, 5.370797853922964],
	[3.6830718927468835, 2.687750855372267, 2.352792568215877, 1.2705896287029512, -2.0603066404401815, 3.815433662084912, 5.515694172372803]])
	#optimal_waypoints = np.array([[3.6830718927468835, 2.6877508553722667, 2.352792568215878, 1.2705896287029512, 4.222878666739405, 3.815433662084912, -0.7674911348067829], [3.5733276503632125, 2.9541093755189114, 2.8232156985858126, 1.394956494474402, 4.456915326423859, 3.548000250304093, -0.9678769424402349]])
	home = np.asarray([[80.363975525, 197.091796875, 179.857910156, 43.4620018005, -94.3617858887, 257.270996094, 287.989074707]])*(np.pi/180)
	controller = PIDController(None)
	controller.execute_trajectory(optimal_waypoints, 20.0)