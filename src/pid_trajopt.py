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
        for dof in trajectory:
            dof[2] -= np.pi
        return trajectory

    def update_robot(self, dof):
        if self.robot:
            dof = dof.copy().reshape((7))
            dof[2] += np.pi
            self.robot.SetDOFValues(np.append(dof, [0, 0, 0]))


if __name__ == '__main__':
	optimal_waypoints = np.array([[  1.00000000e+00,   3.00000000e+00,   0.00000000e+00,
          1.50000000e+00,   1.22464680e-16,   3.00000000e+00,
         -2.10602227e-16],
       [  1.48301459e+00,   2.70083492e+00,   1.59129496e-01,
          1.23172733e+00,  -3.92355156e-02,   3.06403251e+00,
         -5.45420957e-02],
       [  1.96602918e+00,   2.40166985e+00,   3.18258991e-01,
          9.63454665e-01,  -7.84710312e-02,   3.12806502e+00,
         -1.09084191e-01],
       [  2.17387651e+00,   2.28829424e+00,   2.66757106e-01,
          8.52550418e-01,  -8.30797001e-02,   3.00630806e+00,
         -1.49415761e-01],
       [  2.34732793e+00,   2.19814233e+00,   1.88926298e-01,
          7.61317224e-01,  -8.33600132e-02,   2.86132741e+00,
         -1.87971015e-01],
       [  2.50016836e+00,   2.17932442e+00,   1.63017979e-01,
          7.66416054e-01,  -1.03660058e-01,   2.73549865e+00,
         -2.26526268e-01],
       [  2.64711993e+00,   2.18088766e+00,   1.51944656e-01,
          7.99038319e-01,  -1.29680026e-01,   2.61514185e+00,
         -2.65081522e-01],
       [  2.77942139e+00,   2.14915431e+00,   1.44268268e-01,
          8.32594635e-01,  -1.72285848e-01,   2.48326904e+00,
         -2.72724629e-01],
       [  2.90439780e+00,   2.10077265e+00,   1.38290346e-01,
          8.66617977e-01,  -2.23184596e-01,   2.34563823e+00,
         -2.64911662e-01],
       [  3.01161104e+00,   2.06676738e+00,   1.37747509e-01,
          9.13785610e-01,  -2.73712919e-01,   2.28446898e+00,
         -2.44107694e-01],
       [  3.10461376e+00,   2.04426320e+00,   1.41552739e-01,
          9.71468675e-01,  -3.23944901e-01,   2.28446898e+00,
         -2.12910925e-01],
       [  3.18340691e+00,   2.01555634e+00,   1.48253781e-01,
          1.07373204e+00,  -3.92496003e-01,   2.30450982e+00,
         -1.80878216e-01],
       [  3.24443809e+00,   1.97909614e+00,   1.58574586e-01,
          1.23172079e+00,  -4.83946003e-01,   2.34960172e+00,
         -1.47800582e-01],
       [  3.28544782e+00,   1.93732044e+00,   1.63602524e-01,
          1.39311246e+00,  -5.78253540e-01,   2.39362063e+00,
         -9.32078972e-02],
       [  3.28641464e+00,   1.88491374e+00,   1.58044728e-01,
          1.56131001e+00,  -6.78276151e-01,   2.43549355e+00,
          4.41488727e-03],
       [  3.27731581e+00,   1.82763581e+00,   1.53506503e-01,
          1.73557004e+00,  -7.73691665e-01,   2.47622869e+00,
          1.07301779e-01],
       [  3.23298719e+00,   1.75330857e+00,   1.52536774e-01,
          1.93104879e+00,  -8.52982336e-01,   2.51298159e+00,
          2.28613044e-01],
       [  3.18865851e+00,   1.67949043e+00,   1.59481383e-01,
          2.12771254e+00,  -9.27491200e-01,   2.54377448e+00,
          3.55579739e-01],
       [  3.14432925e+00,   1.60974522e+00,   2.29740691e-01,
          2.33385627e+00,  -9.63745600e-01,   2.52688724e+00,
          5.27789870e-01],
       [  3.10000000e+00,   1.54000000e+00,   3.00000000e-01,
          2.54000000e+00,  -1.00000000e+00,   2.51000000e+00,
          7.00000000e-01]])
    controller = PIDController(None)
	controller.execute_trajectory(optimal_waypoints, 20.0)