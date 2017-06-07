#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from HelperMethods import *
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
import tf.broadcaster as tfb
import tf.listener as tfl
import tf.transformations as tft
import numpy as np
import signal
import csv
import matplotlib.pyplot as plt
from os import path

class OdometryNode:

    def __init__(self):
        self.log_scans = rospy.get_param("/odom/log_scans", False)    # Determines if log of laser scans should be written to file
        self.log_scans_file = None                              # File handler for laser log file
        self.log_scans_writer = None                            # Writer for scans
        self.log_odom_calc = rospy.get_param("/odom/log_odom_calc", False) # Determines if log of calculated odometry should be written to file
        self.log_odom_calc_file = None                          # File handler for calculated odometry log file
        self.log_odom_calc_writer = None                        # Writer for real odometry
        self.log_odom_real = rospy.get_param("/odom/log_odom_real", False) # Determines if log of real odometry should be written to file
        self.log_odom_real_file = None                          # File handler for laser log file
        self.log_odom_real_writer = None                        # Writer for odom
        self.robot_frame = rospy.get_param("robot_frame", "/robot0")
        self.map_frame = rospy.get_param("map_frame", "/map_static")
        self.can_go = False                                     # Determines if initial pose fas found from robot to map
        self.was_first_scan = False
        self.odom = {"x": 0., "y": 0., "angle": 0.}             # Odometry values: x and y positions and angle in radians
        self.deltas = {"x": 0., "y": 0., "angle": 0.}           # Deltas from scan to scan
        self.graph_angle_step = 1  # angle histogram graph discrete step in degrees
        self.graph_distance_step = 0.05  # x,y histogram graph discrete step im meters
        self.angles = []  # x axis for all graphs for angles
        self.distance = []  # x axis for all graphs for distances
        self.percentage_of_finite_lasers = 0  # percentage of lasers used in calculation
        self.angle_min = 0  # angle of the first laser beam
        self.angle_max = 0  # angle of the last laser beam
        self.angle_step = 0  # angle between two laser beams
        self.range_min = 0  # minimal range that laser can detect
        self.range_max = 0  # maximal range that laser can detect
        self.number_of_laser_beams=0
        self.raw_data = []  # contains 'scan.ranges' from LaserScan message
        self.s_ref_copy = []  # copy of referent laser scan, filtered
        self.s_curr_copy = []  # copy of current laser scan, filtered
        self.histogram_angle_s_ref = []  # data for y-axis for S_ref histogram
        self.histogram_angle_s_curr = []  # data for y-axis for S_curr histogram
        self.relative_angles_s_ref = []  # relative angles for S_ref
        self.relative_angles_s_curr = []  # relative angles for S_curr
        self.histogram_x_s_ref = []  # y-axis data for x histogram for S_ref
        self.histogram_y_s_ref = []  # y-axis data for y histogram for S_ref
        self.histogram_x_s_curr = []  # y-axis data for x histogram for S_curr
        self.histogram_y_s_curr = []  # y-axis data for y histogram for S_curr
        self.crosscorrelated_angles = []  # angle cross correlation result
        self.norm_crosscorrelated_angles = []
        self.crosscorrelated_x = []  # x cross correlation result
        self.crosscorrelated_y = []  # y cross correlation result
        self.norm_crosscorrelated_x = []  # x cross correlation result
        self.norm_crosscorrelated_y = []  # y cross correlation result
        self.pointsInRange = []  # filtered point, points in range
        self.have_init_pose = False  # is used for transformation waiting
        self.s_curr_untouched = []  # copy of scan without rotation or translation

        if self.log_scans:
            print("Logging of scan data enabled.")
            self.log_scans_file = open(path.expanduser("~")+'/scans.csv', 'w')
            self.log_scans_writer = csv.writer(self.log_scans_file, delimiter=',')
        if self.log_odom_calc:
            print("Logging of calculated odometry data enabled.")
            self.log_odom_calc_file = open(path.expanduser("~")+'/odom_calc.csv', 'w')
            self.log_odom_calc_writer = csv.writer(self.log_odom_calc_file, delimiter=',')
        if self.log_odom_real:
            self.subscriber_odom = rospy.Subscriber("odom", Odometry, self.odom_callback)
            print("Logging of real odometry data enabled.")
            self.log_odom_real_file = open(path.expanduser("~")+'/odom_real.csv', 'w')
            self.log_odom_real_writer = csv.writer(self.log_odom_real_file, delimiter=',')
        rospy.init_node('OdometryNode')

        # Catch SIGTERM and SIGINT signals
        signal.signal(signal.SIGINT, self.handler_stop_signals)
        signal.signal(signal.SIGTERM, self.handler_stop_signals)

        self.subscriber_laser = rospy.Subscriber("laser_scan", LaserScan, self.laser_callback)
        self.odom_publisher = rospy.Publisher("histograms_odom", Odometry, queue_size=50)
        self.broadcaster = tfb.TransformBroadcaster()
        self.tf_listener = tfl.TransformListener()
        print("Started OdometryNode")
        rospy.spin()

    def odom_callback(self,odom):
        pose = odom.pose.pose.position
        time = odom.header.stamp.to_sec()
        o = odom.pose.pose.orientation
        euler = tft.euler_from_quaternion([o.x, o.y, o.z, o.w])
        fileodomwriter(self.log_odom_real_writer, [time, pose.x, pose.y, euler[2]])

    def broadcast(self):
        try:
            odom_quat = tft.quaternion_from_euler(0, 0, self.odom["angle"])
            pos = (self.odom["x"], self.odom["y"], 0)

            current_time = rospy.Time.now()
            if self.log_odom_calc:
                fileodomwriter(self.log_odom_calc_writer, [current_time.to_sec(), self.odom["x"], self.odom["y"], self.odom["angle"]])

            self.broadcaster.sendTransform(
                pos,
                odom_quat,
                current_time,
                "odom",
                self.map_frame
            )

            odom = Odometry()
            odom.header.stamp = current_time
            odom.header.frame_id = self.map_frame
            odom.pose.pose = Pose(Point(pos[0], pos[1], pos[2]), Quaternion(*odom_quat))
            odom.child_frame_id = "odom"
            odom.twist.twist = Twist(Vector3(0, 0, 0), Vector3(0, 0, 0))
            self.odom_publisher.publish(odom)
        except:
            pass

    def handler_stop_signals(self, signum, frame):
        if self.log_scans:
            self.log_scans_file.close()
        if self.log_odom_calc:
            self.log_odom_calc_file.close()
        if self.log_odom_real:
            self.log_odom_real_file.close()
            self.subscriber_odom.unregister()
        self.odom_publisher.unregister()
        self.subscriber_laser.unregister()
        rospy.signal_shutdown("Ended by user")

    def laser_callback(self, scan):
        while not self.have_init_pose:
            self.get_init_pose()

        if not self.was_first_scan:
            # Gets metadata about scan resource
            self.range_max = scan.range_max
            self.range_min = scan.range_min
            self.angle_max = scan.angle_max
            self.angle_min = scan.angle_min
            self.angle_step = scan.angle_increment
            self.distance = init_distances(self.graph_distance_step, scan.range_max)
            self.angles = np.arange(-90, 90, self.graph_angle_step)
            self.number_of_laser_beams = len(scan.ranges)
            self.raw_data = scan.ranges
            self.pointsInRange = self.filter_points_in_range()
            self.s_ref_copy = self.pointsInRange[:]
            self.relative_angles_s_ref = calculate_relative_angles(self.s_ref_copy)
            self.description()
            if self.log_scans: filescanwriter(self.log_scans_writer, self.s_ref_copy, scan.header.stamp.to_sec())
            self.was_first_scan = True
        elif self.was_first_scan:
            self.raw_data = scan.ranges
            self.pointsInRange = self.filter_points_in_range()
            self.s_curr_copy = self.pointsInRange[:]
            if self.log_scans: filescanwriter(self.log_scans_writer, self.s_curr_copy, scan.header.stamp.to_sec())
            self.relative_angles_s_curr = calculate_relative_angles(self.s_curr_copy)

            # calculate angle histograms
            self.calculate_angle_histograms()
            a = [i[2] for i in self.histogram_angle_s_ref]
            b = [i[2] for i in self.histogram_angle_s_curr]
            self.crosscorrelated_angles = cross_correlation(a, b)
            self.norm_crosscorrelated_angles = cross_correlation(a, b,cc_type="normalized")
            j = np.argmax(self.crosscorrelated_angles)
            angle = self.angles[j]

            self.deltas["angle"] = np.deg2rad(angle)

            self.odom["angle"] += np.deg2rad(angle)
            self.odom["angle"] = wrap_to_pi(self.odom["angle"])

            # save copy of scans until the end
            self.s_curr_untouched = self.s_curr_copy[:]
            self.s_ref_copy = rotate(self.s_ref_copy, angle)

            # create x and y histograms after rotation
            (self.histogram_x_s_ref, self.histogram_y_s_ref) = self.calculate_x_y_histograms(self.s_ref_copy)
            (self.histogram_x_s_curr, self.histogram_y_s_curr) = self.calculate_x_y_histograms(self.s_curr_copy)

            # cross-correlate x and y histograms
            self.crosscorrelated_x = cross_correlation(self.histogram_x_s_ref, self.histogram_x_s_curr)
            self.crosscorrelated_y = cross_correlation(self.histogram_y_s_ref, self.histogram_y_s_curr)
            self.norm_crosscorrelated_x = cross_correlation(self.histogram_x_s_ref, self.histogram_x_s_curr, cc_type="normalized")
            self.norm_crosscorrelated_y = cross_correlation(self.histogram_y_s_ref, self.histogram_y_s_curr, cc_type="normalized")
            j_x = np.argmax(self.crosscorrelated_x)
            j_y = np.argmax(self.crosscorrelated_y)
            x_translation = self.distance[j_x]
            y_translation = self.distance[j_y]

            self.deltas["x"] = x_translation
            self.deltas["y"] = y_translation

            if self.deltas["angle"] != 0:
                self.odom["x"] += 0
                self.odom["y"] += 0
                print("Theta delta: {}".format(self.deltas["angle"]))
            else:
                y_component = self.deltas["x"] * np.sin(self.odom["angle"])
                x_component = self.deltas["x"] * np.cos(self.odom["angle"])
                self.odom["x"] += x_component
                self.odom["y"] += y_component
                print("X delta: {0}   Y delta: {1}".format(x_component, y_component))

            self.broadcast()
            # if self.deltas["x"] != 0:
            #     print("was change")
            #     self.draw_graphs()
            #     rospy.signal_shutdown("graph")
            self.replace_previous_with_current()

    def filter_points_in_range(self):
        # removes ranges that are below or above distance limit
        temp = []
        for index, laser_range in enumerate(self.raw_data):
            if np.isfinite(laser_range):
                angle = self.angle_min + self.angle_step * index
                x = laser_range * np.cos(angle)
                y = laser_range * np.sin(angle)
                temp.append((index, angle, laser_range, x, y))
        self.percentage_of_finite_lasers = float(len(temp))/len(self.raw_data) * 100
        print("% of lasers {}".format(np.round(self.percentage_of_finite_lasers,2)))
        return temp

    def calculate_x_y_histograms(self, points):
        # Calculates x and y coordinates and create histograms
        # TODO: makni y kalkulacije?
        x_distance_x_axis = np.array(self.distance[:])
        x_distance_y_axis = np.zeros((len(x_distance_x_axis),), dtype=np.int)
        y_distance_y_axis = np.zeros((len(x_distance_x_axis),), dtype=np.int)

        for index, angle, laser_range, x, y in points:
            idx_x = find_closest(x_distance_x_axis, x)
            idx_y = find_closest(x_distance_x_axis, y)
            x_distance_y_axis[idx_x] += 1
            y_distance_y_axis[idx_y] += 1
        return x_distance_y_axis, y_distance_y_axis

    def calculate_angle_histograms(self):
        # x axis with discrete values from -90 to 90 degrees with step
        s_curr_angle_histogram_x_axis = np.array(self.angles[:])

        # y axis that represents number points that have the same relative angle
        s_curr_angle_histogram_y_axis = np.zeros((len(self.angles),), dtype=np.int)

        for (alpha, i1, r1, i2, r2) in self.relative_angles_s_ref:
            idx = find_closest(s_curr_angle_histogram_x_axis, np.rad2deg(alpha))
            s_curr_angle_histogram_y_axis[idx] += 1

        # save angles and number of lasers per angle to histogram variable
        self.histogram_angle_s_ref = []
        for i in range(len(s_curr_angle_histogram_x_axis)):
            self.histogram_angle_s_ref.append((i, s_curr_angle_histogram_x_axis[i], s_curr_angle_histogram_y_axis[i]))

        # x axis with discrete values from -90 to 90 degrees with step
        s_ref_angle_histogram_x_axis = np.array(self.angles[:])

        # y axis that represents number points that have the same relative angle
        s_ref_angle_histogram_y_axis = np.zeros((len(self.angles),), dtype=np.int)

        for (alpha, i1, r1, i2, r2) in self.relative_angles_s_curr:
            idx = find_closest(s_ref_angle_histogram_x_axis, np.rad2deg(alpha))
            s_ref_angle_histogram_y_axis[idx] += 1

        # save angles and number of lasers per angle to histogram variable
        self.histogram_angle_s_curr = []
        for i in range(len(self.angles)):
            self.histogram_angle_s_curr.append((i, s_ref_angle_histogram_x_axis[i], s_ref_angle_histogram_y_axis[i]))

    def replace_previous_with_current(self):
        self.s_ref_copy = self.s_curr_untouched[:]
        self.relative_angles_s_ref = self.relative_angles_s_curr[:]

    def get_init_pose(self):
        try:
            # Find translation and orientation transformation between 'map' and 'robot_frame'
            (t, o) = self.tf_listener.lookupTransform(self.map_frame, self.robot_frame, rospy.Time(0))
            euler = tft.euler_from_quaternion(o)
            self.odom["angle"] = euler[2]
            self.odom["x"] = t[0]
            self.odom["y"] = t[1]
            self.broadcast()
            self.have_init_pose = True
        except:
            pass

    def description(self):
        print("min angle-> deg:{} rad:{}".format(np.rad2deg(self.angle_min),self.angle_min))
        print("max angle-> {}".format(np.rad2deg(self.angle_max),self.angle_max))
        print("angle step-> {}".format(self.angle_step))
        print("min range-> {}".format(self.range_min))
        print("max range-> {}".format(self.range_max))
        print("#lasers-> {}".format(self.number_of_laser_beams))


    def draw_graphs(self):

        plt.figure(facecolor='white')

        # x histograms
        plt.subplot(2, 2, 1)
        m = np.max(np.maximum(self.histogram_x_s_curr,self.histogram_x_s_ref))
        plt.plot(self.distance,self.histogram_x_s_curr,'r-',self.distance,self.histogram_x_s_ref,'b-')
        plt.title("X axis histogram")
        plt.ylabel("x-h (number of lasers)")
        plt.xlabel("distance (m)")
        plt.ylim([0,m+40])

        # x cross-correlation graph
        m = np.max(self.crosscorrelated_x)
        idx = np.round(self.distance[np.argmax(self.crosscorrelated_x)],2)
        plt.subplot(2,2,2)
        plt.plot(self.distance,self.crosscorrelated_x)
        plt.title("X axis cross-correlation")
        plt.ylim([0,m+4000])
        plt.ylabel("c(j)")
        plt.xlabel("distance (m)")
        # plt.text(idx, m, "{},{}".format(idx,m))

        # x normalized cross correlation
        mm = np.round(np.max(self.norm_crosscorrelated_x),3)
        mi = np.min(self.norm_crosscorrelated_x)
        idx = np.round(self.distance[np.argmax(self.norm_crosscorrelated_x)])
        plt.subplot(2,2,3)
        plt.plot(self.distance,self.norm_crosscorrelated_x,'b-')
        plt.title("Normalized x cross-correlation")
        plt.xlabel('distance (m)')
        plt.ylabel('cn(j)')
        plt.ylim([mi-1,mm+1])
        plt.text(idx,mm,"({},{})".format(idx,mm))
        plt.tight_layout()
        plt.show()

        # # y histograms
        plt.subplot(2,2,1)
        plt.plot(self.distance,self.histogram_y_s_curr,'r-',self.distance,self.histogram_y_s_ref,'b-')
        plt.title("Y axis histogram")
        plt.ylabel("y-h")
        plt.xlabel("distance (m)")

        # y cross-correlation graph
        plt.subplot(2,2,2)
        plt.plot(self.distance,self.crosscorrelated_y)
        plt.title("Y axis cross-correlate")
        plt.ylabel("c(j)")
        plt.xlabel("distance (m)")

        # y normalized cross correlation
        mm = np.round(np.max(self.norm_crosscorrelated_y),3)
        mi = np.min(self.norm_crosscorrelated_y)
        idx = np.round(self.distance[np.argmax(self.norm_crosscorrelated_y)])
        plt.subplot(2,2,3)
        plt.plot(self.distance,self.norm_crosscorrelated_x,'b-')
        plt.title("Normalized y cross-correlation")
        plt.xlabel('distance (m)')
        plt.ylabel('cn(j)')
        plt.ylim([mi-1,mm+1])
        plt.text(idx,mm,"({},{})".format(idx,mm))
        plt.tight_layout()
        plt.show()

        # angle histograms
        current_y = [h[2] for h in self.histogram_angle_s_curr]
        referent_y = [h[2] for h in self.histogram_angle_s_ref]
        plt.subplot(2, 2, 1)
        plt.plot(self.angles,current_y,'r-',self.angles,referent_y,'b-')
        plt.title("Angle Histogram")
        plt.ylabel('h (number of lasers)')
        plt.xlabel('angle (deg)')

        # angle cross-correlated histogram
        m = np.max(self.crosscorrelated_angles)
        idx = self.distance[np.argmax(self.crosscorrelated_angles)]
        plt.subplot(2, 2, 2)
        plt.plot(self.angles,self.crosscorrelated_angles,'r-')
        plt.title("Angle cross-correlation Histogram")
        plt.ylim([0,m+4000])
        plt.ylabel('c(j)')
        plt.xlabel('angle (deg)')
        # plt.text(idx,m,"({},{})".format(idx,m))

        # angle normalized cross correlation
        plt.subplot(2,2,3)
        m = np.max(self.norm_crosscorrelated_angles)
        idx = self.angles[np.argmax(self.norm_crosscorrelated_angles)]
        plt.plot(self.angles,self.norm_crosscorrelated_angles,'b-')
        plt.title("Normalized angle cross-correlation")
        plt.xlabel('angle (deg)')
        plt.ylabel('cn(j)')
        plt.ylim([-1.5,1.5])
        plt.text(m,idx,"({},{})".format(idx,np.round(m,3)))

        plt.tight_layout()
        plt.show()
