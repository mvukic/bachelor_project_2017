#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def find_closest(array, target):
    # Finds index of the closest number in 'array' to 'target'
    idx = array.searchsorted(target)
    idx = np.clip(idx, 1, len(array) - 1)
    left = array[idx - 1]
    right = array[idx]
    idx -= target - left < right - target
    return idx


def rotate(points, delta):
    # Rotates scans for 'angle'
    if delta == 0:
        return points
    temp = []
    for (index, laser_range, angle, x, y) in points:
        temp.append((index, laser_range, wrap_to_pi(angle+delta), x, y))
    return temp


def translate(points, x_delta, y_delta):
    # Translates every point in 'points' for 'x_delta' and 'y_delta'
    if x_delta == 0 and y_delta == 0:
        return points
    temp = []
    for index, laser_range, angle, x, y in points:
        temp.append((index, laser_range, angle, x+x_delta, y+y_delta))
    return temp


def calculate_relative_angles(points):
    # Returns relative angles between two subsequent points
    temp = []
    (index1, angle1, prev_range, x1, y1) = points[0]
    for index2, angle2, next_range, x2, y2 in points[1:]:
        alpha = np.arctan((y2-y1) / (x2-x1))
        temp.append((alpha, index1, prev_range, index2, next_range))
        angle1, x1, y1, index1, prev_range = angle2, x2, y2, index2, next_range
    return temp


def wrap_to_pi(angle):
    # Wraps angle between -pi and pi
    return -1.0 * ((-angle + np.pi) % (2.0 * np.pi) - np.pi)


def init_distances(distance_step, range_max):
    part1 = np.arange(0, range_max, distance_step)
    part2 = np.arange(-range_max, 0, distance_step)
    return np.concatenate([part2, part1])


def cross_correlation(array_1, array_2, cc_type="regular"):
    # Returns result of regular cross-correlation or normalized
    # cross-correlation depending on 'type' variable
    if cc_type == "regular":
        return np.correlate(array_1, array_2, "same")
    elif cc_type == "normalized":
        a = (array_1 - np.mean(array_1)) / (np.std(array_1) * len(array_1))
        v = (array_2 - np.mean(array_2)) / np.std(array_2)
        return np.correlate(a, v, "same")
    else:
        raise AttributeError("Wrong type of cross correlation.")


def filescanwriter(writer, data, timestamp):
    d = []
    d.append(timestamp)
    for (_,angle,laser_range,_,_) in data:
        d.append(angle)
        d.append(laser_range)
    writer.writerow(d)


def fileodomwriter(writer,data):
    writer.writerow(data)