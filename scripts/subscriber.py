#!/usr/bin/env python

import rospy
from Odometry import OdometryNode

if __name__ == '__main__':
    try:
        on = OdometryNode()
    except rospy.ROSInterruptException:
        pass
