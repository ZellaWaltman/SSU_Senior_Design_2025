#!/usr/bin/env python3

"""
====================================================================================================
Title: OAK Perception Node
-----------------------------
Program Details:
-----------------------------
Purpose: This program initializes and runs the complete perception stack of the
HandiMan project, including initializing ROS and its vision stack publishers. 

Dependencies:
Date: 2026-01-06 6:46:24 PM PT
Author: Zella Waltman

OUTPUT: 
INPUT: 

Versions: 
   v1: Original Version
====================================================================================================
"""

import rclpy # ROS Client Library for Python
from oak_perception.perception_core import OakPerception
from oak_perception.ros_interface import OakROSInterface

def main():
    rclpy.init() # Start ROS Client lib, set up DDS, time, logging

    perception = OakPerception() # Create perception core
    ros_iface = OakROSInterface() # Create ROS2 interface

    try:
        while rclpy.ok(): # Run while ROS2 = alive

            rclpy.spin_once(ros_iface, timeout_sec=0.0) # handle logging, parameters, shutdown

            result = perception.step() # Call perception core & run detection step on frame

        # If no new data
            if result is None:
                continue

        # Publish detection results (convert to ROS messages)
        # - - - - - - - - - - - - - - - - - - - - - - - - - -
            ros_iface.publish_hand(result["hand_present"])
            ros_iface.publish_blocks(result["blocks"])

    except KeyboardInterrupt:
        pass

    # ROS2 shutdown
    # - - - - - - - - - - -
    ros_iface.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
