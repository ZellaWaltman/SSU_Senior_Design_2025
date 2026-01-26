#!/usr/bin/env python3

"""
====================================================================================================
Title: Tracking Viewer
-----------------------------
Program Details:
-----------------------------
Purpose: This program provides a lightweight, human-readable view of the tracked block data
from block_tracker.py. It subscribes to the output of the block tracking system and prints
a simplified summary of each tracked object to the terminal.

Dependencies:
Date: 2026-01-06 6:30:14 PM PT
Author: Zella Waltman

OUTPUT: 
INPUT: 

Versions: 
   v1: Original Version
====================================================================================================
"""

import rclpy # ROS Client Library for Python
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray

# - - - - - - - - - - - - - - - - - - - - -
# Simplified Tracking Viewer
# - - - - - - - - - - - - - - - - - - - - -
class TrackingViewer(Node):

    # 
    def __init__(self):
        super().__init__('tracking_viewer')

    # Subscribe to tracked block output from block_tracker
        self.create_subscription(
            Detection3DArray,
            '/perception/tracked_blocks',
            self.cb,
            10
        )

    # Callback for each incoming tracked detection message
    def cb(self, msg):
        if not msg.detections:
            return
    
    # Simplified output for tracked blocks
    # FORMAT: ID (X, Y, Z)
        print("\n---- TRACKED BLOCKS ----")

        for det in msg.detections:
        # Skip detections without classification results
            if not det.results:
                continue

            hyp = det.results[0]
            pos = hyp.pose.pose.position
        
        # Simplified output
            print(
                f"{det.id:<16} "
                f"{hyp.hypothesis.class_id:<12} "
                f"x={pos.x:.3f} y={pos.y:.3f} z={pos.z:.3f}"
            )

# ====================================
# MAIN CODE
# ====================================
def main():
    rclpy.init()
    node = TrackingViewer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
