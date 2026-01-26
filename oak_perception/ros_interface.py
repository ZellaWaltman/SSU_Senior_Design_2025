#!/usr/bin/env python3

"""
====================================================================================================
Title: ROS Interface
-----------------------------
Program Details:
-----------------------------
Purpose: The purpose of this program is to translate existing perception data from the
oak_perception.py program into standard ROS2 messages. These messages include Detection3DArray,
Detection3D, ObjectHypothesisWithPose, ObjectHypothesis, and PoseWithCovariance.

Dependencies:
Date: 2026-01-07 3:15 PM PT
Author: Zella Waltman

OUTPUT: 
INPUT: 

Versions: 
   v1: Original Version
====================================================================================================
"""

import rclpy # ROS Client Library for Python
from rclpy.node import Node # Create ROS2 Nodes
from std_msgs.msg import Bool # Boolean
# Detection result messages w/ classifications & poses (coords, orientation)
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose, ObjectHypothesis
from geometry_msgs.msg import PoseWithCovariance

# ROS2 Node Class: Includes name, publishing topics, & ROS time
class OakROSInterface(Node):

    #-------------------------------------------------------------
    # Constructor - Init Node
    #-------------------------------------------------------------
    def __init__(self):
        super().__init__('oak_perception') # Name = oak_perception

        # - - - - - - - - - - - - - - - - - - - - - - - - 
    # Hand Detection Publisher
    # - - - - - - - - - - - - - -
    # - std_msgs/Bool - 1 for hand present, 0 for no hand
    # - publisher on topic "/perception/hand_detected"
    # - Queue Size = 1
        self.hand_pub = self.create_publisher(
            Bool,
            '/perception/hand_detected',
            1
        )

        # - - - - - - - - - - - - - - - - - - - - - - - - 
    # Block Detection Publisher
    # - - - - - - - - - - - - - - - - - - - - - - - - 
    # - vision_msgs.msg Detection3DArray - List of detections w/ classification & spatial coords
        # - publisher on topic "/perception/blocks"
        # - Queue Size = 10
        self.block_pub = self.create_publisher(
            Detection3DArray,
            '/perception/blocks',
            10
        )

        # Object ID counter (PLACEHOLDER FOR LATER TRACKER)
        self._next_id = 0

    #-------------------------------------------------------------
    # Hand publisher helper: convert Python bool -> ROS message
    #-------------------------------------------------------------
    def publish_hand(self, hand_present: bool):
        msg = Bool()
        msg.data = hand_present
        self.hand_pub.publish(msg) # Publish Message

    #-------------------------------------------------------------
    # Block publisher
    #-------------------------------------------------------------
    def publish_blocks(self, blocks):
    # Create Detection3DArray()
        msg = Detection3DArray()
        msg.header.stamp = self.get_clock().now().to_msg() # Timestamp
        msg.header.frame_id = "robot_base" # Coordinate frame of all detections

    # For every block detection in frame
        for block in blocks:
            det = Detection3D() # Individual detection results, w/ classification & pose

            # Per-detection header (object-level timestamp)
            det.header.stamp = msg.header.stamp
            det.header.frame_id = msg.header.frame_id

            # Assign Object ID (PLACEHOLDER FOR LATER TRACKER)
            det.id = f"{block['label']}_{self._next_id}"
            self._next_id += 1


            # Classification result
            # - - - - - - - - - - - - - - - - - - - - - 
            # One hypothesis per detection
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis = ObjectHypothesis()
            hyp.hypothesis.class_id = block["label"]
            hyp.hypothesis.score = block["confidence"]

            # Pose in robot base frame (comes from perception)
            pose = PoseWithCovariance()
            pose.pose.position.x = block["x"]
            pose.pose.position.y = block["y"]
            pose.pose.position.z = block["z"]

            # Orientation currently unknown = identity quaternion
            pose.pose.orientation.w = 1.0
    
            # Attatch pose to hypothesis
            hyp.pose = pose
            det.results.append(hyp) # Append hypothesis to detection

            msg.detections.append(det) # Append detection to full detections array for single frame

    # Publish the Message
        self.block_pub.publish(msg)







#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray, Detection3D
from vision_msgs.msg import ObjectHypothesisWithPose, ObjectHypothesis
from geometry_msgs.msg import PoseWithCovariance
import math
import time


# ===================== CONFIG =====================
MATCH_DISTANCE = 0.10          # meters
TRACK_TIMEOUT = 2.0            # seconds
NEW_TRACK_CONFIRM_FRAMES = 2   # frames before creating a new ID
# =================================================


class Track:
    def __init__(self, track_id, label, pos, t):
        self.id = track_id
        self.label = label
        self.pos = pos              # (x, y, z)
        self.last_seen = t


class PendingTrack:
    """
    Temporary candidate before becoming a real track.
    Prevents instant ID creation from noise.
    """
    def __init__(self, label, pos, t):
        self.label = label
        self.pos = pos
        self.count = 1
        self.last_seen = t


class BlockTracker(Node):

    def __init__(self):
        super().__init__('block_tracker')

        self.sub = self.create_subscription(
            Detection3DArray,
            '/perception/blocks',
            self.callback,
            10
        )

        self.pub = self.create_publisher(
            Detection3DArray,
            '/perception/tracked_blocks',
            10
        )

        self.tracks = []
        self.pending = []
        self.next_id = 0


    def callback(self, msg):
        now = self.get_clock().now().nanoseconds * 1e-9

        # ------------------------------------------------
        # Extract detections into simple form
        # ------------------------------------------------
        detections = []
        for det in msg.detections:
            if not det.results:
                continue

            hyp = det.results[0]
            pos = hyp.pose.pose.position

            detections.append({
                "label": hyp.hypothesis.class_id,
                "pos": (pos.x, pos.y, pos.z)
            })

        # ------------------------------------------------
        # Match detections to existing tracks
        # ------------------------------------------------
        matched_tracks = set()

        for det in detections:
            best_track = None
            best_dist = float('inf')

            for track in self.tracks:
                if track.label != det["label"]:
                    continue

                dx = det["pos"][0] - track.pos[0]
                dy = det["pos"][1] - track.pos[1]
                dz = det["pos"][2] - track.pos[2]
                dist = math.sqrt(dx*dx + dy*dy + dz*dz)

                if dist < best_dist:
                    best_dist = dist
                    best_track = track

            if best_track and best_dist < MATCH_DISTANCE:
                best_track.pos = det["pos"]
                best_track.last_seen = now
                matched_tracks.add(best_track)
            else:
                # No match → candidate for new track
                self._update_pending(det, now)

        # ------------------------------------------------
        # Remove stale confirmed tracks
        # ------------------------------------------------
        self.tracks = [
            t for t in self.tracks
            if now - t.last_seen < TRACK_TIMEOUT
        ]

        # ------------------------------------------------
        # Remove stale pending tracks
        # ------------------------------------------------
        self.pending = [
            p for p in self.pending
            if now - p.last_seen < TRACK_TIMEOUT
        ]

        # ------------------------------------------------
        # Publish tracked detections
        # ------------------------------------------------
        out = Detection3DArray()
        out.header = msg.header

        for track in self.tracks:
            det = Detection3D()
            det.header = msg.header
            det.id = track.id

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis = ObjectHypothesis()
            hyp.hypothesis.class_id = track.label
            hyp.hypothesis.score = 1.0

            pose = PoseWithCovariance()
            pose.pose.position.x = track.pos[0]
            pose.pose.position.y = track.pos[1]
            pose.pose.position.z = track.pos[2]
            pose.pose.orientation.w = 1.0

            hyp.pose = pose
            det.results.append(hyp)
            out.detections.append(det)

        self.pub.publish(out)


    def _update_pending(self, det, now):
        """
        Update or create a pending track.
        Promote to a real track only after confirmation.
        """
        for p in self.pending:
            if p.label != det["label"]:
                continue

            dx = det["pos"][0] - p.pos[0]
            dy = det["pos"][1] - p.pos[1]
            dz = det["pos"][2] - p.pos[2]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)

            if dist < MATCH_DISTANCE:
                p.pos = det["pos"]
                p.count += 1
                p.last_seen = now

                if p.count >= NEW_TRACK_CONFIRM_FRAMES:
                    track_id = f"{p.label}_{self.next_id}"
                    self.next_id += 1

                    self.tracks.append(
                        Track(track_id, p.label, p.pos, now)
                    )

                    self.pending.remove(p)
                return

        # No matching pending track → create one
        self.pending.append(PendingTrack(det["label"], det["pos"], now))

def main():
    rclpy.init()
    node = BlockTracker()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
