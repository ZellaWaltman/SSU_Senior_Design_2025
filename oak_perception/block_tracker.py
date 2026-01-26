#!/usr/bin/env python3

"""
====================================================================================================
Title: Block Tracker
-----------------------------
Program Details:
-----------------------------
Purpose: The purpose of this program is to track individual blocks on top of raw
perception detections. It assigns stable IDs to blocks over time by comparing
detections frame-by-frame using nearest-neighbot logic in robot base frame
coords. In other words, the program attempts to detect and track which block 
has moved.

Dependencies:
Date: 2026-01-20 5:35:35 PM PT
Author: Zella Waltman

OUTPUT: 
INPUT: 

Versions: 
   v1: Original Version
====================================================================================================
"""

import rclpy # ROS Client Library for Python
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose, ObjectHypothesis
from geometry_msgs.msg import PoseWithCovariance
import math
import time

# ====================================================================================================
# CONFIG:
# ====================================================================================================

MATCH_DISTANCE = 0.10 # Max allowed spatial distance (m) to match detects
TRACK_TIMEOUT = 1.0 # time (s) before confirmed object tracking expires if unseen
PENDING_TIMEOUT = 1.0 # time (s) before pending tracking candidate expires
NEW_TRACK_CONFIRM_FRAMES = 1 # frames before creating a new ID

# ====================================================================================================

class Track:
    def __init__(self, track_id, label, pos, t):
        self.id = track_id
        self.label = label
        self.pos = pos # (x, y, z)
        self.last_seen = t
        self.matched_this_frame = False

# - - - - - - - - - - - - - - - - - - - - -
# Pending Candidates
# - - - - - - - - - - - - - - - - - - - - -
class PendingTrack:

# - Temp candidate before becoming a real track
# - avoids instant ID creation from noise

    def __init__(self, label, pos, t):
        self.label = label # Class label
        self.pos = pos # (x, y, z) - robot base coords
        self.count = 1 
        self.last_seen = t # Timestamp of last successful match


# - - - - - - - - - - - - - - - - - - - - -
# Tracker
# - - - - - - - - - - - - - - - - - - - - -
class BlockTracker(Node):

# - Uses nearest-neighbor tracking

    def __init__(self):
        super().__init__('block_tracker')

    # Subscribe to raw perception output
        self.sub = self.create_subscription(
            Detection3DArray,
            '/perception/blocks',
            self.callback,
            10
        )

    # Publish tracked detections
        self.pub = self.create_publisher(
            Detection3DArray,
            '/perception/tracked_blocks',
            10
        )

    # Internal tracker state
        self.tracks = [] # Confirmed tracks
        self.pending = [] # Candidate tracks (awaiting confirm)
        self.next_id = 0 # ID Counter (reset on node restart)

    # Callback - called once per incoming frame
    def callback(self, msg):
        now = self.get_clock().now().nanoseconds * 1e-9
    
    # Reset match flags at start of each frame
        for track in self.tracks:
             track.matched_this_frame = False

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

        # -----------------------------------------------------------------
        # Match detections to existing tracks (nearest-neighbor per class)
        # -----------------------------------------------------------------
        matched_tracks = set()

        for det in detections:
            best_track = None
            best_dist = float('inf')

        # Search for closest compatible track
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

        # DECISION: Match or treat as new object?
        # If object not detected last frame, match = stricter
            if best_track:
                if best_track.matched_this_frame:
                    max_dist = MATCH_DISTANCE
                else:
                    max_dist = MATCH_DISTANCE * 0.6

                if best_dist < max_dist:
                    best_track.pos = det["pos"]
                    best_track.last_seen = now
                    best_track.matched_this_frame = True
                    continue

            else:
                # No match = candidate for new tracked object
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
            if now - p.last_seen < PENDING_TIMEOUT
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


    # - - - - - - - - - - - - - - - - - - - - -
    # Update Tracked Objects
    # - - - - - - - - - - - - - - - - - - - - -
    def _update_pending(self, det, now):

        # Update or create pending track - real track only after confirmation!

        for p in self.pending:
            if p.label != det["label"]:
                continue

            dx = det["pos"][0] - p.pos[0]
            dy = det["pos"][1] - p.pos[1]
            dz = det["pos"][2] - p.pos[2]
            # dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        # Ignore z for now as it is most noisy 
            dist = math.sqrt(dx*dx + dy*dy)

            if dist < MATCH_DISTANCE:
                p.pos = det["pos"]
                p.count += 1
                p.last_seen = now

        # Promode pending object candidate to confirmed track
                if p.count >= NEW_TRACK_CONFIRM_FRAMES:
                    track_id = f"{p.label}_{self.next_id}"
                    self.next_id += 1

                    self.tracks.append(
                        Track(track_id, p.label, p.pos, now)
                    )

                    self.pending.remove(p)
                return

        # No matching pending track = create new one
        self.pending.append(PendingTrack(det["label"], det["pos"], now))

# ====================================
# MAIN CODE
# ====================================
def main():
    rclpy.init()
    node = BlockTracker()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
