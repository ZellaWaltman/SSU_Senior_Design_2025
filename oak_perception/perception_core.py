#!/usr/bin/env python3

"""
====================================================================================================
Title: OAK Perception Core
-----------------------------
Program Details:
-----------------------------
Purpose: The perception core of HandiMan's vision system, which runs on a Luxonis OAK-D W camera.
This program performs two parallel tasks: Hand detection utilizing a custom YOLOv8n hand model, 
and block detection with a seperate custom YOLOv8n model. Block detection utilizes the OAK-D W 
camera's depth perception and spatial detection network in order to provide 3D spatial coordinates
in the camera frame, which is then translated into the Dobot Magician's base frame utilizing 
a previous Apriltag calibration from apriltag_calibration.py.

Dependencies:
Date: 2026-01-07 3:13 PM PT
Author: Zella Waltman

OUTPUT: 
INPUT: 

Versions: 
   v1: Original Version
   v1.1: Added RGB Buffer to minimize delay between block detections & current RGB frame
====================================================================================================
"""

import depthai as dai # Luxonis OAK pipline API
import time
import yaml
import numpy as np

# ====================================================================================================
# CONFIG:
# ====================================================================================================
IMG_SIZE = 320 # 320x320 input resolution for YOLO, for faster perception - MUST BE SAME SIZE AS .blob!

# .blob file paths
HAND_BLOB = "/home/sd/Desktop/hand/hand.blob"
BLOCK_BLOB = "/home/sd/Desktop/robot_and_blocks/robot_and_blocks.blob"

# Confidence and IOU Thresholds
HAND_CONF = 0.2
BLOCK_CONF = 0.4
IOU_THRESH = 0.5

# BLOCK DETECTION CLASS LIST AND FILTER:
BLOCK_CLASSES = ["Block_Blue", "Block_Green", "Block_Red", "End_Effector", "Robot_Arm"]
BLOCK_CLASS_SET = {"Block_Blue", "Block_Green", "Block_Red"} # Blocks only filter

# Spatial constraints to filter out noisy/inaccurate depths (mm)
DEPTH_MIN = 200
DEPTH_MAX = 3000

# # of RGB frames kept for sequence-number synch
RGB_BUFFER_SIZE = 10

# Apriltag Calibration yaml (from apriltag_calibration.py)
CALIB_FILE = "/home/sd/calibration/camera_robot_calibration.yaml"
# ====================================================================================================

# Load Apriltag Calibration File & Parse Into Python Dict
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def load_calibration(calib_file):
    with open(calib_file, "r") as f: # open calib file read-only, close when done
        data = yaml.safe_load(f) # convert yaml -> dict

    R = np.array(data["rotation_matrix"], dtype=float) # 3x3 rot matrix
    t = np.array(data["translation_m"], dtype=float).reshape(3) # 3x1 translation vector

    return R, t # Return calibration transform matrix (Cam -> Robot Base)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# RGB Buffer to store & sequence recent frames (for synchronization)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class RGBBuffer:
    # Construct buffer w/ max frames to keep
    def __init__(self, size=10):
        self.size = size
        self.buffer = {} # dictionary (seq #: RGB frame)

    # Add new RGB Frame to buffer
    def add(self, frame):
        seq = frame.getSequenceNum() # Get seq # of DepthAI packet (RGB, Depth, NN output)
        self.buffer[seq] = frame.getCvFrame() # current RGB frame corresponds to seq #
        if len(self.buffer) > self.size:
            del self.buffer[min(self.buffer.keys())] # Drop oldest frame  (smallest key)

    # When Block Detection exists, draw on correct (synced) frame
    def get(self, seq):
        return self.buffer.get(seq, None)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Perception Core
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class OakPerception:
    
    def __init__(self):
        self.pipeline = self._create_pipeline() # Build DepthAI Pipeline
        self.device = dai.Device(self.pipeline) # Upload pipeline to OAK camera

	# Queue size = 1, will store 1 frame
        # maxSize=1, blocking=False avoids app stalling if one stream lags; old frames drop instead
        self.q_rgb = self.device.getOutputQueue("rgb", maxSize=1, blocking=False)
        self.q_hand = self.device.getOutputQueue("hand", maxSize=1, blocking=False)
        self.q_block = self.device.getOutputQueue("block", maxSize=1, blocking=False)

	# Init Buffer
        self.rgb_buffer = RGBBuffer(RGB_BUFFER_SIZE)
        self.last_hand = []

	# Load Calibration Data
        self.R, self.t = load_calibration(CALIB_FILE)

    def _create_pipeline(self):
        pipeline = dai.Pipeline()

        # RGB camera
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setPreviewSize(IMG_SIZE, IMG_SIZE)
        cam.setInterleaved(False) # Color Channels stored seperately
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR) # OpenCV compatible format

        # Stereo (depth) cameras
        monoL = pipeline.create(dai.node.MonoCamera)
        monoR = pipeline.create(dai.node.MonoCamera)
        monoL.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        monoR.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        monoL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoR.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY) # High Density depth (better accuracy)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A) # Align depth to RGB camera
        stereo.setLeftRightCheck(True) # Consistency checks between left & right cams
        stereo.setSubpixel(False)

        monoL.out.link(stereo.left)
        monoR.out.link(stereo.right)

	# --------------------------------------
        # HAND DETECTION
        # --------------------------------------
	# - Hand detection runs on the OAK & does not provide 3D (depth) info

        hand_nn = pipeline.create(dai.node.YoloDetectionNetwork)
        hand_nn.setBlobPath(HAND_BLOB)
        hand_nn.setConfidenceThreshold(HAND_CONF)
        hand_nn.setIouThreshold(IOU_THRESH)
        hand_nn.setNumClasses(1)
        hand_nn.setCoordinateSize(4)

        cam.preview.link(hand_nn.input)

        # --------------------------------------
        # BLOCK DETECTION
	# --------------------------------------
	# - Block Detection runs on the OAK & provides 3D Spatial info

        block_nn = pipeline.create(dai.node.YoloSpatialDetectionNetwork) # Spatial Detection provides DepthAI ROI median
        block_nn.setBlobPath(BLOCK_BLOB)
        block_nn.setConfidenceThreshold(BLOCK_CONF)
        block_nn.setIouThreshold(IOU_THRESH)
        block_nn.setNumClasses(len(BLOCK_CLASSES))
        block_nn.setCoordinateSize(4)
	
	# SpatialDetectionNetwork Depth ROI Control
        block_nn.setBoundingBoxScaleFactor(0.3)
        block_nn.setDepthLowerThreshold(DEPTH_MIN)
        block_nn.setDepthUpperThreshold(DEPTH_MAX)

        cam.preview.link(block_nn.input)
        stereo.depth.link(block_nn.inputDepth)

        # Outputs
	# -------------------------------------
	# RGB Stream
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam.preview.link(xout_rgb.input)

	# Hand Detections
        xout_hand = pipeline.create(dai.node.XLinkOut)
        xout_hand.setStreamName("hand")
        hand_nn.out.link(xout_hand.input)

	# Block Detections
        xout_block = pipeline.create(dai.node.XLinkOut)
        xout_block.setStreamName("block")
        block_nn.out.link(xout_block.input)

        return pipeline
	
    # Run & process one frame, return dict
    def step(self):
        
        # RGB
	# - - - - - - - - - - - - -
        in_rgb = self.q_rgb.tryGet()
        if in_rgb is None:
            return None # If no new frame, skip

        self.rgb_buffer.add(in_rgb) # Store frame in buffer for sync

        # Hand (async)
        # - - - - - - - - - - - - -
	# Drain Queue completely (keep only recent hand detections)
        while True:
            pkt = self.q_hand.tryGet()
            if pkt is None:
                break
            self.last_hand = pkt.detections

	# Bool (binary) safety signal: 1 = present, 0 = not present
        hand_present = len(self.last_hand) > 0

        # Block (frame-synced)
        # - - - - - - - - - - - - -
        blocks = [] # List of detected blocks in frame
        while True:
            pkt = self.q_block.tryGet()
            if pkt is None:
                break # Drain detection queue

            frame = self.rgb_buffer.get(pkt.getSequenceNum()) # Ensure detections are synced to correcct frame
            if frame is None:
                continue # Skip if misaligned

            for d in pkt.detections:
                label = BLOCK_CLASSES[d.label] # Class index -> class name

                # FILTER OUT NON-BLOCKS
		# - - - - - - - - - - - -
                if label not in BLOCK_CLASS_SET:
                    continue

                # Get Camera-frame coordinates (meters)
                P_cam = np.array([
                    d.spatialCoordinates.x,
                    d.spatialCoordinates.y,
                    d.spatialCoordinates.z
                ], dtype=float) / 1000.0

                # Apply AprilTag calibration (Cam -> Robot Base Frame)
                P_robot = self.R @ P_cam + self.t

		# Store block data w/ label & robot frame coords
                blocks.append({
                    "label": label,
                    "x": float(P_robot[0]),
                    "y": float(P_robot[1]),
                    "z": float(P_robot[2]),
                    "confidence": float(d.confidence)
                })

	# Return structured detection result as dict
        return {
            "hand_present": hand_present,
            "blocks": blocks
        }
