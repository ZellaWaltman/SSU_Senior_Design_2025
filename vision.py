#!/usr/bin/env python3

import cv2
import depthai as dai
import time
from collections import deque

# CONFIG
# ================================================================
IMG_SIZE = 320

HAND_BLOB = "/home/sd/Desktop/hand/hand.blob"
BLOCK_BLOB = "/home/sd/Desktop/robot_and_blocks/robot_and_blocks.blob"

HAND_CONF = 0.2
BLOCK_CONF = 0.4
IOU_THRESH = 0.5

BLOCK_CLASSES = ["Block_Blue", "Block_Green", "Block_Red", "End_Effector", "Robot_Arm"]

# Spatial constraints (mm)
DEPTH_MIN = 200
DEPTH_MAX = 3000

RGB_BUFFER_SIZE = 10
# ================================================================

# -------------------------------------------------
# RGB Frame Buffer (for block synchronization)
# -------------------------------------------------
class RGBBuffer:
    def __init__(self, size=10):
        self.size = size
        self.buffer = {}

    def add(self, frame):
        seq = frame.getSequenceNum()
        self.buffer[seq] = frame.getCvFrame()
        if len(self.buffer) > self.size:
            del self.buffer[min(self.buffer.keys())]

    def get(self, seq):
        return self.buffer.get(seq, None)


# -------------------------------------------------
# Hand Detector (ASYNC, LOW LATENCY)
# -------------------------------------------------
class HandDetector:
    def __init__(self):
        self.last_detections = []

    def update(self, queue):
        while True:
            pkt = queue.tryGet()
            if pkt is None:
                break
            self.last_detections = pkt.detections

    def draw(self, frame):
        h, w = frame.shape[:2]
        for d in self.last_detections:
            x1 = int(d.xmin * w)
            y1 = int(d.ymin * h)
            x2 = int(d.xmax * w)
            y2 = int(d.ymax * h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame,
                "HAND",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )

# -------------------------------------------------
# Block Detector (SPATIAL, FRAME-SYNCHRONIZED)
# -------------------------------------------------
class BlockDetector:
    def __init__(self, class_names):
        self.class_names = class_names

    def update_and_draw(self, queue, rgb_buffer):
        while True:
            pkt = queue.tryGet()
            if pkt is None:
                break

            frame = rgb_buffer.get(pkt.getSequenceNum())
            if frame is None:
                continue

            h, w = frame.shape[:2]

            for d in pkt.detections:
                label = self.class_names[d.label]
                z_mm = int(d.spatialCoordinates.z)

                x1 = int(d.xmin * w)
                y1 = int(d.ymin * h)
                x2 = int(d.xmax * w)
                y2 = int(d.ymax * h)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} Z:{z_mm}mm",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

            cv2.imshow("Block Spatial Detection", frame)


# -------------------------------------------------
# Pipeline Creation
# -------------------------------------------------
def create_pipeline():
    pipeline = dai.Pipeline()

    # ---------------- Camera ----------------
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setPreviewSize(IMG_SIZE, IMG_SIZE)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    # ---------------- Stereo Depth ----------------
    monoL = pipeline.create(dai.node.MonoCamera)
    monoR = pipeline.create(dai.node.MonoCamera)
    monoL.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    monoR.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    monoL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoR.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(False)

    monoL.out.link(stereo.left)
    monoR.out.link(stereo.right)

    # ---------------- Hand YOLO (2D) ----------------
    hand_nn = pipeline.create(dai.node.YoloDetectionNetwork)
    hand_nn.setBlobPath(HAND_BLOB)
    hand_nn.setConfidenceThreshold(HAND_CONF)
    hand_nn.setIouThreshold(IOU_THRESH)
    hand_nn.setNumClasses(1)
    hand_nn.setCoordinateSize(4)

    cam.preview.link(hand_nn.input)

    # ---------------- Block YOLO (Spatial) ----------------
    block_nn = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
    block_nn.setBlobPath(BLOCK_BLOB)
    block_nn.setConfidenceThreshold(BLOCK_CONF)
    block_nn.setIouThreshold(IOU_THRESH)
    block_nn.setNumClasses(len(BLOCK_CLASSES))
    block_nn.setCoordinateSize(4)

    block_nn.setBoundingBoxScaleFactor(0.3)
    block_nn.setDepthLowerThreshold(DEPTH_MIN)
    block_nn.setDepthUpperThreshold(DEPTH_MAX)

    cam.preview.link(block_nn.input)
    stereo.depth.link(block_nn.inputDepth)

    # ---------------- Outputs ----------------
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam.preview.link(xout_rgb.input)

    xout_hand = pipeline.create(dai.node.XLinkOut)
    xout_hand.setStreamName("hand")
    hand_nn.out.link(xout_hand.input)

    xout_block = pipeline.create(dai.node.XLinkOut)
    xout_block.setStreamName("block")
    block_nn.out.link(xout_block.input)

    return pipeline

# -------------------------------------------------
# Main Orchestration
# -------------------------------------------------
def main():
    pipeline = create_pipeline()

    with dai.Device(pipeline) as device:
        q_rgb   = device.getOutputQueue("rgb",   maxSize=1, blocking=False)
        q_hand  = device.getOutputQueue("hand",  maxSize=1, blocking=False)
        q_block = device.getOutputQueue("block", maxSize=1, blocking=False)

        rgb_buffer = RGBBuffer(size=RGB_BUFFER_SIZE)
        hand = HandDetector()
        block = BlockDetector(BLOCK_CLASSES)

        while True:
            in_rgb = q_rgb.tryGet()
            if in_rgb is None:
                continue

            rgb_buffer.add(in_rgb)
            frame = in_rgb.getCvFrame()

            hand.update(q_hand)
            hand.draw(frame)

            block.update_and_draw(q_block, rgb_buffer)

            cv2.imshow("Hand Detection (Async)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == "__main__":
    main()
