#!/usr/bin/env python3

import cv2
import depthai as dai
import time
from collections import queue

# ================= CONFIG =================
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
# ========================================

def create_pipeline():
    pipeline = dai.Pipeline()

    # ---------------- Color Camera ----------------
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

    # ==================================================
    # HAND YOLO (2D)
    # ==================================================
    hand_nn = pipeline.create(dai.node.YoloDetectionNetwork)
    hand_nn.setBlobPath(HAND_BLOB)
    hand_nn.setConfidenceThreshold(HAND_CONF)
    hand_nn.setIouThreshold(IOU_THRESH)
    hand_nn.setNumClasses(1)
    hand_nn.setCoordinateSize(4)
    
    hand_nn.setNumInferenceThreads(2)
    hand_nn.input.setBlocking(False) # Allow parallel inference
    hand_nn.input.setQueueSize(1) # Minimize queue size, avoid stale frame build-up

    cam.preview.link(hand_nn.input)

    # ==================================================
    # BLOCK YOLO (SPATIAL)
    # ==================================================
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


def main():
    pipeline = create_pipeline()

    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        q_hand = device.getOutputQueue("hand", maxSize=4, blocking=False)
        q_block = device.getOutputQueue("block", maxSize=1, blocking=False)
        
        rgb_buffer = {}
        rgb_buffer_size = 10


        last_hand = []
        last_hand_time = 0.0
        hand_max_age = 0.15 # 150 ms

        last_block = []
        block_updated = False

        while True:
            in_rgb = q_rgb.tryGet()
            if in_rgb is None:
                seq = in_rgb.getSequenceNum()
                rgb_buffer[seq] = in_rgb.getCvFrame()

                # Limit buffer size
                if len(rgb_buffer) > rgb_buffer_size:
                    oldest = min(rgb_buffer.keys())
                    del rgb_buffer[oldest]
                continue

            frame = in_rgb.getCvFrame()
            h, w = frame.shape[:2]

            # Drain hand detections
            while True:
                det = q_hand.tryGet()
                if det is None:
                    break
                last_hand = det.detections
                last_hand_time = time.time()

            # Drain block detections
            while True:
                det = q_block.tryGet()
                if det is None:
                    break

                block_seq = det.getSequenceNum()

                if block_seq in rgb_buffer:
                    frame = rgb_buffer[block_seq]
                    

                last_block = det.detections
                block_updated = True

            # Draw hand detections
            if time.time() - last_hand_time < hand_max_age:
                for d in last_hand:
                    x1 = int(d.xmin * w)
                    y1 = int(d.ymin * h)
                    x2 = int(d.xmax * w)
                    y2 = int(d.ymax * h)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                    cv2.putText(frame, "HAND", (x1,y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

            # Draw block detections + XYZ
            if block_updated:
                for d in last_block:
                    label = BLOCK_CLASSES[d.label] # Convert # label -> string
                    #conf = det.confidence
 
                    x1 = int(d.xmin * w)
                    y1 = int(d.ymin * h)
                    x2 = int(d.xmax * w)
                    y2 = int(d.ymax * h)

                    x_mm = int(d.spatialCoordinates.x)
                    y_mm = int(d.spatialCoordinates.y)
                    z_mm = int(d.spatialCoordinates.z)

                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(
                         frame,
                         f"{label} Z:{z_mm}mm",
                         (x1, y2+15),
                         cv2.FONT_HERSHEY_SIMPLEX,
                         0.5,
                         (0,255,0),
                         1
                    )

            cv2.imshow("Hand + Block Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == "__main__":
    main()
