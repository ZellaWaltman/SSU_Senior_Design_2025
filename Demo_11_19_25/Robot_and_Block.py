#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import time
from ultralytics import YOLO

# Config
#------------------------------------------------------------------------------------------
MODEL_PATH = "/home/robotics-3/runs/detect/train/weights/best.pt"

IMG_SIZE = 640
CONF_THRESH = 0.4

CLASS_NAMES = ["Block_Blue", "Block_Green", "Block_Red", "End_Effector", "Robot_Arm"]
#------------------------------------------------------------------------------------------

# Build DepthAI pipeline (RGB + Depth only, NO NN on device)
#------------------------------------------------------------------------------------------
def create_pipeline():
    pipeline = dai.Pipeline()

    # RGB Camera
    # - - - - - - - - - - - - - -
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setPreviewSize(IMG_SIZE, IMG_SIZE)  # 640x640 for YOLO
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    # Mono cameras for depth
    # - - - - - - - - - - - - - -
    monoL = pipeline.createMonoCamera()
    monoR = pipeline.createMonoCamera()
    monoL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoR.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoL.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    monoR.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    # Stereo depth node
    # - - - - - - - - - - - - - -
    stereo = pipeline.createStereoDepth()
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    stereo.setSubpixel(True)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # align depth to RGB camera

    monoL.out.link(stereo.left)
    monoR.out.link(stereo.right)

    # XLink outputs to host
    # - - - - - - - - - - - - - -
    xout_rgb = pipeline.createXLinkOut()
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    xout_depth = pipeline.createXLinkOut()
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    return pipeline

# Main
#------------------------------------------------------------------------------------------
def main():
    # Load YOLOv8 model on host
    print("Loading YOLO model on host...")
    model = YOLO(MODEL_PATH)

    pipeline = create_pipeline()

    with dai.Device(pipeline) as device:
        q_rgb   = device.getOutputQueue("rgb",   maxSize=4, blocking=False)
        q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)

        last_fps_time = time.time()
        frame_count = 0
        fps = 0.0

        while True:
            in_rgb   = q_rgb.get()      # ImgFrame
            in_depth = q_depth.get()    # ImgFrame (depth)

            frame = in_rgb.getCvFrame()    # 640x640 BGR
            depth = in_depth.getFrame()    # depth in millimeters, aligned to RGB

            # Run YOLOv8 on host (CPU by default)
            results = model(frame, imgsz=IMG_SIZE, device="cpu", verbose=False)
            r = results[0]

            # r.boxes: each box has .xyxy, .conf, .cls
            detections = []
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < CONF_THRESH:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1 = int(max(0, min(IMG_SIZE - 1, x1)))
                y1 = int(max(0, min(IMG_SIZE - 1, y1)))
                x2 = int(max(0, min(IMG_SIZE - 1, x2)))
                y2 = int(max(0, min(IMG_SIZE - 1, y2)))

                cls_id = int(box.cls[0])

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "conf": conf,
                    "cls":  cls_id,
                })

            # Optional: keep only best detection per class
            best_by_class = {}
            for det in detections:
                c = det["cls"]
                if c not in best_by_class or det["conf"] > best_by_class[c]["conf"]:
                    best_by_class[c] = det
            detections = list(best_by_class.values())

            # Draw detections + depth
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                conf = det["conf"]
                cls_id = det["cls"]
                label = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else str(cls_id)

                # Depth inside bbox (median of valid pixels)
                roi = depth[y1:y2, x1:x2]
                z_m = 0.0
                if roi.size > 0:
                    valid = roi[roi > 0]
                    if valid.size > 0:
                        z_m = float(np.median(valid)) / 1000.0  # mm -> m

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label} {conf:.2f} {z_m:.2f}m"
                cv2.putText(
                    frame,
                    text,
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA
                )

            # FPS
            frame_count += 1
            now = time.time()
            if now - last_fps_time >= 1.0:
                fps = frame_count / (now - last_fps_time)
                last_fps_time = now
                frame_count = 0

            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("YOLOv8 on Host + Depth (OAK-D-W)", frame)
            if cv2.waitKey(1) == ord('q'):
                break


if __name__ == "__main__":
    main()
