#!/usr/bin/env python3

import depthai as dai
import cv2
from ultralytics import YOLO

# ---------------------------
# CONFIG
# ---------------------------

MODEL_PATH = "/home/robotics-3/runs/detect/train/weights/best.pt" # path to yolov8 model
N_FRAMES   = 300 # number of frames for test
CONF_THRES = 0.8 # confidence threshold
SHOW_WINDOW = True # False for headless testing

threshold_ok = True # Confidence Threshold >= 0.9

# Confidence threshold check: if < 0.9, fails
if CONF_THRES < 0.9:
    threshold_ok = False

# Map YOLO class IDs to class names
CLASS_ID_TO_NAME = {
    0: "Block_Blue",
    1: "Block_Green",
    2: "Block_Red",
    3: "End_Effector",
    4: "Robot_Arm",
}

ALL_CLASSES = [
    "Block_Blue",
    "Block_Green",
    "Block_Red",
    "End_Effector",
    "Robot_Arm"
]

# ---------------------------
# Load YOLOv8 model
# ---------------------------

model = YOLO(MODEL_PATH)

# ---------------------------
# Metrics: TP, FN, FP (class-level)
# ---------------------------
"""
  - TP: True Positive - each correct object detection
  - FN: False Negative - each time an object is present in the frame but the system
                         has failed to detect it or has misclassified the object.
  - FP: False Positive - NOT USED FOR RECALL!!! Each incorrectly classified object.
"""

# Counters for each TP, FN, and FP for each class. Ex: Robot_Arm = 3 FP, 2 FN, and 0 RP
TP = {c: 0 for c in ALL_CLASSES}
FN = {c: 0 for c in ALL_CLASSES}
FP = {c: 0 for c in ALL_CLASSES}

# Ground Truth: for each frame, all 5 objects are present
GT_CLASSES_PER_FRAME = set(ALL_CLASSES)

# ---------------------------
# Build DepthAI pipeline
# ---------------------------
# Only need RGB camera, since detections w/out depth are being made

# Build Pipeline
pipeline = dai.Pipeline()

# Create RGB Camera
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(1280, 720)
cam_rgb.setInterleaved(False) # Seperate color channels
cam_rgb.setFps(30)

# Link RGB -> Output Node
xout = pipeline.createXLinkOut()
xout.setStreamName("rgb")
cam_rgb.preview.link(xout.input)

# ---------------------------
# Run device + test loop
# ---------------------------
with dai.Device(pipeline) as device:
    # Queue size = 4, will store 4 frames
    # maxSize=4, blocking=False avoids app stalling if one stream lags; old frames drop instead
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    frame_count = 0 # Initialize Frame Tracker

    # Run until all frames have been processed
    while frame_count < N_FRAMES:
        in_rgb = q_rgb.get() # Get next frame packet from queue
        frame = in_rgb.getCvFrame() # Convert Depthai (OAK) frame to numpy BGR image

        # Run YOLOv8 model on the frame
        results = model.predict(
            source=frame,
            conf=CONF_THRES,
            verbose=False, # Do not print intenal logs each time
            device="cpu" # No GPUs on the computer
	)

        # Collect predicted classes for the frame
        pred_classes = []

        # For each bounding box in the frame
        for box in results[0].boxes:
            cls_id = int(box.cls[0]) # Get predicted class index (0-4)
            conf = float(box.conf[0]) # Get YOLOv8's confidence score

            # Map class index # -> class name
            cname = CLASS_ID_TO_NAME.get(cls_id, None)
            if cname is not None: # If class ID is known & valid
                pred_classes.append(cname) # Add to predicted classes for frame
            # else = unknown class, ignore

        # For recall: Only care about presence/absence per class in each frame.
        # Convert predictedd classes list -> set. Drops duplicates so missclassifications
        # are counted as FNs.
        pred_set = set(pred_classes)

        # Update TP and FN: For each class that should be present in frame, check if it was predicted
        for cname in ALL_CLASSES:
            if cname in GT_CLASSES_PER_FRAME:
                if cname in pred_set: # If object correctly detected
                    TP[cname] += 1
                else: # If object missing or misclassified
                    FN[cname] += 1

        # Update FP (for info only): predicted classes that are NOT in GT
        # GT = ALL_CLASSES every frame - catches extra classes
        for cname in pred_set:
            if cname not in GT_CLASSES_PER_FRAME:
                FP[cname] += 1

        # Show detections in window for observation while test is running
        if SHOW_WINDOW:
            annotated = results[0].plot() # Draw bounding boxes & labels on frame
            cv2.imshow("YOLO Detections", annotated) # Display annotations
            if cv2.waitKey(1) == ord('q'): # Press 'q' to quit
                break

        frame_count += 1 # Increment Frame Tracker
        print(f"Processed frame {frame_count}/{N_FRAMES}") # Track how many frames have been processed

# Once all frames processed, close window if the option is available.
if SHOW_WINDOW:
    cv2.destroyAllWindows()

# ---------------------------
# Compute Recall
# ---------------------------

print("\nRECALL RESULTS:")
print("------------------------------------------\n")


# Initialize TP & FN counts
overall_TP = 0
overall_FN = 0

# Read TP, FN, & FP for each class
for cname in ALL_CLASSES:
    tp = TP[cname]
    fn = FN[cname]
    fp = FP[cname]
    total_gt = tp + fn  # how many times class should have been detected

    # If # of total expected objects = valid
    # total_gt = (TP + FN)
    if total_gt > 0:
        recall = tp / total_gt # TP / (TP + FN)
        recall_pct = recall * 100.0 # Get percent recall
    else: # Invalid # of expected objects
        recall = float("nan")
        recall_pct = float("nan")

    # Add Class' TP & FN to overall total for all classes
    overall_TP += tp
    overall_FN += fn

    # Print Results
    print(f"{cname}:")
    print(f"    TP = {tp}, FN = {fn}, FP = {fp}, GT instances = {total_gt}")
    print(f"    Recall = {recall_pct:.2f}%")

# Compure Overall Recall across all Classes
if overall_TP + overall_FN > 0:
    overall_recall = overall_TP / (overall_TP + overall_FN)
    overall_recall_pct = overall_recall * 100.0 # Get percentage
else: # Invalid amount
    overall_recall = float("nan")
    overall_recall_pct = float("nan")

# Print Results
print("\nOverall:")
print(f"  Total TP = {overall_TP}, Total FN = {overall_FN}")
print(f"  Overall Recall = {overall_recall_pct:.2f}%")

# LIST PASS OR FAIL
# ---------------------------
# Pass/fail: 95% requirement & confidence threshold >= 0.9 (90%)

THRESHOLD = 0.95

if not threshold_ok:
    print(f"\nRESULT: FAIL (overall recall {overall_recall_pct:.2f}%, confidence threshold is < 0.9\n")
    print(f"Current Confidence Threshold: {CONF_THRES:.2f}\n")
elif overall_recall >= THRESHOLD:
    print(f"\nRESULT: PASS (overall recall {overall_recall_pct:.2f}% ≥ 95%)\n")
else:
    print(f"\nRESULT: FAIL (overall recall {overall_recall_pct:.2f}% < 95%)\n")
