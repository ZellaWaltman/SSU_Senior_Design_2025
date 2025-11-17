#!/usr/bin/env python3
import cv2
import numpy as np
import depthai as dai
import blobconverter
import time
from collections import Counter

# Define Frame Size: 416x416
FRAME_SIZE = (416, 416)

# Model Config
#- - - - - - - - - - - - - - - - - - - - - - - - - - -
DET_INPUT_SIZE = (416, 416)
model_name = "yolov5n_coco_416x416"
zoo_type = "depthai"

#----------------------------------------------------------------
# COCO labels (80 classes) for yolov5n_coco_416x416
# ---------------------------------------------------------------
label_map = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Target classes for assignment
TARGET_CLASSES = {"laptop", "cup", "apple", "bottle", "person"}

#-----------------------------------------------------------------------------------
# YOLOv5n detection network - Create & Configure node
#-----------------------------------------------------------------------------------
def create_yolo_node(pipeline,
                     model_name: str = model_name,
                     zoo_type: str = zoo_type,
                     conf_thr: float = 0.45,
                     iou_thr: float = 0.45):
    
    # Create YOLO detection network node
    yolo = pipeline.create(dai.node.YoloDetectionNetwork)

    # Download .blob file from Model Zoo - Blobconverter
	#- - - - - - - - - - - - - - - - - - - - - - - - - - -	
	# blobconverter: Compiles & downloads model
	# shaves: Variable determines # of SHAVE cores used to compile the NN. Higher value = faster NN can run.

    blob_path = blobconverter.from_zoo(
        name=model_name,
        shaves=6,
        zoo_type=zoo_type
    )
						 
    yolo.setBlobPath(blob_path)

    # Basic YOLO parameters
    yolo.setConfidenceThreshold(conf_thr)
    yolo.setIouThreshold(iou_thr)
    yolo.setNumClasses(80)
    yolo.setCoordinateSize(4)

    # YOLOv5 anchors for 416×416
    yolo.setAnchors([
        10,13, 16,30, 33,23,
        30,61, 62,45, 59,119,
        116,90, 156,198, 373,326
    ])
    yolo.setAnchorMasks({
        "side26": [0, 1, 2],
        "side13": [3, 4, 5],
        "side7":  [6, 7, 8],
    })

    yolo.setNumInferenceThreads(2)
    yolo.input.setBlocking(False)
    yolo.input.setQueueSize(1)

    return yolo
						 
#-----------------------------------------------------------------------------------
# Create Pipeline
#-----------------------------------------------------------------------------------
def create_pipeline():
	
	# Define pipeline
	pipeline = dai.Pipeline()
	
	#- - - - - - - - - - - - - - - - - - - - - - - - - - -	
	# Defining Sources (Cameras)
	#- - - - - - - - - - - - - - - - - - - - - - - - - - -
	# CAM_A = RGB, CAM_B = LEFT, CAM_C = RIGHT
	
	# Define RGB Camera
	#- - - - - - - - - - - - - - - - - - - - - - - - - - -
	cam_rgb = pipeline.createColorCamera() # Create RGB Cam
	cam_rgb.setPreviewSize(FRAME_SIZE[0], FRAME_SIZE[1]) # 416x416
	cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
	cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A) # Define real cam location on OAK-DW
	cam_rgb.setInterleaved(False) # Color Channels = Seperate
	cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
	
	# Define Mono Camera Sources for Stereo Depth
	#- - - - - - - - - - - - - - - - - - - - - - - - - - -
	# Left Mono Camera
	mono_left = pipeline.createMonoCamera()
	mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_1080_P)
	
	# Right Mono Camera
	mono_right = pipeline.createMonoCamera()
	mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_1080_P)
	
	# Board sockets: Define real cam location on the OAK-DW, creates XLinkIn node. XLinkIn: connection between device and computer
	mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
	mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
	
	# Create Stereo (Depth) Node
	#- - - - - - - - - - - - - - - - - - - - - - - - - - -
	stereo = pipeline.createStereoDepth()
	# Set High-Density Disparity Mode (DEFAULT)L Less holes & Better performance, slighly higher compute
	stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    stereo.setSubpixel(True) # subpixel interpolation of disparity, up to 1/32 pixel precision
	# left-right consistency checking, rejects pixels where left->right and right->left disagree
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False) # No extended disparity (would reduce accuracy for distances > 2m)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB) # Align depth w/ RGB cam's FOV & pixel grid
	
	# Linking mono cam outputs to stereo node
	mono_left.out.link(stereo.left)
	mono_right.out.link(stereo.right)

	# Create Yolo Node
	#- - - - - - - - - - - - - - - - - - - - - - - - - - -
    yolo = create_yolo_node(pipeline)

    # Link RGB cam -> YOLO
    cam.preview.link(yolo.input)

	# XLink outputs to host - OUTPUT STREAMS
	#- - - - - - - - - - - - - - - - - - - - - - - - - - -
	# Create Preview Output Stream (From Camera)

	xout_rgb = pipeline.create(dai.node.XLinkOut)
	xout_rgb.setStreamName("rgb")
	cam.preview.link(xout_rgb.input)
	
	xout_det = pipeline.create(dai.node.XLinkOut)
	xout_det.setStreamName("detections")
	yolo.out.link(xout_det.input)
	
	xout_depth = pipeline.create(dai.node.XLinkOut)
	xout_depth.setStreamName("depth")
	stereo.depth.link(xout_depth.input)

	return pipeline

#-----------------------------------------------------------------------------------
# Helper: depth measurement via 7x7 ROI (Region of Interest) median
#-----------------------------------------------------------------------------------
def measure_distance(depth_frame_mm, cx, cy, roi_size=7,
                     min_valid_mm=300, max_valid_mm=15000):
	"""
	 - depth_frame_mm: 2D array w/ depth values in mm
	 - cx, cy: center pixel to measure around
	 - roi_size=7: 7×7 region
	 - min_valid_mm=300: ignore depth < 30 cm
	 - max_valid_mm=15000: ignore depth > 15 m
	"""

	# Get image size (height, width)
 	# .shape[:2] = numpy array [h, w, color channels], get only h & w
    h, w = depth_frame_mm.shape[:2]

	# Get RoI Boundaries
    half = roi_size // 2 # Get half of RoI size ( 7 // 2 = 3)
    x1 = max(0, cx - half)
    x2 = min(w - 1, cx + half)
    y1 = max(0, cy - half)
    y2 = min(h - 1, cy + half)

	# Extract RoI
    roi = depth_frame_mm[y1:y2+1, x1:x2+1].astype(np.float32)

    # Filter out 0 & invalid values
    roi = roi[(roi > 0) & (roi >= min_valid_mm) & (roi <= max_valid_mm)]
    if roi.size == 0:
        return None

	# return median depth of RoI
    median_mm = np.median(roi)
    return float(median_mm)

#-----------------------------------------------------------------------------------
# Main
#-----------------------------------------------------------------------------------
# Start Pipeline
# Acquire Video Frames from "preview" queue and get the NN outputs (detections and bounding box mapping) from the "det_out" queue.
# Acquire outputs -> display spacial info & bounding box on the image frame.

# dai.Device(pipeline) as device: transfers code onto the device (camera) from the host (computer)
with dai.Device(pipeline) as device:
	# Queue size = 4, will store 4 frames
	# maxSize=4, blocking=False avoids app stalling if one stream lags; old frames drop instead
    qRgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue("detections", maxSize=4, blocking=False)
    qDepth = device.getOutputQueue("depth", maxSize=4, blocking=False)

	# Statistics Setup (FPS & Class Counts)
    frame_count = 0
    start_time = time.time()
    last_report_time = start_time
    class_counts = Counter()

	#- - - - - - - - - - - - - - - - - - - - - - - - - - -
	# Main Loop
	#- - - - - - - - - - - - - - - - - - - - - - - - - - -
    while True:
		# Get next item from each queue
        inRgb = qRgb.get()
        inDet = qDet.get()
        inDepth = qDepth.get()

		# Convert to numpy arrays
        frame = inRgb.getCvFrame() # Color image (BGR)
        depthFrame = inDepth.getFrame() # Depth image (uint16, mm)

		# Get sizes
        h_rgb, w_rgb = frame.shape[:2]
        h_depth, w_depth = depthFrame.shape[:2]

        # FPS
        frame_count += 1
        now = time.time()
        elapsed = now - start_time # time elapsed
        fps = frame_count / elapsed if elapsed > 0 else 0.0

		# Draw Bounding Boxes for Detections in Target Classes
		#- - - - - - - - - - - - - - - - - - - - - - - - - - -
        detections = inDet.detections

        for det in detections:
            if det.label >= len(label_map):
                continue
            label = label_map[det.label]

            # Filter: Target Classes ONLY
            if label not in TARGET_CLASSES:
                continue

            # Bounding box in RGB frame (Pixel Coords)
            x1 = int(det.xmin * w_rgb)
            y1 = int(det.ymin * h_rgb)
            x2 = int(det.xmax * w_rgb)
            y2 = int(det.ymax * h_rgb)

            # Clamp Pixel Coords to valid positions (dont go outside image)
            x1 = max(0, min(x1, w_rgb - 1))
            x2 = max(0, min(x2, w_rgb - 1))
            y1 = max(0, min(y1, h_rgb - 1))
            y2 = max(0, min(y2, h_rgb - 1))

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Find center point of box in RGB coords
            cx_rgb = int((x1 + x2) / 2)
            cy_rgb = int((y1 + y2) / 2)

            # Map center point to depth frame coords
            cx_depth = int(cx_rgb * w_depth / w_rgb)
            cy_depth = int(cy_rgb * h_depth / h_rgb)

            # Measure depth around center (7x7 RoI median)
            distance_mm = measure_distance(depthFrame, cx_depth, cy_depth)

            # Draw crosshair at measurement point (on RGB frame)
            crosshair_size = 5
            cv2.line(frame,
                     (cx_rgb - crosshair_size, cy_rgb),
                     (cx_rgb + crosshair_size, cy_rgb),
                     (0, 255, 255), 1)
            cv2.line(frame,
                     (cx_rgb, cy_rgb - crosshair_size),
                     (cx_rgb, cy_rgb + crosshair_size),
                     (0, 255, 255), 1)

			# Displaying Labels
			#- - - - - - - - - - - - - - - - - - - - - - - - - - -
            if distance_mm is None:
                dist_str = "dist: N/A"
            else:
                if distance_mm < 1000.0:
                    dist_str = f"dist: {distance_mm / 10.0:.0f} cm"  # mm -> cm
                else:
                    dist_str = f"dist: {distance_mm / 1000.0:.2f} m"  # mm -> m

            # Display Label & Confidence Scores
            conf = det.confidence
            text = f"{label} {conf*100:.1f}% {dist_str}"

            cv2.putText(frame, text, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Update class counts for performance stats
            class_counts[label] += 1

        # Display FPS on screen
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

		# Show RGB & Depth Windows
        cv2.imshow("OAK-D YOLOv5n + Depth", frame)

        # Visualize Depth w/ RGB map
        depth_vis = cv2.normalize(depthFrame, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = depth_vis.astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        cv2.imshow("Depth", depth_vis)

        # Print stats every 5 seconds
        if now - last_report_time > 5.0:
            avg_fps = frame_count / (now - start_time)
            print("----- Stats -----")
            print(f"Average FPS: {avg_fps:.2f}")
            print(f"Detection counts: {dict(class_counts)}")
            last_report_time = now

		# Exit by pressing 'q'
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
