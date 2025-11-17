#!/usr/bin/env python3
import cv2, numpy as np, depthai as dai
import blobconverter


# Define Frame Size: 640x360
FRAME_SIZE = (416, 416)

# Define Input size, name, and the zoo name from where to download the model 
# Using "face-detection-retail-0004" model from DepthAI model zoo!
# Note: If you define the path to the blob file directly, make sure the MODEL_NAME and ZOO_TYPE are None
DET_INPUT_SIZE = (300, 300)
model_name = "yolov5n_coco_416x416"
zoo_type = "depthai"
blob_path = None

#---------------------------------------------
# COCO labels (80 classes) for yolov5n_coco_416x416
# ---------------------------------------------
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

#------------------------------------------------
# Create Pipeline
#------------------------------------------------

# Define pipeline
pipeline = dai.Pipeline()

#------------------------------------------------
# Defining Sources (Cameras)
#------------------------------------------------

# Define RGB Camera 
cam = pipeline.createColorCamera()
cam.setPreviewSize(FRAME_SIZE[0], FRAME_SIZE[1]) # 416x416
cam.setInterleaved(False)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)

# Define Mono Camera Sources for Stereo Depth
#------------------------------------------------
# Left Camera (CAM_B)
mono_left = pipeline.createMonoCamera()
mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)

# Right Camera (CAM_C)
mono_right = pipeline.createMonoCamera()
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

# Create stereo node
stereo = pipeline.createStereoDepth()

# Linking mono cam outputs to stereo node
mono_left.out.link(stereo.left)
mono_right.out.link(stereo.right)

# Blobconverter
#------------------------------------------------
# blobconverter: Compiles & downloads model
# shaves: Variable determines # of SHAVE cores used to compile the NN. Higher value = faster NN can run.

if model_name is not None:
	blob_path = blobconverter.from_zoo(
	        name=model_name,
	        shaves=6,
		zoo_type=zoo_type # Depthai
		)

neural_network.setBlobPath(model_path)
