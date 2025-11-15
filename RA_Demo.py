#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import time

# Config
#------------------------------------------------------------------------------------------
BLOB_PATH = "/home/robotics-3/.cache/blobconverter/best_openvino_2022.1_5shave.blob"

IMG_SIZE = 640
CONF_THRESH = 0.4
IOU_THRESH = 0.5

MAX_DETS = 20

CLASS_NAMES = ["End_Effector", "Robot_Arm"]
#------------------------------------------------------------------------------------------

# NMS + IOU
#------------------------------------------------------------------------------------------
def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def nms(boxes, scores, iou_thres):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = np.array([box_iou(boxes[i], boxes[j]) for j in order[1:]])
        inds = np.where(ious <= iou_thres)[0]
        order = order[inds + 1]

    return keep
    
#------------------------------------------------------------------------------------------
# YOLOv8 Decoder
#------------------------------------------------------------------------------------------
def decode_yolov8(output_flat, img_w, img_h):
    """
    ONNX output shape: [1, 6, 8400]
    Flattened length: 6 * 8400 = 50400

    Layout per anchor i:
        [cx_px, cy_px, w_px, h_px, logit_cls0, logit_cls1]
    where:
        - coordinates are already in pixel units
        - last 2 values are logits (need sigmoid)
    """
    # (1, 6, 8400) -> (6, 8400) -> (8400, 6)
    data = np.array(output_flat, dtype=np.float32).reshape(6, -1).transpose(1, 0)

    # Split channels
    cx = data[:, 0]
    cy = data[:, 1]
    w  = data[:, 2]
    h  = data[:, 3]
    logits = data[:, 4:]             # shape (8400, 2)

    # Class probabilities (single sigmoid, not softmax)
    probs = 1.0 / (1.0 + np.exp(-logits))

    # Best class and its score
    cls_ids   = np.argmax(probs, axis=1)
    cls_confs = probs[np.arange(len(probs)), cls_ids]

    # 1) Confidence filter
    mask = cls_confs > CONF_THRESH
    if not np.any(mask):
        return []

    indices = np.where(mask)[0]

    # 2) Keep only top-K by confidence to avoid NMS over 8400 boxes
    if indices.size > MAX_DETS:
        indices = indices[np.argsort(cls_confs[indices])[-MAX_DETS:]]

    cx = cx[indices]
    cy = cy[indices]
    w  = w[indices]
    h  = h[indices]
    cls_ids   = cls_ids[indices]
    cls_confs = cls_confs[indices]

    # 3) Convert pixel cx,cy,w,h -> xyxy
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # 4) Clip to image bounds
    x1 = np.clip(x1, 0, img_w - 1)
    y1 = np.clip(y1, 0, img_h - 1)
    x2 = np.clip(x2, 0, img_w - 1)
    y2 = np.clip(y2, 0, img_h - 1)

    boxes = np.stack([x1, y1, x2, y2], axis=1)

    # 5) NMS
    keep = nms(boxes.tolist(), cls_confs.tolist(), IOU_THRESH)

    detections = []
    for i in keep:
        detections.append({
            "bbox": (
                int(boxes[i][0]),
                int(boxes[i][1]),
                int(boxes[i][2]),
                int(boxes[i][3]),
            ),
            "conf": float(cls_confs[i]),
            "cls":  int(cls_ids[i]),
        })
    return detections

# Build DepthAI pipeline (DepthAI 2.x API)
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

    # Neural network node
    # - - - - - - - - - - - - - -
    nn = pipeline.createNeuralNetwork()
    nn.setBlobPath(BLOB_PATH)
    cam_rgb.preview.link(nn.input)

    # XLink outputs to host
    # - - - - - - - - - - - - - -
    xout_rgb = pipeline.createXLinkOut()
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    xout_nn = pipeline.createXLinkOut()
    xout_nn.setStreamName("nn")
    nn.out.link(xout_nn.input)

    xout_depth = pipeline.createXLinkOut()
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    return pipeline

# Main
#------------------------------------------------------------------------------------------
def main():
    pipeline = create_pipeline()

    with dai.Device(pipeline) as device:
        q_rgb   = device.getOutputQueue("rgb",   maxSize=4, blocking=False)
        q_nn    = device.getOutputQueue("nn",    maxSize=4, blocking=False)
        q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)

        last_fps_time = time.time()
        frame_count = 0
        fps = 0.0

        while True:
            in_rgb   = q_rgb.get()      # ImgFrame
            in_nn    = q_nn.get()       # NNData
            in_depth = q_depth.get()    # ImgFrame (depth)

            frame = in_rgb.getCvFrame()   # 640x640 BGR
            depth = in_depth.getFrame()   # depth in millimeters (aligned to RGB)

            out_flat = in_nn.getFirstLayerFp16()
            detections = decode_yolov8(out_flat, frame.shape[1], frame.shape[0])

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

            frame_count += 1
            now = time.time()
            if now - last_fps_time >= 1.0:
                fps = frame_count / (now - last_fps_time)
                last_fps_time = now
                frame_count = 0

            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("YOLOv8 + Depth (OAK-D-W, DepthAI 2.x)", frame)
            if cv2.waitKey(1) == ord('q'):
                break

if __name__ == "__main__":
    main()
