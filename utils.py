import os
import cv2
import numpy as np

IMG_PATH = "data/Medical Mask/images"

def get_image_path(image_name):
    return os.path.join(IMG_PATH, image_name)


def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.array(frames)


# Face Detection Model
modelFile = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "face_detector/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

def dnn_detect_faces(frame):
    h,w = frame.shape[:2]
    # blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 117.0, 123.0))
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    faces_bboxes = []
    # Loop over the detections and draw bounding boxes
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.3:
            # Get the coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            faces_bboxes.append((x1, y1, x2, y2))

    return faces_bboxes

