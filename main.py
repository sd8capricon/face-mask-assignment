import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import load_video, make_detections, make_detection_with_tracking, save_video


video = load_video("data/test_video/Test_video1.mp4")
print(video.shape)
num_frames = video.shape[0]


# video = make_detections(video, face_confidence=0.4)

video = make_detection_with_tracking(video, face_confidence=0.4)

# save_video(video, "out/Test3.mp4")
