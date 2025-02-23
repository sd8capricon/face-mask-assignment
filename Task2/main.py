import os

from face_detection_tracking import (
    load_video,
    save_video,
    make_detections,
    make_detections_with_tracking,
)

VIDEO_PATH = "test_videos"
OUT_PATH = "out"

""" 
Face detector algotihms seems to wrork great on front facing images 
but don't perform well on different face angles very well.
Between the Haarcascade, MTCNN and Caffe Model Caffe Model seemed to perform better and faster
"""

video_filenames = [
    "Test_video1.mp4",
    "Test_video2.mp4",
    "Test_video3.mp4",
]

if not os.path.exists("out/with_tracking"):
    os.makedirs("out/with_tracking")

for i, video_name in enumerate(video_filenames):

    print("Loading Video Now")
    save_as = f"output_video{i+1}.mp4"
    video = load_video(os.path.join(VIDEO_PATH, video_name))

    # Make detections without tracking
    print("Print Making Face Mask Detections")
    video_wo_tracking = make_detections(video, face_confidence=0.4)

    # save the video

    print("Print Making Face Mask Detections with Object Tracking")
    save_video(video_wo_tracking, os.path.join(OUT_PATH, save_as))

    video_w_tracking = make_detections_with_tracking(
        video, detection_interval=60, face_confidence=0.4
    )

    # save the tracked video
    save_video(video_w_tracking, os.path.join(OUT_PATH, f"with_tracking/{save_as}"))
