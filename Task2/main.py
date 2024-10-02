import os

from face_detection_tracking import (
    load_video,
    save_video,
    make_detections,
    make_detections_with_tracking,
)

VIDEO_PATH = "data"
OUT_PATH = "out"

video_filenames = [
    "Test_video1.mp4",
    "Test_video2.mp4",
    "Test_video3.mp4",
]

if not os.path.exists("out/with_tracking"):
    os.makedirs("out/with_tracking")

for i, video_name in enumerate(video_filenames):

    print("Loading Video Now")
    save_as = f"output_video{i+1}"
    video = load_video(os.path.join(VIDEO_PATH, video_name))

    # Make detections without tracking
    print("Print Making Face Mask Detections")
    video_wo_tracking = make_detections(video, face_confidence=0.4)

    # save the video

    print("Print Making Face Mask Detections with Object Tracking")
    save_video(video_wo_tracking, os.path.join(OUT_PATH, video_name))

    video_w_tracking = make_detections_with_tracking(
        video, detection_interval=60, face_confidence=0.4
    )

    # save the tracked video
    save_video(
        video_w_tracking, os.path.join(OUT_PATH, f"with_tracking/output _video{i+1}")
    )
