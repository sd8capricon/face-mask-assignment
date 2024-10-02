import os

from face_detection_tracking import (
    load_video,
    save_video,
    make_detections,
    make_detections_with_tracking,
)

VIDEO_PATH = "data/test_video"
OUT_PATH = "out"

video_filenames = [
    "Test_video1.mp4",
    "Test_video2.mp4",
    "Test_video3.mp4",
]

for video_name in video_filenames:
    video = load_video(os.path.join(VIDEO_PATH, video_name))

    # Make detections without tracking
    video_wo_tracking = make_detections(video, face_confidence=0.4)

    # save the video
    save_video(video_wo_tracking, os.path.join(OUT_PATH, video_name))

    video_w_tracking = make_detections_with_tracking(
        video, detection_interval=60, face_confidence=0.4
    )

    # save the tracked video
    save_video(video_w_tracking, os.path.join(OUT_PATH, f"with_tracking/{video_name}"))
