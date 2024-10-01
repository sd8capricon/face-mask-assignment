import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model


IMG_PATH = "data/Medical Mask/images"
model = load_model("models/facemask_detector_model.keras")


def get_image_path(image_name):
    return os.path.join(IMG_PATH, image_name)


# Face Detection Model
modelFile = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "face_detector/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)


def dnn_detect_faces(frame, confidence_threshold=0.5):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()
    faces_bboxes = []
    # Loop over the detections and draw bounding boxes
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > confidence_threshold:
            # Get the coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            faces_bboxes.append((x1, y1, x2, y2))

    return faces_bboxes


def detect_mask(face_roi):
    face_resized = cv2.resize(face_roi, (100, 100))
    face_resized = np.reshape(face_resized, (1, 100, 100, 3))
    prediction = model.predict(face_resized)
    if prediction[0][0] > 0.5:
        return "face_with_mask"
    else:
        return "face_no_mask"


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


def save_video(frames, output_path, fps=30):
    print("Saving your video")
    height, width, _ = frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)

    out.release()
    print(f"Video saved successfully to {output_path}")


# rgb_frame = video[120]
# faces = dnn_detect_faces(rgb_frame)
# if len(faces) == 0:
#     print("No Faces Detected")
# else:
#     for x1, y1, x2, y2 in faces:
#         face_roi = rgb_frame[y1:y2, x1:x2]
#         predicted_class = detect_mask(face_roi)
#         print(predicted_class)
#         cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(
#             rgb_frame,
#             f"{predicted_class}",
#             (x1, y1 - 10),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.5,
#             (0, 255, 0),
#             1,
#         )

# plt.imshow(rgb_frame)
# plt.show()


def make_detections(video, face_confidence):
    print("Making Detections")
    new_frames = []
    for rgb_frame in video:
        faces = dnn_detect_faces(rgb_frame, confidence_threshold=face_confidence)
        if len(faces) == 0:
            print("No Faces Detected")
        else:
            for x1, y1, x2, y2 in faces:
                face_roi = rgb_frame[y1:y2, x1:x2]
                predicted_class = detect_mask(face_roi)
                print(predicted_class)
                cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    rgb_frame,
                    f"{predicted_class}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
        new_frames.append(rgb_frame)
        cv2.imshow("frame", cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    print("Done")
    return np.array(new_frames)


def make_detection_with_tracking(video, face_confidence):
    new_frames = []
    trackers = []
    b_boxes = []

    for frame_index, rgb_frame in enumerate(video):
        # If no trackers/periocidically(every 30frames)
        if len(trackers) == 0 or frame_index % 60 == 0:
            faces = dnn_detect_faces(rgb_frame, confidence_threshold=face_confidence)
            for x1, y1, x2, y2 in faces:
                new_face = True

                for bbox in b_boxes:
                    (bx1, by1, bx2, by2) = bbox
                    # If face close to an already tracked face
                    if abs(bx1 - x1) < 30 and abs(by1 - y1) < 30:
                        new_face = False
                        break

                if new_face:
                    tracker = cv2.legacy.TrackerKCF_create()
                    trackers.append(tracker)
                    b_boxes.append((x1, y1, x2, y2))
                    tracker.init(rgb_frame, (x1, y1, x2, y2))

        # Update all existing trackers
        updated_boxes = []
        for _, tracker in enumerate(trackers):
            success, bbox = tracker.update(rgb_frame)
            if success:
                updated_boxes.append(bbox)
                (x1, y1, x2, y2) = [int(v) for v in bbox]
                cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                # Remove tracker if tracking fails
                updated_boxes.append(None)

        # Remove trackers for lost objects
        trackers = [t for i, t in enumerate(trackers) if updated_boxes[i] is not None]
        b_boxes = [b for b in updated_boxes if b is not None]

        new_frames.append(rgb_frame)

        # Display the frame
        cv2.imshow("Frame", cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    print("Done")
    return np.array(new_frames)
