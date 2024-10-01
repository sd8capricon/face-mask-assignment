import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from utils import load_video, dnn_detect_faces

model = load_model('models/facemask_detector_model.keras')

video = load_video("data/test_video/Test_video1.mp4")
print(video.shape)
num_frames = video.shape[0]

def predict_mask(face_roi): 
    face_resized = cv2.resize(face_roi, (100, 100))
    face_resized = np.reshape(face_resized, (1,100,100,3))
    prediction = model.predict(face_resized)
    if prediction[0][0] > 0.5:
        return "face_with_mask"
    else:
        return "face_no_mask"


rgb_frame = video[120]
faces = dnn_detect_faces(rgb_frame)
if len(faces) == 0:
    print("No Faces Detected")
else:
    for (x1, y1, x2, y2) in faces:
        face_roi = rgb_frame[y1:y2, x1:x2]
        predicted_class = predict_mask(face_roi)
        print(predicted_class)
        cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(rgb_frame, f"{predicted_class}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

plt.imshow(rgb_frame)
plt.show()