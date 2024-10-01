from tensorflow.keras.models import load_model
from utils import load_video

model = load_model('models/facemask_detector_model.keras')

idk = load_video("data/test_video/Test_video1.mp4")

print(idk.shape)