import numpy as np
from data_loader import extract_frames
from tensorflow.keras.models import load_model

model = load_model('saved_model/deepfake_detection_model.h5')

def predict_video(video_path):
    frames = extract_frames(video_path)
    frames = np.array(frames) / 255.0
    if len(frames) == 0:
        return "No frames to predict."

    predictions = [model.predict(np.expand_dims(frame, axis=0))[0][0] for frame in frames]
    avg_pred = np.mean(predictions)
    return "FAKE" if avg_pred > 0.5 else "REAL"