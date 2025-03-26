import os
import cv2
import json5
import numpy as np
from config import fake_path, real_path, metadata_path, IMG_SIZE, FRAMES_PER_VIDEO, LIMITED_VIDEOS

def extract_frames(video_path, num_frames=FRAMES_PER_VIDEO):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frames.append(frame)
    cap.release()
    return frames

def load_dataset():
    x, y = [], []
    real_videos = os.listdir(real_path)[:LIMITED_VIDEOS * 2]
    fake_videos = os.listdir(fake_path)[:LIMITED_VIDEOS * 2]

    for video in real_videos:
        frames = extract_frames(os.path.join(real_path, video))
        x.extend(frames)
        y.extend([0] * len(frames))
    
    for video in fake_videos:
        frames = extract_frames(os.path.join(fake_path, video))
        x.extend(frames)
        y.extend([1] * len(frames))

    return np.array(x) / 255.0, np.array(y)