import cv2
import numpy as np


def resize_frames(frames: list, size: tuple = (224, 224)):
    resized = [cv2.resize(frame, size, interpolation=cv2.INTER_AREA) for frame in frames]
    return np.array(resized)

def extract_frames(video: cv2.VideoCapture) -> list:
    frames = []
    while True:
        r, frame = video.read()
        if not r: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return frames


def downsample_video(video: cv2.VideoCapture, fps: int) -> list:
    # Fix the fps
    interval = int(round(video.get(cv2.CAP_PROP_FPS) / fps))

    frames, idx = [], 0
    while True:
        r, frame = video.read()
        if not r: break

        if idx % interval == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1

    video.release()
    return frames
