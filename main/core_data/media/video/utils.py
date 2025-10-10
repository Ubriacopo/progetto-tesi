import cv2
import numpy as np

from main.core_data.media.video import Video


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


def check_video_data(x, data_type: type):
    if not isinstance(x, Video):
        raise TypeError("Given object is not of required type Video")

    if x.data is None:
        raise ValueError("Video has to be loaded first.")

    if not isinstance(x.data, data_type):
        raise TypeError("Given video object is not valid")
