import cv2
import numpy as np
import torch
from numpy import ndarray
from transformers import VivitImageProcessor, VivitForVideoClassification

from data.media.media import Media


class Video(Media):
    image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
    video_model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")

    def __init__(self, file_path: str, lazy: bool = True):
        self.frames, self.processed_frames = [], []
        super(Video, self).__init__(file_path, lazy)

    def get_info(self):
        video = cv2.VideoCapture(self.file_path)
        return {
            'path': self.file_path,
            'height': video.get(cv2.CAP_PROP_FRAME_HEIGHT),
            'width': video.get(cv2.CAP_PROP_FRAME_WIDTH),
            'fps': video.get(cv2.CAP_PROP_FPS),
            'total_frames': video.get(cv2.CAP_PROP_FRAME_COUNT),
            'duration': video.get(cv2.CAP_PROP_FRAME_COUNT) / video.get(cv2.CAP_PROP_FPS)
        }

    def _get_frames(self, max_frames: int = None):
        video = cv2.VideoCapture(self.file_path)
        count: int = 0
        while max_frames < 0 or count < max_frames:
            success, frame = video.read()
            if not success:
                break  # Video reading is done

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame.astype(np.uint8)
            count += 1

        video.release()

    def _inner_load(self, max_frames: int = 3000):
        # Load the frames from the video.
        self.frames = list(self._get_frames(max_frames))

    @staticmethod
    def frame_resampling(x: ndarray, max_frame: int = 60):
        if len(x) > max_frame:
            return x[0:max_frame]

        diff = max_frame - len(x)
        return np.vstack((x, np.zeros((diff, x.shape[1], x.shape[2], x.shape[3]))))

    def _inner_process(self, **kwargs):
        video = np.array(self.frames)
        video = Video.frame_resampling(video, 32)
        video = Video.image_processor(list(video), return_tensors="pt")

        with torch.no_grad():
            video["pixel_values"] = video["pixel_values"].squeeze(1)
            self.processed_frames = Video.video_model(**video).logits.squeeze(0)
