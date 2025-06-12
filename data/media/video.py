from __future__ import annotations

import cv2
import numpy as np
import torch
from transformers import VivitImageProcessor, VivitForVideoClassification

from .media import MediaPreProcessingPipeline, FileReferenceMediaCollector


class VideoCollector(FileReferenceMediaCollector):
    @staticmethod
    # TODO Load resources?
    def AMIGOS(processor: MediaPreProcessingPipeline):
        return VideoCollector(processor)

    def get_info(self, index: int) -> dict:
        file_path = self.media_collection.iloc[index]["raw"]
        video = cv2.VideoCapture(file_path)
        return super().get_info(index) | {
            'path': file_path,
            'height': video.get(cv2.CAP_PROP_FRAME_HEIGHT),
            'width': video.get(cv2.CAP_PROP_FRAME_WIDTH),
            'fps': video.get(cv2.CAP_PROP_FPS),
            'total_frames': video.get(cv2.CAP_PROP_FRAME_COUNT),
            'duration': video.get(cv2.CAP_PROP_FRAME_COUNT) / video.get(cv2.CAP_PROP_FPS)
        }


class VideoPreProcessingPipeline(MediaPreProcessingPipeline):
    image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
    video_model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")

    @staticmethod
    def get_frames(file: str, max_frames: int = 3000):
        video = cv2.VideoCapture(file)
        count: int = 0

        frames: list = []

        while video.isOpened() and count < max_frames:
            success, frame = video.read()
            if not success:
                break  # Video reading is done

            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            count += 1

        video.release()
        return frames

    @staticmethod
    def frame_resampling(x: np.ndarray, max_frame: int = 60):
        if len(x) > max_frame:
            return x[0:max_frame]
        diff = max_frame - len(x)
        return np.vstack((x, np.zeros((diff, x.shape[1], x.shape[2], x.shape[3]))))

    def process(self, media: str):
        video = np.array(VideoPreProcessingPipeline.get_frames(media))
        video = VideoPreProcessingPipeline.frame_resampling(video, 32)
        video = VideoPreProcessingPipeline.image_processor(list(video), return_tensors="pt")

        with torch.no_grad():
            video["pixel_values"] = video["pixel_values"].squeeze(1)
            return VideoPreProcessingPipeline.video_model(**video).logits.squeeze(0)
