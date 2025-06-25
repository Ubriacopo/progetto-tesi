from __future__ import annotations

import cv2
import numpy as np
import torch
import torchaudio
from transformers import BertTokenizer, BertModel, VivitImageProcessor, VivitForVideoClassification

from common.data.preprocessing import MediaPreProcessingPipeline

# todo vedi che pipeline sono riciclabili
class TextProcessingPipeline(MediaPreProcessingPipeline):
    def process_output_shape(self) -> tuple:
        return ()  # todo

    @staticmethod
    def default() -> TextProcessingPipeline:
        return TextProcessingPipeline(
            BertTokenizer.from_pretrained("bert-base-uncased"),
            BertModel.from_pretrained("bert-base-uncased")
        )

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def process(self, media: list | torch.Tensor | str):
        embeddings = torch.tensor([self.tokenizer.encode(str(media))])
        with torch.no_grad():
            return self.model(embeddings).pooler_output.squeeze(0)


class AudioPreProcessingPipeline(MediaPreProcessingPipeline):
    def process_output_shape(self) -> tuple:
        pass

    def process(self, media: list | str) -> np.ndarray | torch.Tensor:
        audio_data, sample_rate = torchaudio.load(media)
        bundle = torchaudio.pipelines.HUBERT_BASE
        resampled = torchaudio.functional.resample(audio_data, sample_rate, bundle.sample_rate)

        with torch.no_grad():
            item, _ = bundle.get_model().extract_features(resampled)

        return item[-1][0].mean(0)


class SignalMediaPreProcessingPipeline(MediaPreProcessingPipeline):
    def process_output_shape(self) -> tuple:
        return (4,)

    def process(self, media: list | np.ndarray):
        pass


class VideoPreProcessingPipeline(MediaPreProcessingPipeline):
    image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
    video_model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")

    def process_output_shape(self) -> tuple:
        # todo
        return ()

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
