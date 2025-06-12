import torchaudio
import torch

from .media import FileReferenceMediaCollector, MediaPreProcessingPipeline


class AudioCollector(FileReferenceMediaCollector):
    @staticmethod
    def AMIGOS(processor: MediaPreProcessingPipeline):
        return AudioCollector(processor)


class AudioPreProcessingPipeline(MediaPreProcessingPipeline):
    def process(self, media: list | str):
        audio_data, sample_rate = torchaudio.load(media)
        bundle = torchaudio.pipelines.HUBERT_BASE
        resampled = torchaudio.functional.resample(audio_data, sample_rate, bundle.sample_rate)

        with torch.no_grad():
            item, _ = bundle.get_model().extract_features(resampled)

        return item[-1][0].mean(0)
