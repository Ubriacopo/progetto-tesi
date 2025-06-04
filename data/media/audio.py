import wave

import torch
import torchaudio
from torch import Tensor

from data.media.media import Media


class Audio(Media):

    # MFCC (Mel-Frequency Cepstral Coefficients) è un algoritmo utilizzato nella elaborazione
    # audio per estrarre caratteristiche significative dal segnale audio.

    def __init__(self, file_path: str, lazy: bool = True):
        self.audio_data: Tensor | None = None
        self.processed_audio_data: Tensor | None = None
        self.sample_rate: int | None = None
        super().__init__(file_path, lazy)

    def get_info(self):
        wave_object = wave.open(self.file_path)

        info = {
            "video_path": self.file_path,
            "num_channels": wave_object.getnchannels(),
            "sample_width": wave_object.getsampwidth(),
            "frame_rate": wave_object.getframerate(),
            "num_frames": wave_object.getnframes(),
        }

        wave_object.close()
        return info

    def _inner_load(self, **kwargs):
        self.audio_data, self.sample_rate = torchaudio.load(self.file_path)

    def _inner_process(self, **kwargs):
        # Meh, why just take the sample_rate?
        bundle = torchaudio.pipelines.HUBERT_BASE
        resampled = torchaudio.functional.resample(self.audio_data, self.sample_rate, bundle.sample_rate)

        with torch.no_grad():
            item, _ = bundle.get_model().extract_features(resampled)
            self.processed_audio_data = item[-1][0].mean(0)

    def get_num_channels(self) -> int | None:
        return self.audio_data.ndim if self.audio_data is not None else None
