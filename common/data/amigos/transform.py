import torch
import torchaudio.transforms as at
from tokenizers import Tokenizer
from torch import nn
from torchvision.transforms import v2
from torchvision.transforms.v2 import ToTensor

from common.data.audio.transforms import ToMono
from common.data.transform import IDENTITY
from common.data.video import RegularFrameResampling


def train_video_transform(size: tuple[int, int] = (224, 224),
                          grayscale_p: float = 0.6,

                          max_frames: int = 32,

                          means: tuple[float] | None = (0.485, 0.456, 0.406),
                          stds: tuple[float] | None = (0.229, 0.224, 0.225), ) -> nn.Sequential:
    return nn.Sequential(
        v2.Resize(size),
        # Augmentation
        v2.RandomHorizontalFlip(),
        v2.RandomGrayscale(p=grayscale_p),

        RegularFrameResampling(max_frames),
        # Normalization
        v2.ToDtype(torch.float32, scale=True),
        # v2.Normalize(mean=means, std=stds) if means is not None and stds is not None else IDENTITY,
        # TODO: Call the embedder if needed here somewhere.
    )


# todo:
# frequenze audio anche queste poosso fare resampling prima di salvare?
def train_audio_transform(frequency_mapping: tuple[int, int] = (44100, 16000)) -> nn.Sequential:
    # https://www.kaggle.com/code/aisuko/audio-classification-with-hubert
    ogf, nwf = frequency_mapping

    return nn.Sequential(
        ToMono(),
        at.Resample(orig_freq=ogf, new_freq=nwf) if ogf != nwf else IDENTITY
    )


def text_transform(tokenizer=Tokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")) -> nn.Sequential:
    """

    :param tokenizer: This has to be the same as the one that will be used by the model downstream.
    :return: Ideally has to return a TextEntry
    """
    return nn.Sequential(
        v2.Lambda(lambda x: tokenizer.encode(x)),
    )


# todo
# should pad and also add attention mask? Nope
# should re-order channels
# shoudl drop some channels
# to stft
def train_eeg_transform() -> nn.Sequential:
    # TODO: Work on this
    return nn.Sequential(
        v2.Lambda(lambda x: x.to("cuda"))
    )
