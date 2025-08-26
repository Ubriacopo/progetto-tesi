import torch
import torchaudio.transforms as at
from tokenizers import Tokenizer
from torchvision.transforms import v2
from torchvision.transforms.v2 import ToTensor

from common.ds.dataset import AttentionRichObject
from common.ds.custom import CustomAudioTransforms, CustomVideoTransforms
from common.ds.transform import Compose, IDENTITY


def video_transform(means: tuple[float] | None = (0.485, 0.456, 0.406),
                    stds: tuple[float] | None = (0.229, 0.224, 0.225),
                    size: tuple[int, int] = (224, 224),
                    fps_map: tuple[int, int] = (30, 30)) -> Compose:
    return Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        # Just so that 256 -> 224 (We keep the proportion, no real reasoning behind it)
        v2.Resize((size[0] + int(.145 * size[0]), size[1] + int(.145 * size[1]))),
        v2.CenterCrop(size)
    ], [
        # todo augmentations have to be aligned between two models. or does it? MHMM
        #       Could I leverage the fact that samples are augmented to learn more?
        v2.RandomHorizontalFlip(),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        v2.RandomGrayscale(p=0.15),
    ], [
        # These two instructions go together
        v2.ToTensor(),
        CustomVideoTransforms.ResampleFps(fps_map) if fps_map[0] != fps_map[1] else IDENTITY,
        v2.Normalize(mean=means, std=stds) if means is not None and stds is not None else IDENTITY,
    ])


def audio_transform(frequency_mapping: tuple[int, int]) -> Compose:
    # https://www.kaggle.com/code/aisuko/audio-classification-with-hubert
    ogf, nwf = frequency_mapping
    return Compose([CustomAudioTransforms.ToMono(), ToTensor()], [], [
        at.Resample(orig_freq=ogf, new_freq=nwf) if ogf != nwf else IDENTITY
    ])


def text_transform(tokenizer=Tokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")) -> Compose:
    """

    :param tokenizer: This has to be the same as the one that will be used by the model downstream.
    :return: Ideally has to return a TextEntry
    """
    return Compose([], [], [
        v2.Lambda(lambda x: tokenizer.encode(x)),
        v2.Lambda(lambda x: AttentionRichObject(x.ids, x.attention_mask)),
    ])


# todo
# should pad and also add attention mask? Nope
# should re-order channels
# shoudl drop some channels
# to stft
def eeg_transform() -> Compose:
    # TODO: Work on this
    return Compose([ToTensor()], [], [])
