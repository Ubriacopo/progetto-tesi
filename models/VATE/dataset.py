import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import ToTensor
from transformers import VivitImageProcessor, VivitForVideoClassification

from common.data.amigos.dataset import AMIGOSDataset
from common.data.video.transforms import ResampleVideoTensor
from common.data.transform import Compose, IDENTITY


# todo move?
class FrameResamplingNaive:
    def __init__(self, max_frames: int = 32, method: str | None = "pad"):
        self.max_frames: int = max_frames
        assert method in ["pad", None], "Given method must be either pad or none. Others are not supported."
        self.method: str | None = method

    def __call__(self, video: list[torch.Tensor] | torch.Tensor, **kwargs):
        if len(video) > self.max_frames:
            return video[:self.max_frames]

        if self.method == "pad":
            difference = self.max_frames - len(video)
            if isinstance(video, list):
                video = ToTensor()(video)
            t, x, y = video.shape  # Expect frames to be first channel
            video = torch.vstack((video, torch.zeros(difference, t, x, y)))
            return video

        print("Warning: Sequence length fixing method not used. Variable length is returned.")
        return video


class CallViViTModelAndProcessor:
    def __init__(self, model: str = "google/vivit-b-16x2-kinetics400"):
        self.processor = VivitImageProcessor.from_pretrained(model)
        self.model = VivitForVideoClassification.from_pretrained(model)

    def __call__(self, video: list[torch.Tensor] | torch.Tensor, **kwargs):
        if isinstance(video, torch.Tensor):
            video = [video]

        item = self.processor(video, return_tensors="pt")
        with torch.no_grad():
            item = self.model(**item).logits.squeeze(0)

        return item  # The item has been embedded.


class FaceCrop:
    # todo: Non necessario al momento poiche AMIGOS è già FACE ma servirà poi (se usiamo anche altri video che forse ha senso)
    def __call__(self, video, *args, **kwargs):
        pass


def video_transform(fps_map: tuple[int, int] = (30, 30), size: tuple[int, int] = (224, 224)) -> Compose:
    return Compose([
        v2.Resize(size),
    ], [], [
        ResampleVideoTensor(fps_map) if fps_map[0] != fps_map[1] else IDENTITY,
        FrameResamplingNaive(),
        CallViViTModelAndProcessor()
    ])


# todo
def audio_transform() -> Compose:
    return Compose([])


# todo
def text_transform() -> Compose:
    return Compose([])


def default_AMIGOS(base: str) -> AMIGOSDataset:
    return AMIGOSDataset(
        base,
        video_transform=video_transform(),
        audio_transform=audio_transform(),
        text_transform=text_transform(),
    )
