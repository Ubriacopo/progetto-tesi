import torch
from torch import nn


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    Computes the contrastive loss for a given set of logits.

    The contrastive loss is calculated as the cross-entropy loss between the logits and a target tensor containing the sequence of indices from 0 to the length of the logits tensor.

    Args:
        logits (torch.Tensor): A 2D tensor containing the logits for the contrastive task.

    Returns:
        torch.Tensor: The contrastive loss.
    """
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    """
    Computes the contrastive loss for a given similarity matrix.

    The contrastive loss is calculated as the average of the cross-entropy loss for the rows (caption loss) and the columns (image loss) of the similarity matrix.

    Args:
        similarity (torch.Tensor): A 2D tensor containing the similarity scores between captions and images.

    Returns:
        torch.Tensor: The contrastive loss.
    """

    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


class ContrastiveModel(nn.Module):
    def _make_sequential(self, embedding_dim):
        return nn.Sequential(
            nn.Linear(embedding_dim, self.hidden_channels),
            nn.ReLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.ReLU(),
            nn.Linear(self.hidden_channels, self.out_channels)
        )

    def __init__(self, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.hidden_channels: int = hidden_channels
        self.out_channels: int = out_channels

        self.embedding_video = self._make_sequential(400)
        self.embedding_audio = self._make_sequential(768)
        self.embedding_text = self._make_sequential(768)

        logit_scale_init_value = 2.6592
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale_init_value))

    def forward(self, x_video, x_audio, x_text=None):
        x_video = self.embedding_video(x_video)
        x_video = nn.functional.normalize(x_video)

        if x_audio is not None:
            x_audio = self.embedding_audio(x_audio)
            x_audio = nn.functional.normalize(x_audio)

        if x_text is not None:
            x_text = self.embedding_text(x_text)
            x_text = nn.functional.normalize(x_text)

        logit_scale = self.logit_scale.exp()

        loss = 0.0
        if x_text is not None:
            logits_video_text = torch.matmul(x_video, x_text.t()) * logit_scale
            loss += clip_loss(logits_video_text)

        if x_audio is not None:
            logits_video_audio = torch.matmul(x_video, x_audio.t()) * logit_scale
            loss += clip_loss(logits_video_audio)

        if x_audio is not None and x_text is not None:
            logits_audio_text = torch.matmul(x_audio, x_text.t()) * logit_scale
            loss += clip_loss(logits_audio_text)

        return x_video, x_audio, x_text, loss
