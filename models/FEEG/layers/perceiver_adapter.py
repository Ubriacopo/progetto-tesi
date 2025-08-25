from typing import Optional

import torch
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import nn, einsum

from models.FEEG.layers.base_embedding import FoundationEmbedder
from models.FEEG.layers.base_layers import SimpleFeedForward
from models.FEEG.layers.kd import KDHead


class PerceiverAttention(nn.Module):

    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8):
        """
        Perceiver Attention take from OpenFlamingo and adapted.

        :param dim: Input latent dimension
        :param dim_head: Attention head latent size
        :param heads: Number of attention heads
        """
        super().__init__()
        self.scale: float = dim_head ** -0.5
        self.heads: int = heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)
        # Linear projections to fit the attention shapes
        self.q = nn.Linear(dim, dim_head * heads, bias=False)
        self.kv = nn.Linear(dim, dim_head * heads * 2, bias=False)
        self.out = nn.Linear(dim_head * heads, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, n1, D)
            latents (torch.Tensor): latent features
                shape (b, T, n2, D)
        """
        x, latents = self.norm_media(x), self.norm_latents(latents)

        q = self.q(latents)
        kv = torch.cat((x, latents), dim=-2)
        k, v = self.kv(kv).chunk(2, dim=-1)

        q, k, v = rearrange_many((q, k, v), "b t n (h d) -> b h t n d", h=self.heads)
        q *= self.scale

        # Attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=self.heads)
        return self.out(out)


class PerceiverResampler(nn.Module):
    def __init__(self, dim: int, depth: int, dim_head: int = 64, heads: int = 8, num_latens: int = 64,
                 max_num_media: int = None, max_num_frames: int = None, ff_mult: int = 4):
        """
        The PerceiverResampler comes from the Flamingo definition and enrichest the Perceiver architecture.
        It has a lower computational complexity and allows us to draw information from timed sequences.
        It condenses the info where possible.
        It is good to merge multimodal data later down the stream.

        :param dim: Size of the last dimension of the input (latent space).
        :param depth: Number of iterations of Perceiver Attention (nested PerceiverAttention + Projection Head).
        :param dim_head: Size of the heads of Perceiver Attention.
        :param heads: Number of attention heads.
        :param num_latens: Number of latent dimensions for the generation of latens that are passed to the attention.
        :param max_num_media: To create Embeddings for the frames by tagging sequence positioning of media.
        :param max_num_frames: To create Embeddings for the frames by tagging sequence positioning of frames.
        :param ff_mult: Multiplier for the feed forward network to remap the input dim.
        """
        super().__init__()

        self.latents = nn.Parameter(torch.randn(num_latens, dim))

        self.frame_embeddings: Optional[nn.Parameter] = None
        if max_num_frames is not None:
            self.frame_embeddings = nn.Parameter(torch.randn(max_num_frames, dim))

        self.media_time_embeddings: Optional[nn.Parameter] = None
        if max_num_media is not None:
            self.media_time_embeddings = nn.Parameter(torch.randn(max_num_media, 1, dim))

        # Build perceiver layers.
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                SimpleFeedForward(dim=dim, mult=ff_mult)
            ]))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """

        :param x: Features of the modality in input. It is supposed to be a 5D tensor [b, T, F, v, D] with:
                    - b: is batch size of the input. (Fixed during training).
                    - T: is number of time steps.
                    - F: are the frames fed that compose a single time-step (For ViViT it's 16 for example as it has a temporal stride of 2)
                    - v: Are the generated patches by the embedder previously.
                    - D: Is the final embedding space (e.g. 768 for ViViT)
        :return:
        """
        b, T, F, v = x.shape[:4]

        if self.frame_embeddings is not None:
            # Add the frame embeddings to the input to not lose the temporal alignment information
            x += repeat(self.frame_embeddings[:F], "F d -> b T F v d", b=b, T=T, v=v)

        # Flatten the frame and spatial dimensions
        x = rearrange(x, "b T F v d -> b T (F v) d")

        if self.media_time_embeddings is not None:
            # Add to the resampled x the time embeddings. Mostly won't be used as T tends to be 1.
            x += self.media_time_embeddings[:T]

        latents = repeat(self.latents, "n d -> b T n d", b=b, T=T)

        for att, feed_forward in self.layers:
            latents = att(x, latents) + latents  # Residual network style.
            latents = feed_forward(latents) + latents  # Residual network style.

        return self.norm(latents)


class PerceiverAdapter(nn.Module):
    def __init__(self, embedder: FoundationEmbedder, resampler_depth: int,
                 target_shape: int, kd: bool = False, kd_size: int = None) -> None:
        """
        Adapts a modality to be ready for the GatedXAttention with other modalities.

        :param embedder: FoundationEmbedder object. This one generates the embeddings of the actual modality.
        :param resampler_depth: How many layers of perceiver attention + sequential are desired for the PerceiverResampler.
        :param target_shape: The target shape to map the embedding space of the modality.
        :param kd: If the current modality uses knowledge distillation
        :param kd_size: To what size to remap the output of the Resampler to match the teacher
        """
        super().__init__()
        self.resampler = PerceiverResampler(embedder.output_size, resampler_depth)
        self.kd_head: Optional[KDHead] = None

        if kd:
            # Build KD map
            assert kd_size is not None, "If using KD you should provide kd_size to map to teacher"
            self.kd_head = KDHead(input_dimension=embedder.output_size, output_dimension=kd_size)

        self.reshaper: Optional[nn.Linear] = None
        if target_shape != embedder.output_size:
            self.reshaper = nn.Linear(embedder.output_size, target_shape)

        self.embedder = embedder

    def forward(self, x, use_kd: bool = True) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None:
        """
        The processed modality in the adapted space.

        :param x: Complex object fit for the self FoundationEmbedder object.
        :param use_kd: If true the KD module is used (if the head was provided in initialization)
        :return:
        """
        if x is None:
            return None  # We can't work when modality is disabled

        e: torch.Tensor = self.embedder(**x)
        e = self.resampler(e)

        kd_e: Optional[torch.Tensor] = None
        if self.kd_head is not None:
            kd_e = rearrange(e, "b T n d -> b (T n) d")
            # Simple Global Average Pooling.
            # If we want we could introduce an attention pooling block (We have SelfAttentionPooling)
            # TODO: Try different configurations for this.
            kd_e = kd_e.mean(dim=1)
            kd_e = self.kd_head(kd_e)

        if self.reshaper is not None:
            e = self.reshaper(e)

        return e if not use_kd else (e, kd_e)
