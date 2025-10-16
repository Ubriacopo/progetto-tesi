from typing import Optional

import torch
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import nn, einsum

from main.model.layer.base import SimpleFeedForward


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
        self.dim_head: int = dim_head

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)
        # Linear projections to fit the attention shapes
        self.q = nn.Linear(dim, dim_head * heads, bias=False)
        self.k = nn.Linear(dim, dim_head * heads, bias=False)
        self.v = nn.Linear(dim, dim_head * heads, bias=False)
        # self.kv = nn.Linear(dim, dim_head * heads * 2, bias=False)
        self.out = nn.Linear(dim_head * heads, dim, bias=False)

    def forward(self, x: torch.Tensor, latents: torch.Tensor):
        """
        Args:
            x (torch.Tensor): x features
                shape (b, n1, D)
            latents (torch.Tensor): latent features
                shape (b, n2, D)
            mask (torch.Tensor): mask
        """

        b, Nx, D = x.shape
        _, Nz, _ = latents.shape

        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        q = self.q(latents)
        q = rearrange(q, 'b q (h d) -> b h q d', h=self.heads)
        assert q.shape == torch.Size([b, self.heads, Nz, self.dim_head])

        # kv = torch.cat((x, latents), dim=-2)
        kv_input = torch.cat((x, latents), dim=-2)
        k = self.k(kv_input)
        v = self.v(kv_input)

        # k, v = self.kv(x).chunk(2, dim=-1)
        k, v = rearrange_many((k, v), 'b f (h d) -> b h f d', h=self.heads)
        assert v.shape == torch.Size([b, self.heads, Nx + Nz, self.dim_head])

        q *= self.scale

        sim = einsum('b h q d, b h f d -> b h q f', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        alphas = sim.softmax(dim=-1)

        out = einsum('b h q f, b h f v -> b h q v', alphas, v)
        out = rearrange(out, 'b h q v -> b q (h v)')
        return self.out(out)


class PerceiverResampler(nn.Module):
    def __init__(self, dim: int, depth: int, dim_head: int = 64, heads: int = 8, num_latents: int = 64,
                 max_num_time_steps: int = None, max_num_frames: int = None, ff_mult: int = 4):
        """
        The PerceiverResampler comes from the Flamingo definition and enrichest the Perceiver architecture.
        It has a lower computational complexity and allows us to draw information from timed sequences.
        It condenses the info where possible.
        It is good to merge multimodal data later down the stream.

        :param dim: Size of the last dimension of the input (latent space).
        :param depth: Number of iterations of Perceiver Attention (nested PerceiverAttention + Projection Head).
        :param dim_head: Size of the heads of Perceiver Attention.
        :param heads: Number of attention heads.
        :param num_latents: Number of latent dimensions for the generation of latens that are passed to the attention.
        :param max_num_time_steps: To create Embeddings for the frames by tagging sequence positioning of media.
        :param max_num_frames: To create Embeddings for the frames by tagging sequence positioning of frames.
        :param ff_mult: Multiplier for the feed forward network to remap the input dim.
        """
        super().__init__()
        # We learn the latents
        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        # todo prova ad usare questie  fixare

        # todo
        # Set Transformer PMA (Pooling by Multihead Attention):
        # Replace learned queries with PMA seeds (k≈8–32). It’s still cross-attn,
        # but PMA tends to give better coverage/diversity, especially with a coverage loss. Works naturally with masks.

        # oppure
        # Perceiver IO (not just Resampler):
        # Keep a latent array (L≈128–256) with cross-attn in and cross-attn out.
        # Add relative time bias and mask-aware logits. It’s heavier but robust for ragged inputs.
        self.frame_embeddings: Optional[nn.Parameter] = None
        if max_num_frames is not None:
            self.frame_embeddings = nn.Parameter(torch.randn(max_num_frames, dim))

        self.time_pos_embeddings: Optional[nn.Parameter] = None
        if max_num_time_steps is not None:
            self.time_pos_embeddings = nn.Parameter(torch.randn(max_num_time_steps, 1, dim))

        # Build perceiver layers.
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                SimpleFeedForward(dim=dim, mult=ff_mult)
            ]))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask=None):
        """

        :param mask:
        :param x: Features of the modality in input. It is supposed to be a 5D tensor [b, T, F, v, D] with:
                    - b: is batch size of the input. (Fixed during training).
                    - T: is number of time steps.
                    - F: are the frames fed that compose a single time-step (For ViViT it's 16 for example as it has a temporal stride of 2)
                    - v: Are the generated patches by the embedder previously.
                    - D: Is the final embedding space (e.g. 768 for ViViT)

                    or 4D [b, T, Fv, D]
        :return:
        """

        b, T, F, v = x.shape[:4]

        if self.frame_embeddings is not None:
            # Add the frame embeddings to the input to not lose the temporal alignment information
            x += repeat(self.frame_embeddings[:F], "F d -> b T F v d", b=b, T=T, v=v)

        if x.dim() == 5:
            # Flatten the frame and spatial dimensions that we suppose are in [-3:-2]
            x = rearrange(x, "b T F v d -> b T (F v) d")

        if self.time_pos_embeddings is not None:
            # Add to the resampled x the time embeddings. Mostly won't be used as T tends to be 1.
            time_pos_emb = self.time_pos_embeddings[:T].unsqueeze(0).expand(b, -1, -1, -1)
            if mask is not None:
                time_pos_emb = time_pos_emb * mask.unsqueeze(-1).unsqueeze(-1)

            x += time_pos_emb

        # Flatten the frames
        x = rearrange(x, 'b T n d -> b (T n) d')
        latents = repeat(self.latents, "n d -> b n d", b=b)

        # Transformer Block
        for att, feed_forward in self.layers:
            latents = latents + att(x, latents)
            latents = latents + feed_forward(latents)

        return self.norm(latents)
