# Since the data is too big we reduce in size using quantization.
# Each float16 can be readapted as an int8 tensor + float16 scale 1d tensor
import torch

from main.utils.logging import make_logger


class Float16ToInt8Quantizer:
    def __init__(self):
        self.logger = make_logger(self.__class__.__name__)
        self.scale_range = (-127, 127)
        self.eps = 1e-8

        self.to_warn_on_fp32: bool = True

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dtype == torch.float32:
            self.to_warn_on_fp32 and self.logger.warn(
                "The passed tensor is a float32. We will still proceed conversion by halving."
                "So will be for all further input tensors. This message won't be displayed again."
            )

            self.to_warn_on_fp32 = False
            x = x.half()

        scales = x.abs().amax(dim=-1, keepdim=True) / float(self.scale_range[1])
        scales = scales.clamp_min(self.eps).to(torch.float16)

        quantized = (x / scales).round().clamp(self.scale_range[0], self.scale_range[1]).to(torch.int8)
        return quantized, scales

    def dequantize(self, t: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        return (t.half() * scales).half()

    def check_loss(self, og: torch.Tensor, new: torch.Tensor, scales: torch.Tensor) \
            -> tuple[torch.Tensor, torch.Tensor]:
        og = og.reshape(-1, og.shape[-1])  # To 2D
        mask = og.norm(dim=-1) > 0  # When norm not zero the row is valid (used)
        og = og[mask]

        og = torch.nn.functional.normalize(og, dim=-1)

        new = self.dequantize(new, scales)
        new = new.reshape(-1, new.shape[-1])[mask]
        new = torch.nn.functional.normalize(new, dim=-1)

        self_cos = (og * og).sum(dim=1)
        similarity = (og * new).sum(dim=1)

        self_similarity = self_cos.mean()
        if float(self_similarity.mean().item()) != 1.:
            self.logger.warning("[SANITY CHECK FAILED] self-cos-sim:" + str(self_similarity))

        if float(similarity.mean().item()) < 0.7:
            self.logger.warning("You loose some information on this sample be wary! "
                                "Similarity of:" + str(similarity.mean()))

        return (og * og).sum(dim=1), (og * new).sum(dim=1)
