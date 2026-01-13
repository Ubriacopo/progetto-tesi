# Since the data is too big we reduce in size using quantization.
# Each float16 can be readapted as an int8 tensor + float16 scale 1d tensor
import torch

from main.utils.logging import make_logger


class Float16ToInt8Quantization:
    def __init__(self):
        self.logger = make_logger(self.__class__.__name__)
        self.scale_range = (-127, 127)
        self.eps = 1e-8

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dtype == torch.float32:
            self.logger.warn("The passed tensor is a float32. We will still proceed conversion by halving.")
            x = x.half()

        scales = x.abs().amax(dim=-1, keepdim=True) / float(self.scale_range[1])
        scales = scales.clamp_min(self.eps).to(torch.float16)

        quantized = (x / scales).round().clamp(self.scale_range[0], self.scale_range[1]).to(torch.int8)
        return quantized, scales

    def dequantize(self, t: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        return (t.half() * scales).half()
