import pytest
import torch
from torch import nn
from torch.nn import functional as F

from main.core_data.quantization import Float16ToInt8Quantizer
from main.model.blocks.dropout import BernoulliSupportsModalityDropout
from main.model.blocks.kd import KDHead
from main.model.blocks.modality_stream import ModalityStream
from main.model.blocks.xattention import GatedXAttentionBlock, MaskedCrossAttention, GatedXAttentionFactory, GatedXAttentionCustomArgs
from main.model.blocks.time_masked import TimeMaskSwitchableProperties
from main.model.loss import SiglipLoss
from main.model.neegavi.model import EegInterAviModel, EegInterAviModelConfiguration


class Adapter(nn.Module):
    def forward(self, data, mask=None):
        return {"data": data, "mask": mask}


def tiny_model(mode="causal"):
    return EegInterAviModel(
        EegInterAviModelConfiguration(
            pivot_dim=4, support_dim=4, output_size=4,
            modality=TimeMaskSwitchableProperties(mode),
        ),
        ModalityStream("eeg", 4, Adapter(), 1),
        ModalityStream("vid", 4, Adapter(), 1),
        attn_blocks=[GatedXAttentionBlock(4, 4, dim_head=2, heads=2) for _ in range(2)],
    )


def test_quantization_has_positive_scales_for_padding_and_small_values():
    quantizer = Float16ToInt8Quantizer()
    x = torch.tensor([[0., 0.], [1e-6, -1e-6]], dtype=torch.float16)
    q, scales = quantizer.quantize(x)
    assert (scales > 0).all()
    assert q[0].eq(0).all()
    assert q[1].ne(0).all()
    assert torch.isfinite(quantizer.dequantize(q, scales)).all()


def test_cross_attention_matches_single_scaled_attention():
    torch.manual_seed(1)
    block = MaskedCrossAttention(4, 4, dim_head=4, heads=1)
    q, kv = torch.randn(2, 3, 4), torch.randn(2, 5, 4)
    k, v = block.kv(kv).chunk(2, dim=-1)
    expected = block.out(F.scaled_dot_product_attention(block.q(q)[:, None], k[:, None], v[:, None])[:, 0])
    torch.testing.assert_close(block(q, kv), expected)


def test_causal_cross_attention_allows_past_and_excludes_future():
    model = tiny_model()
    times = torch.arange(3)[None]
    assert torch.equal(model.build_allow_mask(times, times)[0], torch.ones(3, 3, dtype=torch.bool).tril())


def test_window_attention_respects_lookback_and_lookahead():
    model = tiny_model()
    model.set_attention_modality(TimeMaskSwitchableProperties("window", lookback=1, lookahead=0))
    times = torch.arange(4)[None]
    expected = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=torch.bool)
    assert torch.equal(model.build_allow_mask(times, times)[0], expected)


@pytest.mark.parametrize("mode", ["causal", "bidirectional"])
def test_future_inputs_cannot_change_earlier_causal_embeddings(mode):
    torch.manual_seed(2)
    model = tiny_model(mode).eval()
    eeg, vid = torch.randn(1, 3, 4), torch.randn(1, 3, 1, 4)
    batch = {"eeg": {"data": eeg, "mask": torch.ones(1, 3, dtype=torch.bool)},
             "vid": {"data": vid, "mask": torch.ones(1, 3, dtype=torch.bool)}}
    before = model(batch).embeddings["data"][:, 0].detach().clone()
    eeg[:, 1:] = torch.randn_like(eeg[:, 1:]) * 5
    vid[:, 1:] = torch.randn_like(vid[:, 1:]) * 5
    after = model(batch).embeddings["data"][:, 0]
    if mode == "causal":
        torch.testing.assert_close(before, after)
    else:
        assert not torch.allclose(before, after)


def test_support_processing_preserves_batch_alignment_after_dropout():
    model = tiny_model()
    support = {"data": torch.randn(3, 2, 1, 4), "mask": torch.ones(3, 2, dtype=torch.bool)}
    result = model.process_support(support, torch.tensor([0, 2]), model.supports[0], False, model.init_output("cpu"))
    assert result[1].tolist() == [[True, True], [False, False], [True, True]]
    assert result[0][1].eq(0).all()


def test_missing_masks_and_eeg_only_forward_are_supported():
    model = tiny_model().eval()
    batch = {"eeg": {"data": torch.randn(2, 3, 4)}}
    assert torch.isfinite(model(batch).cls).all()
    batch["vid"] = {"data": torch.randn(2, 3, 1, 4)}
    assert torch.isfinite(model(batch).cls).all()


def test_post_kd_adapter_receives_adapted_features():
    class DoubleAdapter(Adapter):
        def forward(self, data, mask=None):
            return super().forward(data * 2, mask)

    stream = ModalityStream("vid", 4, Adapter(), 1, kd_head=KDHead(4, 4), post_kd_adapter=DoubleAdapter())
    data = torch.randn(2, 3, 1, 4)
    out = stream(data, torch.ones(2, 3, dtype=torch.bool))
    torch.testing.assert_close(out["data"], data * 2)
    assert "kd" in out


def test_ensure_one_support_handles_multiple_dead_samples():
    dropout = BernoulliSupportsModalityDropout(3, drop_p=1.)
    dropout.ensure_one = True
    assert dropout(5, "cpu").sum(1).eq(1).all()


def test_single_custom_attention_layer_and_different_dimensions():
    assert len(GatedXAttentionFactory(4, 6).build([GatedXAttentionCustomArgs(heads=2)])) == 1
    block = GatedXAttentionBlock(4, 6, dim_head=2, heads=2)
    assert block(torch.randn(2, 3, 4), torch.randn(2, 2, 6)).shape == (2, 3, 4)


def test_empty_ignore_mask_does_not_change_siglip_scale():
    loss = SiglipLoss()
    a, b = torch.randn(3, 4), torch.randn(3, 4)
    torch.testing.assert_close(loss(a, b), loss(a, b, ignore_mask=torch.zeros(3, 3, dtype=torch.bool)))


def test_fp16_missing_modality_has_finite_kd_gradients():
    head = KDHead(4, 4).half()
    x = torch.randn(2, 2, 1, 4, dtype=torch.float16, requires_grad=True)
    mask = torch.tensor([[False, False], [True, True]])
    result = head(x, mask)
    result["data"][result["mask"]].square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad[0].eq(0).all()


def test_fp16_attention_pool_handles_fully_masked_rows():
    from main.model.blocks.pooling import MaskedAttnPool

    pool = MaskedAttnPool(4).half()
    x = torch.randn(2, 2, 4, dtype=torch.float16, requires_grad=True)
    result = pool(x, torch.tensor([[False, False], [True, True]]), dim=1)
    assert torch.isfinite(result).all()
    assert result[0].eq(0).all()
    result.sum().backward()
    assert torch.isfinite(x.grad).all()
