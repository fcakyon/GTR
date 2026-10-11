"""GTR ONNX export with GLA in plain ONNX ops, following fla's chunk_gla math, instead of GTR's TensorRT plugin.

GTR's plugin kernel splits exp(G_i - G_j) into exp(G_i) * (k_j / exp(G_j)), and k_j / exp(G_j) overflows once a trained
gate drives the in-chunk sum G below about -87 (gla_chunk.cuh:267-269). Here every exponent is a difference of
in-order gate sums, so it is <= 0 and nothing can overflow. GTR's files are not modified: export_onnx.py runs as is,
with gtr_onnx_patches._GLAOp4 swapped and its _manual_rms_norm run in FP32, since x^2 overflows FP16 once |x| > 256
(trained GTR-S reaches 387 in block 0). --no-fuse-gate is required so the RMSNorm gate is plain ops too. The GTR repo
is the checkout holding this file.

Usage: python gtr_stable.py <export_onnx.py args>  # e.g. -c CFG -o X.onnx --dtype fp16 --resume CKPT --no-fuse-gate
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

GTR = str(Path(__file__).resolve().parents[1])
sys.path[:0] = [GTR, f"{GTR}/tensorrt_plugin"]
import gtr_onnx_patches as gop


def gla_stable(q, k, v, gk, scale, C=64):
    """GLA forward o_t = q_t h_t, h_t = exp(gk_t) h_{t-1} + k_t^T v_t, in 64-token chunks with plain ops.

    q, k, gk are [B, T, H, K], v is [B, T, H, V] and gk is the per-token log gate (<= 0). Runs in FP32 and returns v's
    dtype.
    """
    dt, (B, T, H, K), N = v.dtype, q.shape, -(-q.shape[1] // C)
    q, k, v, gk = (
        F.pad(t.float(), (0, 0, 0, 0, 0, N * C - T)).view(B, N, C, H, -1).permute(0, 3, 1, 2, 4) for t in (q, k, v, gk)
    )  # [B, H, N, C, D]
    q = q * scale
    G = gk.cumsum(3)  # gate sum from the chunk start, inclusive
    # inside a chunk: o_i = sum_{j <= i} (sum_d q_id k_jd exp(G_id - G_jd)) v_j
    # masks multiply after exp(min(d, 0)), so no exponent is ever positive, not even where a mask zeroes it
    causal = torch.ones(C, C, device=q.device).tril()[:, :, None]
    d = (G.unsqueeze(4) - G.unsqueeze(3)).clamp_max(0)  # [B, H, N, C, C, K]
    o = (q.unsqueeze(4) * k.unsqueeze(3) * d.exp() * causal).sum(-1) @ v
    # across chunks: U_m is chunk m's state at its own end, the decay from the end of chunk m to the start of chunk n
    # is the gate sum over the chunks strictly between them, summed directly so no large totals cancel
    tot = G[:, :, :, -1]  # [B, H, N, K]
    U = (k * (tot.unsqueeze(3) - G).exp()).transpose(-1, -2) @ v  # [B, H, N, K, V]
    i = torch.arange(N, device=q.device)
    n, m, l = i.view(-1, 1, 1), i.view(1, -1, 1), i.view(1, 1, -1)
    dec = (((m < l) & (l < n)).float().view(N * N, N) @ tot).view(B, H, N, N, K).exp() * (
        m < n
    ).float()  # [B, H, n, m, K]
    h = (dec.permute(0, 1, 4, 2, 3) @ U.transpose(2, 3)).permute(
        0, 1, 3, 2, 4
    )  # state entering chunk n [B, H, N, K, V]
    o = o + (q * G.exp()) @ h
    return o.permute(0, 2, 3, 1, 4).reshape(B, N * C, H, -1)[:, :T].to(dt)


class StableGLA:
    """Stands in for gtr_onnx_patches._GLAOp4: same call, plain ops in the graph instead of the plugin node."""

    @staticmethod
    def apply(q, k, v, gk, scale, gk_normalizer):
        assert gk_normalizer == 0.0, "gk must already be logsigmoid(x) / normalizer (export without --gk-in-plugin)"
        return gla_stable(q, k, v, gk, scale)


def rms_norm_fp32(x, weight, eps):
    """GTR's _manual_rms_norm computed in FP32, as fla's fused norm does, and returned in x's dtype."""
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (out * weight.float() if weight is not None else out).to(x.dtype)


# _GLAOp6 (fused gate) would put the plugin back, hence --no-fuse-gate
gop._GLAOp4, gop._GLAOp6, gop._manual_rms_norm = StableGLA, None, rms_norm_fp32


if __name__ == "__main__":
    import export_onnx

    sys.argv = ["export_onnx.py", *sys.argv[1:]]
    export_onnx.main()
