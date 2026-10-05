"""InfoTS 式学增强（WP17-ACTS AL，results/acts/al/PROTOCOL.md §1）。

忠实于官方代码 github.com/chengw07/InfoTS（InfoTS.zip：models/augmentations.py::AutoAUG、models/augclass.py、infots.py::fit / meta_fit），
与论文的差异以代码为准（PROTOCOL §0）：
- 7 种候选变换（官方参数）；增强视图 = softmax((q + logistic 噪声) / 1) 加权混合 Σ wᵢ tᵢ(x)，每 batch 以 p1 = 0.2 的概率增强，否则原样；
- 两视图 = 增强混合 vs 原始 x；
- 元学习步：编码器 eval，zᵥ / zₓ = 时间维最大池化，最小化 meta_beta·(CE(h(zᵥ), y) + CE(h(zₓ), y))，同步更新 q 与分类头 h（AdamW lr 0.01）；
  官方每 meta_epoch = 2 遍数据做一整遍（多样性项在官方代码中被注释掉，此处同样不用）。
随机性一律走传入的 torch.Generator（CPU），不碰全局 RNG。张量约定 (B, T, C)。
时间扭曲官方依赖 tsaug.TimeWarp(n_speed_change=100, max_speed_ratio=10)，此处用分段变速 + 线性插值实现（论文未给全，按 tsaug 语义补全）。
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

TRANSFORM_NAMES = ("subsequence", "cutout", "jitter", "scaling", "time_warp", "window_slice", "window_warp")   # = 官方 all_augs 顺序
P1 = 0.2
META_EVERY = 2
META_LR = 0.01
META_BETA = 1.0


def _randint(g: torch.Generator, low: int, high: int) -> torch.Tensor:
    """[low, high) 均匀整数（high > low）。"""
    return torch.randint(int(low), int(high), (1,), generator=g)[0]


def _interp(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """按逐样本的分数位置 pos (B, T') 在时间轴上线性插值 x (B, T, C) → (B, T', C)。"""
    B, T, C = x.shape
    pos = pos.clamp(0, T - 1)
    lo = pos.floor().long(); hi = (lo + 1).clamp(max=T - 1); w = (pos - lo.to(pos.dtype)).unsqueeze(-1).to(x.dtype)
    xl = x.gather(1, lo.unsqueeze(-1).expand(-1, -1, C)); xh = x.gather(1, hi.unsqueeze(-1).expand(-1, -1, C))
    return xl * (1 - w) + xh * w


def subsequence(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """官方 subsequence：整批同一 crop_l ~ U[2, T]、同一起点，窗外置零。"""
    T = x.shape[1]
    crop_l = int(_randint(g, 2, T + 1)); start = int(_randint(g, 0, T - crop_l + 1))
    y = x.clone(); y[:, :start] = 0.0; y[:, start + crop_l:] = 0.0
    return y


def cutout(x: torch.Tensor, g: torch.Generator, perc: float = 0.1) -> torch.Tensor:
    """官方 cutout：整批同一 10% 窗置零（起点 ~ U[0, T − win − 1)）。"""
    T = x.shape[1]; win = int(perc * T)
    start = int(_randint(g, 0, max(T - win - 1, 1)))
    y = x.clone(); y[:, start:start + win] = 0.0
    return y


def jitter(x: torch.Tensor, g: torch.Generator, sigma: float = 0.3) -> torch.Tensor:
    return x + sigma * torch.randn(x.shape, generator=g).to(x.device, x.dtype)


def scaling(x: torch.Tensor, g: torch.Generator, sigma: float = 0.5) -> torch.Tensor:
    """官方 scaling：逐样本逐通道乘 N(1, σ)。"""
    B, _T, C = x.shape
    f = 1.0 + sigma * torch.randn((B, C), generator=g)
    return x * f.unsqueeze(1).to(x.device, x.dtype)


def time_warp(x: torch.Tensor, g: torch.Generator, n_speed_change: int = 100, max_speed_ratio: float = 10.0) -> torch.Tensor:
    """tsaug TimeWarp 语义：K 个变速段、段内速度 ~ U[1, max_speed_ratio]，累积成单调时间映射后线性插值回原长。"""
    B, T, _C = x.shape
    if T < 3:
        return x.clone()
    K = max(1, min(int(n_speed_change), T - 1))
    speeds = 1.0 + (max_speed_ratio - 1.0) * torch.rand((B, K), generator=g)
    seg = (torch.arange(T - 1) * K) // (T - 1)
    steps = speeds[:, seg]
    pos = torch.cat([torch.zeros(B, 1), torch.cumsum(steps, 1)], 1)
    pos = pos / pos[:, -1:] * (T - 1)
    return _interp(x, pos.to(x.device))


def window_slice(x: torch.Tensor, g: torch.Generator, reduce_ratio: float = 0.5) -> torch.Tensor:
    """官方 window_slice（diff_len=True）：逐样本截 ⌈0.5T⌉ 长窗，线性插值回原长。"""
    B, T, _C = x.shape
    target = int(math.ceil(reduce_ratio * T))
    if target >= T or T - target < 1:
        return x.clone()
    starts = torch.randint(0, T - target, (B,), generator=g).to(torch.float32)
    rel = torch.linspace(0, target - 1, T)
    return _interp(x, (starts.unsqueeze(1) + rel.unsqueeze(0)).to(x.device))


def window_warp(x: torch.Tensor, g: torch.Generator, window_ratio: float = 0.3, scales=(0.5, 2.0)) -> torch.Tensor:
    """官方 window_warp：逐样本 ⌈0.3T⌉ 窗按 0.5 / 2 倍变速，拼接后线性插值回原长。"""
    B, T, _C = x.shape
    warp = int(math.ceil(window_ratio * T))
    if T - warp - 1 <= 1:
        return x.clone()
    starts = torch.randint(1, T - warp - 1, (B,), generator=g)
    sc = torch.tensor(scales)[torch.randint(0, len(scales), (B,), generator=g)]
    out = []
    xt = x.transpose(1, 2)                                    # (B, C, T)
    for i in range(B):
        s, e = int(starts[i]), int(starts[i]) + warp
        seg = F.interpolate(xt[i:i + 1, :, s:e], size=max(1, int(warp * float(sc[i]))), mode="linear", align_corners=False)
        r = torch.cat([xt[i:i + 1, :, :s], seg, xt[i:i + 1, :, e:]], -1)
        out.append(F.interpolate(r, size=T, mode="linear", align_corners=False))
    return torch.cat(out, 0).transpose(1, 2)


TRANSFORMS = {"subsequence": subsequence, "cutout": cutout, "jitter": jitter, "scaling": scaling,
              "time_warp": time_warp, "window_slice": window_slice, "window_warp": window_warp}


class AutoAug(nn.Module):
    """官方 AutoAUG 的单视图版本（aug_p2 = 0 ⇒ 第二视图恒为原样，只用 weight[0]）。q 留在 CPU。"""

    def __init__(self, seed: int = 0, p1: float = P1):
        super().__init__()
        g = torch.Generator().manual_seed(int(seed))
        self.q = nn.Parameter(0.01 * torch.randn(len(TRANSFORM_NAMES), generator=g))   # 官方 init.normal_(std 0.01)
        self.p1 = float(p1)

    def sample_weights(self, g: torch.Generator, temperature: float = 1.0, bias: float = 1e-4) -> torch.Tensor:
        """官方 get_sampling（训练态）：eps ∈ (bias, 1 − bias) 的 logistic 噪声 + q，除温度后 softmax。"""
        eps = (bias - (1 - bias)) * torch.rand(self.q.shape, generator=g) + (1 - bias)
        gate = torch.log(eps) - torch.log(1 - eps)
        return torch.softmax((gate + self.q) / temperature, -1)

    def view(self, x: torch.Tensor, g: torch.Generator, p1: Optional[float] = None) -> torch.Tensor:
        """以概率 p1 返回 Σ wᵢ tᵢ(x)，否则原样（官方：random() > p1 → 不增强）。"""
        p1 = self.p1 if p1 is None else float(p1)
        if float(torch.rand((), generator=g)) > p1:
            return x
        w = self.sample_weights(g)
        xs = torch.stack([TRANSFORMS[n](x, g) for n in TRANSFORM_NAMES], 0)
        return (w.to(x.device, x.dtype).view(-1, 1, 1, 1) * xs).sum(0)

    def weights(self) -> np.ndarray:
        with torch.no_grad():
            return torch.softmax(self.q.detach(), -1).cpu().numpy()


def meta_pass(aug: AutoAug, head: nn.Module, encoder: nn.Module, X: torch.Tensor, y: torch.Tensor, *, batch_size: int, device: str,
              gen: torch.Generator, drop_last: bool, meta_opt, cls_opt, meta_beta: float = META_BETA) -> int:
    """官方 meta_fit：编码器 eval，对完整 TRAIN 一遍，最小化 meta_beta·(CE(h(zᵥ), y) + CE(h(zₓ), y))，同步更新 q 与 h。返回步数。"""
    was = encoder.training
    encoder.eval()
    N = X.shape[0]; bs = min(int(batch_size), N); n = 0
    perm = torch.randperm(N, generator=gen)
    for start in range(0, N, bs):
        idx = perm[start:start + bs]
        if (drop_last and idx.numel() < bs) or idx.numel() < 2:
            continue
        x = X[idx]; yy = y[idx].to(device)
        meta_opt.zero_grad(); cls_opt.zero_grad()
        v = aug.view(x, gen)
        zv = encoder(v.to(device)).max(dim=1).values
        zx = encoder(x.to(device)).max(dim=1).values
        loss = meta_beta * (F.cross_entropy(head(zv), yy) + F.cross_entropy(head(zx), yy))
        loss.backward()
        meta_opt.step(); cls_opt.step(); n += 1
    encoder.zero_grad(set_to_none=True)
    if was:
        encoder.train()
    return n
