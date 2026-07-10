"""数据增强：AutoCL 附录 A 的 6 种增强 + Table 8 的 5 种施加顺序。

张量约定 (B, T, C)。每个增强关联参数 p ∈ [0,1]，p=0.0 表示不启用（恒等）。

两视图生成语义（论文正文只说"变换出两个相关视图"，未给出两视图间随机性的
协调细节，此处按对比学习通用语义补全并注明）：
- 除 crop 外的增强对两个视图独立随机施加（同一 pipeline 跑两遍）；
- crop 在两视图间协同：共用同一组裁剪坐标 (t1, t2, t1', t2')，保证共享段对齐；
- resize 改变序列长度，两视图共享同一目标长度（否则 crop 坐标/批内形状不一致）。
"""
from __future__ import annotations

from typing import Optional

import torch

# AutoCL Table 8：5 种预设顺序（order 字段取 0..4 对应 Order-1..Order-5）。
# resize 恒排第一；crop 的位置随 order 依次前移（第 6、5、4、3、2 位）。
ORDERS: list[list[str]] = [
    ["resize", "rescale", "freq_mask", "jitter", "point_mask", "crop"],  # Order-1
    ["resize", "rescale", "freq_mask", "jitter", "crop", "point_mask"],  # Order-2
    ["resize", "rescale", "freq_mask", "crop", "jitter", "point_mask"],  # Order-3
    ["resize", "rescale", "crop", "freq_mask", "jitter", "point_mask"],  # Order-4
    ["resize", "crop", "rescale", "freq_mask", "jitter", "point_mask"],  # Order-5
]


def jitter(x: torch.Tensor, p: float, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """附录 A Jittering：x = x + N(0, p)，p 为高斯噪声标准差。"""
    if p == 0.0:
        return x
    noise = torch.randn(x.shape, generator=generator, dtype=x.dtype, device=x.device)
    return x + noise * p


def sample_crop_coords(
    T: int, p: float, generator: Optional[torch.Generator] = None
) -> tuple[int, int, int, int]:
    """采样附录 A Random Cropping 的坐标 (t1, t2, t1', t2')。

    t1 ≤ t2 < t1' ≤ t2'，共享段 x[t2:t1']，p = T'/T，T' = t1' - t2。
    简化说明：整个 batch 共用同一组裁剪坐标（TS2Vec 在此基础上还做逐行随机
    偏移，论文未要求，此处从简）。
    """
    T_shared = max(1, round(p * T))

    def _randint(low: int, high: int) -> int:  # [low, high] 闭区间
        return int(torch.randint(low, high + 1, (1,), generator=generator).item())

    t2 = _randint(0, T - T_shared)      # 共享段起点
    t1p = t2 + T_shared                 # 共享段终点 = t1'
    t1 = _randint(0, t2)                # v1 起点 ≤ t2
    t2p = _randint(t1p, T)              # v2 终点 ≥ t1'
    return t1, t2, t1p, t2p


def _crop_with_coords(
    va: torch.Tensor, vb: torch.Tensor, coords: tuple[int, int, int, int]
) -> tuple[torch.Tensor, torch.Tensor, slice, slice]:
    """用共享坐标分别裁剪两个（可能已被前置增强分叉的）视图。"""
    t1, t2, t1p, t2p = coords
    T_shared = t1p - t2
    sl1 = slice(t2 - t1, t2 - t1 + T_shared)
    sl2 = slice(0, T_shared)
    return va[:, t1:t1p], vb[:, t2:t2p], sl1, sl2


def crop_views(
    x: torch.Tensor, p: float, generator: Optional[torch.Generator] = None
) -> tuple[torch.Tensor, torch.Tensor, slice, slice]:
    """附录 A Random Cropping：从同一序列裁出带共享段的两个子序列。

    x1 = x[t1:t1'], x2 = x[t2:t2']，共享段 x[t2:t1']。
    返回 (x1, x2, sl1, sl2)，sl1/sl2 为共享段在各自视图内的时间切片，
    编码后取 z1[:, sl1] 与 z2[:, sl2] 做对比（对齐 TS2Vec 的 overlap 用法）。
    """
    _, T, _ = x.shape
    if p == 0.0:  # 不启用：两视图均为全长序列
        return x, x, slice(0, T), slice(0, T)
    coords = sample_crop_coords(T, p, generator)
    return _crop_with_coords(x, x, coords)


def resize(x: torch.Tensor, p: float, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """附录 A Resizing (length)：p 为高斯噪声标准差，缩放序列长度。

    论文未给全（"The final length is given by" 后公式被截断），按"长度乘
    (1+N(0,p)) 类缩放"语义补全：T_new = round(T·(1+ε))，ε~N(0,p)，再对时间维
    做线性插值；T_new 夹在 [2, 2T] 内防止退化/爆炸。
    """
    if p == 0.0:
        return x
    _, T, _ = x.shape
    eps = torch.randn(1, generator=generator).item() * p
    T_new = int(max(2, min(2 * T, round(T * (1.0 + eps)))))
    if T_new == T:
        return x
    # (B, T, C) → (B, C, T) 线性插值 → 还原
    out = torch.nn.functional.interpolate(
        x.transpose(1, 2), size=T_new, mode="linear", align_corners=False
    )
    return out.transpose(1, 2)


def rescale(x: torch.Tensor, p: float, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """附录 A Rescaling (amplitude)：p 为高斯噪声标准差，缩放幅度。

    论文未给全（"The final amplitude is given by" 后公式被截断），按"幅度乘
    (1+N(0,p)) 类缩放"语义补全：每个样本独立采样一个标量缩放因子。
    """
    if p == 0.0:
        return x
    B = x.shape[0]
    factor = 1.0 + torch.randn(B, 1, 1, generator=generator, dtype=x.dtype, device=x.device) * p
    return x * factor


def point_mask(x: torch.Tensor, p: float, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """附录 A Point Masking：x = x ⊙ m，m[t] ~ Bernoulli(1-p)。

    m 按时间步采样（跨通道广播），p 为被掩掉的时间点比例。
    """
    if p == 0.0:
        return x
    B, T, _ = x.shape
    keep = (torch.rand(B, T, 1, generator=generator, device=x.device) >= p).to(x.dtype)
    return x * keep


def freq_mask(x: torch.Tensor, p: float, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """附录 A Frequency Masking：x_f = DFT(x)，x_f = x_f ⊙ m（m[t]~Bernoulli(1-p)），
    x = IDFT(x_f)。

    实现细节：用 rfft/irfft（实序列的单边谱），逆变换天然为实序列，与论文
    DFT→IDFT 语义等价；m 按频率采样（跨通道广播）。
    """
    if p == 0.0:
        return x
    B, T, _ = x.shape
    xf = torch.fft.rfft(x, dim=1)                       # (B, F, C) 复数
    n_freq = xf.shape[1]
    keep = (torch.rand(B, n_freq, 1, generator=generator, device=x.device) >= p)
    xf = xf * keep
    return torch.fft.irfft(xf, n=T, dim=1).to(x.dtype)


# 逐视图独立施加的增强（crop 特殊处理，不在此表）
_PER_VIEW_AUGS = {
    "jitter": jitter,
    "rescale": rescale,
    "point_mask": point_mask,
    "freq_mask": freq_mask,
}


def augment_two_views(
    x: torch.Tensor,
    params: dict[str, float],
    order: int,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor, slice, slice]:
    """按 ORDERS[order] 依次施加 6 种增强，生成两个视图。

    params：{增强名: p}，p=0.0 跳过该增强。
    返回 (v1, v2, sl1, sl2)，sl1/sl2 含义同 crop_views；未启用 crop 时为全长切片。
    """
    v1, v2 = x, x
    sl1 = sl2 = None
    for name in ORDERS[order]:
        p = params[name]
        if p == 0.0:
            continue
        if name == "resize":
            # 两视图共享同一目标长度（见模块 docstring）。Table 8 中 resize 恒排
            # 第一，此时两视图尚未分叉，缩放一次即可。
            if v1 is not v2:
                raise RuntimeError("resize 必须在两视图分叉前施加（Table 8 中恒排第一）")
            v1 = v2 = resize(v1, p, generator=generator)
        elif name == "crop":
            # 两视图可能已被前置增强分叉，用共享坐标分别裁剪以保证共享段对齐
            coords = sample_crop_coords(v1.shape[1], p, generator)
            v1, v2, sl1, sl2 = _crop_with_coords(v1, v2, coords)
        else:
            fn = _PER_VIEW_AUGS[name]
            v1 = fn(v1, p, generator=generator)
            v2 = fn(v2, p, generator=generator)
    if sl1 is None:  # crop 未启用：共享段 = 全长
        sl1 = slice(0, v1.shape[1])
        sl2 = slice(0, v2.shape[1])
    return v1, v2, sl1, sl2
