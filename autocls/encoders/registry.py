"""encoder 注册表：按 configs/encoder.yaml 切换（TECHNICAL_DESIGN.md 关键决策 2）。

encoder.yaml 结构：
    encoder: dilated_cnn            # 或 transformer
    dilated_cnn: {超参...}          # 各 encoder 的构造参数（可省略用默认值）
    transformer: {超参...}
"""
from __future__ import annotations

from pathlib import Path

import yaml
from torch import nn

from .dilated_cnn import DilatedCNNEncoder
from .transformer import TransformerEncoder

DEFAULT_ENCODER_YAML = Path(__file__).resolve().parents[2] / "configs" / "encoder.yaml"

ENCODERS: dict[str, type[nn.Module]] = {
    "dilated_cnn": DilatedCNNEncoder,
    "transformer": TransformerEncoder,
}


def build_encoder(
    input_dims: int,
    config_path: str | Path = DEFAULT_ENCODER_YAML,
    overrides: dict | None = None,
) -> nn.Module:
    """按 yaml 配置实例化一个**未预训练**的 encoder。

    "固定 encoder" 语义：架构固定；每个 CLS 各自调用本函数拿到新实例、
    从头预训练一份权重（不是权重冻结）。

    overrides（默认 None = 零改动）：覆盖 yaml 里该 encoder 的构造参数——仅供全量
    协议对齐的 --ladder 阶段打开 dilated_cnn 的 input_mask / repr_dropout；造 label
    的所有调用方一律不传，故默认路径与历史完全一致。
    """
    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    name = cfg["encoder"]
    if name not in ENCODERS:
        raise ValueError(f"未知 encoder: {name}（可选: {sorted(ENCODERS)}）")
    kwargs = dict(cfg.get(name) or {})
    if overrides:
        kwargs.update(overrides)
    return ENCODERS[name](input_dims=input_dims, **kwargs)
