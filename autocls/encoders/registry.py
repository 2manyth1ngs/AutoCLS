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


def build_encoder(input_dims: int, config_path: str | Path = DEFAULT_ENCODER_YAML) -> nn.Module:
    """按 yaml 配置实例化一个**未预训练**的 encoder。

    "固定 encoder" 语义：架构固定；每个 CLS 各自调用本函数拿到新实例、
    从头预训练一份权重（不是权重冻结）。
    """
    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    name = cfg["encoder"]
    if name not in ENCODERS:
        raise ValueError(f"未知 encoder: {name}（可选: {sorted(ENCODERS)}）")
    kwargs = cfg.get(name) or {}
    return ENCODERS[name](input_dims=input_dims, **kwargs)
