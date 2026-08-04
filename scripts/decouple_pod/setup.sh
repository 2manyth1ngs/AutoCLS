#!/usr/bin/env bash
# 解耦实验 pod 环境准备（migration 后容器层重置，依赖按 runbook §2.4 pin 重装）。
# 用法：bash scripts/decouple_pod/setup.sh
set -euo pipefail
cd /workspace/repo/CLS

echo "===== ① 依赖重装（--break-system-packages：镜像 python 是 PEP 668 受管环境）====="
pip install -q --break-system-packages "aeon==1.3.0" "scikit-learn==1.7.2" \
    "pyyaml==6.0.3" scipy pytest
python - <<'PY'
import numpy, sklearn, yaml, aeon, scipy, torch
print("numpy", numpy.__version__, "| sklearn", sklearn.__version__,
      "| aeon", aeon.__version__, "| scipy", scipy.__version__,
      "| torch", torch.__version__, "| cuDNN", torch.backends.cudnn.version())
assert torch.__version__.startswith("2.8.0+cu128"), "torch 指纹不符（Blackwell 闸门）"
assert torch.backends.cudnn.version() == 91002, "cuDNN 指纹不符（Blackwell 闸门）"
PY

echo "===== ② GPU 前向冒烟 ====="
python - <<'PY'
import torch
assert torch.cuda.is_available()
x = torch.randn(4096, 4096, device="cuda")
print(" ", torch.cuda.get_device_name(0), "matmul OK", f"{(x@x).sum().item():.3e}")
PY

echo "===== ③ 目标池 fail-fast（铁律 6 代码层）====="
python - <<'PY'
from autocls.data.source_pool import load_source_dataset
for name in ("Epilepsy", "FaceDetection", "UCI-HAR"):
    try:
        load_source_dataset(name)
        raise SystemExit(f"!!! {name} 竟然加载成功 —— 铁律 6 防护失效，停 !!!")
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 —— 拒绝类型随加载路径不同，拒绝即通过
        print(f"  {name}: 正确拒绝 ({type(e).__name__})")
PY

echo "===== ④ 两份 CLS 目录到位 ====="
python - <<'PY'
from autocls.labeling import load_probe_cls
probe, rank = load_probe_cls("results/route_a_probe/probe_cls_catalog.json", 10)
rand, rrank = load_probe_cls("results/decouple_random_cls/random_cls_catalog.json", 20)
assert len(probe) == 10 and len(rank) == 10
assert len(rand) == 20 and rrank == {}, "随机目录必须 20 条且无 har_rank"
print(f"  探针 {len(probe)} 条（har_rank {len(rank)} 项）｜随机 {len(rand)} 条（无排名字段）")
PY

echo "===== ⑤ 单测冒烟 ====="
python -m pytest tests/ -q 2>&1 | tail -2
echo "===== setup 全绿 ====="
