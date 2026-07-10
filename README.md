# AutoCLS

Zero-Shot NAS for Contrastive Learning Strategy Search：用 AutoCTS++ 的零样本比较器机制，
在 AutoCL 的对比学习策略空间（CLS）上做零样本搜索，encoder 固定。

- 完整设计：`docs/TECHNICAL_DESIGN.md`
- 项目约定：`CLAUDE.md`
- 策略空间唯一真源：`configs/space.yaml`
- encoder 选择：`configs/encoder.yaml`（`dilated_cnn` 默认 / `transformer`）

**"固定 encoder" 语义**：架构固定（超参见 `configs/encoder.yaml`）；评估每个 CLS 时
都用 `build_encoder()` 新建实例、按该 CLS **从头重新预训练一份权重**——不是权重冻结。

当前进度：M1（单 CLS 在 HAR 上端到端：数据增强 → 对比预训练 → SVM-RBF 下游评估，
GGS 复现 AutoCL 量级）。已完成模块：space、contrastive、encoders、downstream、data。

准备数据（自动下载 UCI HAR 并缓存到 `data/HAR/har.npz`）：

```bash
python scripts/1_prepare_data.py --dataset har
```

运行测试：

```bash
conda run -n pytorch python -m pytest tests -q
```
