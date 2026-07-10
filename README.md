# AutoCLS

Zero-Shot NAS for Contrastive Learning Strategy Search：用 AutoCTS++ 的零样本比较器机制，
在 AutoCL 的对比学习策略空间（CLS）上做零样本搜索，encoder 固定。

- 完整设计：`docs/TECHNICAL_DESIGN.md`
- 项目约定：`CLAUDE.md`
- 策略空间唯一真源：`configs/space.yaml`

当前进度：M0（目录骨架 + CLS 策略空间 + CLS 向量编码 + 单测）。

运行测试：

```bash
conda run -n pytorch python -m pytest tests -q
```
