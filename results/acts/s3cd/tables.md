# S3-C-D 部署口径读数（官方 train / test，按 test acc 选，seed 0）

| 数据集 | A T-CLSC | D 任务盲比较器 | ④ 4 探针 | TS2Vec 论文 | 旧部署 SH10-4-3（3 种子，旧目录） | 旧目录 oracle |
|---|---:|---:|---:|---:|---:|---:|
| Handwriting | 0.542 | 0.375 | 0.525 | 0.515 | 0.531 | 0.549 |
| PhonemeSpectra | 0.208 | 0.180 | 0.249 | 0.233 | 0.256 | 0.253 |
| UWaveGestureLibrary | 0.912 | 0.887 | 0.919 | 0.906 | 0.918 | 0.934 |
| Cricket | 0.917 | 0.972 | 0.986 | 0.972 | 0.986 | 1.000 |
| EthanolConcentration | 0.278 | 0.316 | 0.308 | 0.308 | 0.303 | 0.331 |
| SelfRegulationSCP1 | 0.771 | 0.846 | 0.911 | 0.812 | 0.881 | 0.901 |
| **均值** | 0.605 | 0.596 | 0.650 | 0.624 | 0.646 | — |

配对（6 集）：A-D +0.008（3+ / 3−，p 0.844）；A-④ -0.045（1+ / 5−，p 0.094）；④-D +0.053（5+ / 1−，p 0.062）

≥ TS2Vec 论文数的数据集：A 2/6、D 3/6、④ 5/6

## 各臂 3 条提名的 test acc

| 数据集 | 臂 | 3 条 test acc | 3 条均值 | 选中 |
|---|---|---|---:|---|
| Handwriting | A | 0.536 / 0.542 / 0.524 | 0.534 | c363fec4d05c |
| Handwriting | D | 0.361 / 0.362 / 0.375 | 0.366 | 02cf848fadca |
| Handwriting | ④ | 0.447 / 0.442 / 0.525 | 0.471 | 051c49f0034e |
| PhonemeSpectra | A | 0.201 / 0.197 / 0.208 | 0.202 | d55464dc636c |
| PhonemeSpectra | D | 0.163 / 0.180 / 0.157 | 0.167 | dce3645e7192 |
| PhonemeSpectra | ④ | 0.238 / 0.233 / 0.249 | 0.240 | 018d72f26201 |
| UWaveGestureLibrary | A | 0.863 / 0.894 / 0.912 | 0.890 | 12e3a90ffea2 |
| UWaveGestureLibrary | D | 0.869 / 0.819 / 0.887 | 0.858 | c3e414679b65 |
| UWaveGestureLibrary | ④ | 0.916 / 0.884 / 0.919 | 0.906 | 051c49f0034e |
| Cricket | A | 0.875 / 0.917 / 0.458 | 0.750 | c79f7a2bc920 |
| Cricket | D | 0.958 / 0.972 / 0.944 | 0.958 | 2bc0ec029986 |
| Cricket | ④ | 0.958 / 0.986 / 0.972 | 0.972 | a04011c2e679 |
| EthanolConcentration | A | 0.247 / 0.270 / 0.278 | 0.265 | 25d990691854 |
| EthanolConcentration | D | 0.255 / 0.316 / 0.274 | 0.281 | cf9f647162a7 |
| EthanolConcentration | ④ | 0.251 / 0.243 / 0.308 | 0.267 | ce013ded0e79 |
| SelfRegulationSCP1 | A | 0.761 / 0.686 / 0.771 | 0.739 | f336d444a329 |
| SelfRegulationSCP1 | D | 0.833 / 0.823 / 0.846 | 0.834 | 4cda9cf934f0 |
| SelfRegulationSCP1 | ④ | 0.823 / 0.911 / 0.829 | 0.854 | 9e289c7404a4 |

A 与 D 提名重合条数：Handwriting 0，PhonemeSpectra 0，UWaveGestureLibrary 0，Cricket 0，EthanolConcentration 0，SelfRegulationSCP1 0
