# WP16 GEO-S 检验 PROTOCOL：几何子空间搜索——UEA 29 官方靶 × 18 个损失几何家族（干净基），量干净的家族净空、E_T 能否按靶选家族、E 排序 + SH 的部署值

**会话定位**：GEO-E 用目录标签（家族与温度 / 增强混淆）测 E_T 预测家族 → DEAD；用户 2026-09-18 拍板：部署在 UEA，就在 UEA 上做——增强固定 crop-only 干净基、非增强旋钮固定成基，只让损失几何（loss_type × sim_func × norm = 18 家族）变化，部署口径打标签，UEA 内部留一。本协议同时回答三问：(S1) 干净的几何家族净空有多大；(S2) E_T 能否按靶选家族；(S3) E 排序 18 条 + SH 是否赢固定顺序 + SH。
产物 `results/tclsc_geos/`；脚本 `scripts/82_tclsc_geos.py`；单测 `tests/test_tclsc_geos.py`；pod 三件套 `pa_pod/geos_{bundle,setup,run}.sh`。用户已同意开 pod（"那请你开始吧"）。

## §0 已看过的读数

- GEO-E：29 靶目录（11 条）分 8 家族：家族 oracle − 固定最好家族 +0.023（含赢家诅咒、温度 / 增强混淆）；E_T@60 命中 0.14 / 增益 −0.026。
- WP15：UCR 52 集干净两基（B_T = triplet/dot/none τ 1.0、B_I = info_nce/cosine/l2 τ 0.1，crop 0.5、其余增强 0）：ICC 0.72、oracle − 永远 B_T +0.007；GEOM / A3 不可预测。
- H2：23 靶干净基官方 acc 均值 0.750，vs 全增强同旋钮 +0.021。
- **没看过**：任何靶上 18 家族里除 B_T / B_I（及目录里 8 家族的混淆版）以外家族的官方口径结果；E_T 与干净家族标签的关系。

## §1 设计

- **靶**：29 个有 E_T 的 UEA 官方靶（WP3 六靶 + UEA-30 二十三靶；`configs/uea30.yaml` 尺寸 / iters 规则 / EigenWorms 分段）。
- **家族网格**：loss_type {info_nce, triplet} × sim_func {dot, cosine, euclidean} × norm {none, layer_norm, l2} = 18，全部 `validate` 合法；温度按 WP15 惯例固定：triplet → 1.0、info_nce → 0.1（一个温度档，先不扫）；其余字段 = 干净基（crop 0.5、其他增强 0、order 2、instance / temporal / cross_scale True、kernel 3、pool avg、adj_neighbor True、input_mask 关）。
- **轨迹**：`ts2vec_official`（batch 8、iters 按尺寸 200 / 600、截点 25 / 50 / 100%、SWA、scaler train、svm_grid ts2vec），seed 0；两个锚家族（info_nce/euclidean/none = GGS 家族、triplet/dot/none = 先验 top-1 家族）加 seed 1 作噪声 σ_d。**29 × 20 = 580 格**，估 5090 ≈ 5.5 h ≈ $6–7。缓存 `runs.jsonl` + `--resume` + `--verify`。
- **S1 净空**：U[d, f] = acc(f) − max_f acc；家族 oracle − 固定最好家族（训练折上定，LODO 诚实）/ − 先验 top-1 家族 / − GGS 家族；超 2σ_d 靶占比；两大家族占冠军比例；与目录家族冠军的一致率（混淆程度）。
- **S2 可预测**：81 号同式（LODO 岭、内层 LOO 选 α、置换 1000）：E_T@60 主、E_T@290 / φ 副；增益 vs 固定最好家族、命中率 vs 猜多数。
- **S3 部署翻译**：18 条官方轨迹的截点上模拟 SH6-3-3（25% 留 3 → 50% 留 3 → 100% 取最好；WP3 同式）：起跑顺序 = E_T 预测的家族效用降序 vs 固定顺序（训练折均值）vs 先验 top-1 优先 vs oracle；同成本配对。
- **ctx / 特征**：E_T 表 `tclsc_s1et_deploy/et_official.jsonl` + `tclsc_s1et_uea30/et_uea30.jsonl`（22 维 @60）；折内标准化。

## §2 判读（先于数据）

| 门 | 定义 |
|---|---|
| C0 | 580 格 key 唯一；每格 3 截点齐；18 家族全部合法；两锚家族 seed 0 vs seed 1 的 |Δ| 中位 ≤ 0.03 |
| **G-S1 干净净空** | 家族 oracle − 固定最好家族 ≥ +0.015（29 靶均值）且 ≥ 1/3 靶超 2σ_d |
| **G-S2 E 可预测** | E_T@60 LODO 增益 vs 固定最好家族 ≥ +0.005 且置换 p < 0.05 |
| **G-S3 部署** | E 排序 SH6-3-3 − 固定顺序 SH6-3-3 ≥ +0.005（29 靶配对，p < 0.05） |

**读数码**：G-S1 ✓ ∧ G-S2 ✓ ∧ G-S3 ✓ → **GEOS-GO**（几何子空间路线成立：扩温度档、写主臂）；G-S1 ✓ ∧ G-S2 ✗ → **GEOS-HEADROOM-ONLY**（几何是数据集级但 E 读不出 → 只能测量式："两家族起跑"的 SH 作为可兑现形式）；G-S1 ✗ → **GEOS-FLAT**（干净几何净空 ≈ 噪声，目录里的 +0.023 是混淆；几何路线关闭）；G-S1 ✓ ∧ G-S2 ✓ ∧ G-S3 ✗ → **GEOS-RANK-ONLY**。
只报：逐靶 18 家族表；温度惯例的敏感性（无扫描，只标注）；φ / E_T@290 臂；目录冠军家族 vs 干净冠军家族的一致率。

## §3 预测（先验）

| # | 预测 | 先验 |
|---|---|---|
| P1 | G-S1 | 60% |
| P2 | G-S2 | 20% |
| P3 | G-S3 | 20% |
| P4 | 两大家族（info_nce/euclidean/none、triplet/dot/none）占冠军 ≥ 60% | 55% |
| P5 | 目录冠军家族与干净冠军家族一致 ≤ 60% 的靶（混淆是真的） | 55% |
| P6 | 增益 / 净空集中在 EEG / 传感器类靶 | 60% |

## §4 执行

1. 本文件 + `tests/test_tclsc_geos.py`（先红）→ 协议锁 commit；2. `scripts/82_tclsc_geos.py`（--plan / --run / --verify / --analyze）→ 绿 → 本地冒烟（1 靶 2 家族 8 iters）→ commit；3. pod（已授权）：nvidia-smi + CUDA 闸 → 上传 → geos_setup → nohup geos_run（分 2 进程按靶 → 合并 → verify 不过补跑 → 打包）；3 分钟内确认首格、10 分钟无进度必查；4. 拉回 → --analyze → REPORT → stop + confirm 3/3 → commit → memory。

## §5 铁律对照

铁律 1（家族取值来自 space.yaml，validate 把关）；2（iters 按协议规则、截点同一轨迹）；3（runs.jsonl + resume + verify）；5（先红）；6（E_T 只由目标自己的探针得到；预测器 LODO；固定最好家族在训练折上定）；7；8。

## §6 执行台账（跑后填写）

（空）
