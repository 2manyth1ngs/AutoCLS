"""候选数据集 adapter 包（换池调研第1步：广撒网 + 实际加载确认统计）。

每个候选一个模块，模块名 = 小写数据集名（如 `wisdm.py`）。契约只有两项：

    SPEC = {
        "name": "WISDM",          # canonical 名（= data/candidates/<name>.npz 的文件名）
        "domain": "HAR",          # HAR / EEG / ECG / Audio / Motion / Other(...)
        "fs": 20,                 # 原生采样率 Hz（写实际值，不写论文声称值）
        "url": "https://...",     # 下载地址（多份则给主档）
        "n_subjects": 36,         # 受试者/记录数，不知则 None
        "note": "一句话：来源与已知坑",
    }

    def build() -> tuple[np.ndarray, np.ndarray, dict]:
        '''下载（未缓存时）→ 解析 → 按连续段开窗 → 返回 (X, y, meta)。

        X: (N, 128, C) float64/float32，**未标准化**（z-score 由驱动统一做）
        y: (N,) int64，任意整数标签（驱动统一重编号为 0..k-1）
        meta: 补充信息 dict（class_names / n_subjects / 实际解析到的文件数 …）
        '''

**为什么标准化/剔稀类/降采样不在 adapter 里做**：它们是预注册口径
（`results/candidate_survey/PROTOCOL.md` §1），8 个 adapter 各写一遍必然发散。
驱动 `scripts/prepare_candidates.py` 统一处理，adapter 只负责"把原始格式读对"。

开窗必须**按连续段**调用 `window_stream`（同一受试者的同一次记录为一段）：跨段拼接
会造出物理上不存在的过渡窗。
"""
