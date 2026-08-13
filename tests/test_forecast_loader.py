"""forecast_loader 单测（TS2Vec 官方 datautils.py 逐行对照适配的行为钉死）。

对照源：yuezhihan/ts2vec datautils.py
- _get_time_features (L125-134)：7 个日历协变量
- load_forecast_csv (L137-181)：univar 取 OT、ETTh1/h2 按 12/4/4 月切片、
  StandardScaler 只 fit train 切片、协变量拼在前 n_covariate_cols 列
本项目新增 make_pretrain_instances：cl_trainer 是实例批语义（铁律 2），把 train
切片滑窗成 (N, inst_len, C)；窗口绝不可越出 train 切片（预训练不可见 valid/test）。
"""
import numpy as np
import pandas as pd
import pytest

from autocls.data.forecast_loader import (
    load_forecast_csv,
    make_pretrain_instances,
    time_features,
)


def _write_csv(path, n_rows, freq="h", drift=0.0):
    """合成预测 CSV：date 索引 + 2 个负载列 + OT，可加线性漂移（考 scaler 泄漏）。"""
    idx = pd.date_range("2016-07-01", periods=n_rows, freq=freq)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "date": idx,
            "HUFL": rng.normal(size=n_rows),
            "HULL": rng.normal(size=n_rows),
            "OT": rng.normal(size=n_rows) + drift * np.arange(n_rows),
        }
    )
    df.to_csv(path, index=False)
    return df


class TestTimeFeatures:
    def test_seven_calendar_columns(self):
        # 官方 _get_time_features：minute/hour/dayofweek/day/dayofyear/month/weekofyear
        idx = pd.date_range("2016-07-01 00:00", periods=3, freq="h")
        feats = time_features(idx)
        assert feats.shape == (3, 7)
        # 2016-07-01 是周五（dayofweek=4）、年内第 183 天（2016 闰年）、ISO 第 26 周
        assert feats[0].tolist() == [0.0, 0.0, 4.0, 1.0, 183.0, 7.0, 26.0]
        assert feats[2][1] == 2.0  # 第三行 hour=2

    def test_float_dtype(self):
        idx = pd.date_range("2016-07-01", periods=2, freq="h")
        assert time_features(idx).dtype == np.float64


class TestLoadForecastCsv:
    def test_generic_ratio_slices_and_shapes(self, tmp_path):
        # 非 ETT 名 → 0.6/0.2/0.2 切片（官方 L160-162）
        p = tmp_path / "synth.csv"
        _write_csv(p, 100)
        data, tr, va, te, scaler, pred_lens, n_cov = load_forecast_csv(p)
        assert data.shape == (1, 100, 7 + 3)  # 7 协变量 + 3 数据列
        assert n_cov == 7
        # 官方 L162：test 切片右端是 None（到末尾），不是显式 len
        assert (tr, va, te) == (slice(None, 60), slice(60, 80), slice(80, None))

    def test_univar_takes_last_column_generic(self, tmp_path):
        # 非 ETT 名的 univar 取最后一列（官方 L148 iloc[:, -1:]）
        p = tmp_path / "synth.csv"
        _write_csv(p, 50)
        data, *_ , n_cov = load_forecast_csv(p, univar=True)
        assert data.shape == (1, 50, 7 + 1)
        assert n_cov == 7

    def test_etth1_month_slices_and_univar_ot(self, tmp_path):
        # ETTh1 名 → 12/4/4 月切片（官方 L151-154）；univar 取 OT 列（官方 L143-144）
        p = tmp_path / "ETTh1.csv"
        n = 20 * 30 * 24 + 100
        df = _write_csv(p, n)
        data, tr, va, te, scaler, pred_lens, n_cov = load_forecast_csv(p, univar=True)
        assert (tr, va, te) == (
            slice(None, 12 * 30 * 24),
            slice(12 * 30 * 24, 16 * 30 * 24),
            slice(16 * 30 * 24, 20 * 30 * 24),
        )
        assert pred_lens == [24, 48, 168, 336, 720]  # 官方 L176-177
        # 最后一列 = z 后的 OT：与手工用 train 切片统计量标准化的 OT 一致
        ot = df["OT"].to_numpy()
        mu, sd = ot[: 12 * 30 * 24].mean(), ot[: 12 * 30 * 24].std()
        np.testing.assert_allclose(data[0, :, -1], (ot - mu) / sd, rtol=1e-6)

    def test_scaler_fits_train_slice_only(self, tmp_path):
        # 带漂移的序列：若 scaler 偷看全量，train 段均值会显著偏负
        p = tmp_path / "synth.csv"
        _write_csv(p, 200, drift=0.5)
        data, tr, *_ = load_forecast_csv(p, univar=True)
        train_ot = data[0, tr, -1]
        assert abs(train_ot.mean()) < 1e-9  # train 段自身均值恰为 0 ⇔ 只 fit train
        assert data[0, :, -1].mean() > 1.0  # 全量均值被漂移拉高 ⇔ 没做全量 fit


class TestMakePretrainInstances:
    def test_window_count_content_and_dtype(self, tmp_path):
        p = tmp_path / "synth.csv"
        _write_csv(p, 100)
        data, tr, *_ = load_forecast_csv(p, univar=True)
        inst = make_pretrain_instances(data, tr, inst_len=20, stride=10)
        # T_train=60 → 起点 0,10,20,30,40 → 5 个窗
        assert inst.shape == (5, 20, 8)
        assert inst.dtype == np.float32
        np.testing.assert_allclose(inst[1], data[0, 10:30], rtol=1e-6)

    def test_windows_never_cross_train_boundary(self, tmp_path):
        # 最后一个窗的终点必须 ≤ train 切片终点（预训练不可见 valid/test，铁律 6 同款）
        p = tmp_path / "synth.csv"
        _write_csv(p, 100, drift=100.0)  # 越界一步就会带进量级悬殊的值，好抓
        data, tr, *_ = load_forecast_csv(p, univar=True)
        inst = make_pretrain_instances(data, tr, inst_len=25, stride=10)
        # T_train=60 → 起点 0,10,20,30 → 4 个窗（起点 40 的窗到 65 越界，必须丢弃）
        assert inst.shape[0] == 4
        # 与 float32 铸型后的 train 段比（inst 本身是 float32，防铸型舍入假阳）
        assert np.abs(inst).max() <= np.abs(data[0, tr].astype(np.float32)).max() + 1e-9

    def test_train_shorter_than_window_raises(self, tmp_path):
        p = tmp_path / "synth.csv"
        _write_csv(p, 50)
        data, tr, *_ = load_forecast_csv(p)
        with pytest.raises(ValueError, match="inst_len"):
            make_pretrain_instances(data, tr, inst_len=64, stride=8)
