"""测试公用夹具。"""
import pytest

from autocls.space.cls_space import CLS


@pytest.fixture
def make_cls():
    """构造 CLS 的工厂：默认为最小路径策略（仅 jitter+crop、instance、InfoNCE+cosine），
    可用关键字覆盖任意字段。默认值均为 space.yaml 的合法档位。"""

    def _make(**overrides) -> CLS:
        values = dict(
            resize=0.0, rescale=0.0, jitter=0.3, point_mask=0.0, freq_mask=0.0,
            crop=0.5, order=0,
            emb_jitter=0.0, emb_mask=0.0, norm="none",
            instance=True, temporal=False, cross_scale=False,
            kernel=None, pool=None, adj_neighbor=None,
            loss_type="info_nce", sim_func="cosine", temperature=0.1,
        )
        values.update(overrides)
        return CLS(**values)

    return _make
