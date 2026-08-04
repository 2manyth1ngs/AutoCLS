"""解耦实验随机 CLS 目录生成的单测（铁律 5 先测后写）。

这里守的是采样端的"无人为挑选"性质——预注册（results/decouple_random_cls/
PREREGISTRATION.md §2）的全部意义就在采样只由 (seed, n) 决定：

1. **确定性**：同 (seed, n) 两次生成逐字相同——目录可由任何人复现，不存在
   "重采到满意为止"的自由度。
2. **前缀稳定**：加大 n 不改变前 n' 条（逐个接受制，与 labeling._sample_unique
   同语义）——将来扩样本不作废既有 (task, cls) → R' 缓存（铁律 3）。
3. **合法性**：每条都过 space.validate()（门控字段无效档正确），key 与
   labeling.cls_key 一致（缓存键同构）。
4. **不携带性能信息**：目录条目不得含 har_rank / acc 之类字段——随机目录若带
   排名字段，等于又给了下游"按好坏取前 k 条"的口子，探针挑选循环性原样回归。
5. **钉死已提交目录**：results/ 里的目录必须等于当前 space.yaml 下的再生成结果
   ——space.yaml 变了目录还留着旧样本，实验就在测一个不存在的空间。
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autocls.labeling import cls_key  # noqa: E402
from autocls.space.cls_space import CLS, CLSSpace  # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location(
        "sample_decouple_mod", ROOT / "scripts" / "sample_random_cls_catalog.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sample_decouple_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load()
SPACE = CLSSpace.from_yaml()


def test_deterministic():
    a = M.build_catalog(SPACE, n=20, seed=20260804)
    b = M.build_catalog(SPACE, n=20, seed=20260804)
    assert a["cls"] == b["cls"]


def test_prefix_stable():
    small = M.build_catalog(SPACE, n=10, seed=20260804)
    big = M.build_catalog(SPACE, n=20, seed=20260804)
    assert big["cls"][:10] == small["cls"]


def test_entries_valid_and_keys_consistent():
    cat = M.build_catalog(SPACE, n=20, seed=20260804)
    keys = [e["cls_key"] for e in cat["cls"]]
    assert len(keys) == 20 and len(set(keys)) == 20
    for e in cat["cls"]:
        c = CLS(**e["cls"])
        assert SPACE.validate(c)
        assert e["cls_key"] == cls_key(c)


def test_no_performance_fields():
    cat = M.build_catalog(SPACE, n=20, seed=20260804)
    for e in cat["cls"]:
        assert set(e) == {"cls_key", "cls"}, f"目录条目只许 cls_key+cls，实际 {set(e)}"


def test_committed_catalog_matches_regeneration():
    p = ROOT / "results" / "decouple_random_cls" / "random_cls_catalog.json"
    if not p.exists():  # 首次生成前允许缺席；提交后本测试钉死它
        return
    committed = json.loads(p.read_text(encoding="utf-8"))
    regen = M.build_catalog(SPACE, n=committed["n"], seed=committed["seed"])
    assert committed["cls"] == regen["cls"]
