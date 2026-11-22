import pytest

from proxopt.refs import REFS, cite

KEYS = {"williams1992", "schulman2015trpo", "schulman2016gae", "schulman2017ppo", "christiano2017", "ziegler2019",
        "stiennon2020", "ouyang2022", "bradley1952", "rafailov2023", "shao2024", "guo2025", "lambert2024", "olmo2024"}

def test_refs_are_complete():
    assert set(REFS) == KEYS
    for r in REFS.values():
        assert r.authors and r.title and r.venue and r.url.startswith("https://") and 1950 <= r.year <= 2026

def test_cite():
    out = cite("rafailov2023", "shao2024")
    assert out.count("\n- ") + out.startswith("- ") == 2 and REFS["rafailov2023"].title in out
    assert out.index(REFS["rafailov2023"].title) < out.index(REFS["shao2024"].title)
    with pytest.raises(KeyError):
        cite("nope")
