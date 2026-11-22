import json

import pytest
import torch

from proxopt import olmo


def test_response_logprob_alignment():
    g = torch.Generator().manual_seed(0); logits = torch.randn(6, 11, generator=g); ids = torch.tensor([3, 1, 4, 1, 5, 9])
    lp = logits.log_softmax(-1); want = lp[2, 1] + lp[3, 5] + lp[4, 9]            # tokens 3, 4, 5
    got, n = olmo.response_logprob(logits, ids, 3)
    assert n == 3 and got == pytest.approx(want.item(), abs=1e-5)


def test_next_token_kl_both_directions():
    g = torch.Generator().manual_seed(0); a = torch.randn(6, 11, generator=g); b = torch.randn(6, 11, generator=g)
    la, lb = a.log_softmax(-1)[2:5], b.log_softmax(-1)[2:5]
    ab, ba = olmo.next_token_kl(a, b, 3)
    assert ab == pytest.approx((la.exp() * (la - lb)).sum(-1).mean().item(), abs=1e-5)
    assert ba == pytest.approx((lb.exp() * (lb - la)).sum(-1).mean().item(), abs=1e-5)
    assert olmo.next_token_kl(a, a, 3) == pytest.approx((0.0, 0.0), abs=1e-6)


def test_summarize_margins():
    rows = [dict(pair=0, n_w=2, n_l=4, lp_w_sft=-4, lp_l_sft=-4, lp_w_dpo=-2, lp_l_dpo=-8, lp_w_instruct=-4, lp_l_instruct=-4),  # noqa: C408
            dict(pair=1, n_w=2, n_l=2, lp_w_sft=-4, lp_l_sft=-4, lp_w_dpo=-6, lp_l_dpo=-4, lp_w_instruct=-3, lp_l_instruct=-4)]  # noqa: C408
    s = olmo.summarize_margins(rows)
    assert s["n"] == 2
    assert s["dpo"]["sum"]["mean_margin"] == pytest.approx((6 + -2) / 2) and s["dpo"]["sum"]["share_positive"] == 0.5
    assert s["dpo"]["mean"]["mean_margin"] == pytest.approx(((1 - (-1)) + (-1 - 0)) / 2)
    assert s["instruct"]["sum"]["share_positive"] == 0.5


def test_implicit_margins_per_pair():
    rows = [dict(pair=0, n_w=2, n_l=4, lp_w_sft=-4, lp_l_sft=-4, lp_w_dpo=-2, lp_l_dpo=-8, lp_w_instruct=-4, lp_l_instruct=-4),  # noqa: C408
            dict(pair=1, n_w=2, n_l=2, lp_w_sft=-4, lp_l_sft=-4, lp_w_dpo=-6, lp_l_dpo=-4, lp_w_instruct=-3, lp_l_instruct=-4)]  # noqa: C408
    assert olmo.implicit_margins(rows, "dpo", per_token=False) == pytest.approx([6, -2])
    assert olmo.implicit_margins(rows, "dpo", per_token=True) == pytest.approx([1 - (-1), -1 - 0])
    assert olmo.implicit_margins(rows, "instruct", per_token=False) == pytest.approx([0, 1])


def test_loaders_return_none_when_files_are_missing(tmp_path):                    # Review Focus 5
    assert olmo.load_margins(tmp_path) is None and olmo.load_drift(tmp_path) is None and olmo.load_meta(tmp_path) is None
    (tmp_path / "margins.jsonl").write_text(json.dumps({"pair": 0}) + "\n")
    assert olmo.load_margins(tmp_path) == [{"pair": 0}] and olmo.load_drift(tmp_path) is None


def test_default_root_honours_the_environment(tmp_path, monkeypatch):
    (tmp_path / "meta.json").write_text(json.dumps({"n": 1}))
    monkeypatch.setenv("PROXOPT_OLMO_DIR", str(tmp_path))
    assert olmo.load_meta() == {"n": 1} and olmo.load_margins() is None
