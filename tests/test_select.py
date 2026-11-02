import pandas as pd

from darija_asr.config import apply_overrides
from darija_asr.data.select import eligible, visit_order


def test_overrides_parse_yaml_values():
    cfg = apply_overrides({"select": {"n": 1000}}, ["select.n=30", "select.duration=[2, 20]",
                                                    "run_name=pilot", "agreement.sources=[a,b]"])
    assert cfg["select"] == {"n": 30, "duration": [2, 20]}
    assert cfg["run_name"] == "pilot" and cfg["agreement"]["sources"] == ["a", "b"]


def test_eligible_filters_without_audio():
    meta = pd.DataFrame({"id": ["a", "b", "c", "d"], "channel": ["x"] * 4,
                         "pesq_hyp": [3.2, 2.0, 3.0, 3.5], "text": ["t", "t", "", "t"]})
    out = eligible(meta, {"pesq_min": 2.5}, has_reference=True, exclude={"d"})
    assert list(out["id"]) == ["a"]


def test_visit_order_round_robins_channels():
    rows = pd.DataFrame({
        "_file": ["f1"] * 3 + ["f2"] * 2 + ["f3"],
        "_rg": [0, 1, 2, 0, 1, 0],
        "channel": ["x", "x", "x", "y", "y", "z"],
    })
    order = visit_order(rows, seed=1)
    assert len(order) == 6
    first_round = {rows.set_index(["_file", "_rg"]).loc[k, "channel"] for k in order[:3]}
    assert first_round == {"x", "y", "z"}
