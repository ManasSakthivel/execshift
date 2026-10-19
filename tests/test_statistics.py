"""Statistical layer: paired vs unpaired designs, practical verdicts, the frozen SESOI."""

import json

import numpy as np
import pytest

from adexec import stats as frozen_stats
from adexec.analysis import SESOI
from adexec.evaluation import statistics as S
from adexec.paths import RESULTS


def test_preregistered_threshold_is_unchanged():
    assert SESOI == 0.25 == S.PREREGISTERED_SESOI_BP


@pytest.mark.parametrize("est,lo,hi,verdict", [
    (-0.50, -0.70, -0.30, "a practically better"),
    (0.50, 0.30, 0.70, "b practically better"),
    (-0.013, -0.024, -0.004, "significant but below threshold"),  # the study's H4
    (0.01, -0.10, 0.12, "practically equivalent"),
    (-0.20, -0.60, 0.20, "inconclusive"),
])
def test_practical_verdict(est, lo, hi, verdict):
    assert S.practical_verdict(est, lo, hi, 0.25) == verdict


def test_study_h4_is_significant_but_below_threshold():
    a = json.loads((RESULTS / "authoritative.json").read_text())
    h4 = a["primary"]["H45"]["H4"]["comparison"]
    assert S.verdict_for_study_result(h4) == "significant but below threshold"
    assert a["primary"]["H45"]["H4"]["supported"] is False


def test_paired_is_more_powerful_than_unpaired_under_common_random_numbers():
    rng = np.random.default_rng(0)
    common = rng.normal(0, 10, 2000)  # market noise shared by both policies
    a, b = common + 0.3 + rng.normal(0, 0.5, 2000), common + rng.normal(0, 0.5, 2000)
    p = S.paired_comparison(a, b, ("a", "b"), B=2000)
    u = S.unpaired_comparison(a, b, ("a", "b"), B=2000)
    assert p.design == "paired" and u.design == "unpaired"
    assert (p.ci_high - p.ci_low) < 0.2 * (u.ci_high - u.ci_low)
    assert p.significant and not u.significant
    assert p.verdict == "b practically better"


def test_paired_comparison_rejects_2d_input():
    with pytest.raises(ValueError, match="crossed_comparison"):
        S.paired_comparison(np.zeros((2, 3)), np.zeros((2, 3)))


def test_every_comparison_reports_effect_size_and_threshold():
    rng = np.random.default_rng(1)
    c = S.paired_comparison(rng.normal(size=300), rng.normal(size=300), B=500)
    d = c.to_dict()
    for k in ("estimate", "ci_low", "ci_high", "p_value", "effect_size", "threshold_bp",
              "threshold_source", "verdict", "significant"):
        assert k in d
    assert "pre-registered" in d["threshold_source"]


def test_crossed_comparison_delegates_to_frozen_bootstrap():
    rng = np.random.default_rng(2)
    diffs = {"ID": rng.normal(-0.1, 1, (4, 300)), "X": rng.normal(0.2, 1, (4, 200))}
    w = {"ID": 0.5, "X": 0.5}
    c = S.crossed_comparison(diffs, w, B=500)
    r = frozen_stats.crossed_bootstrap(diffs, w, B=500)
    assert (c.estimate, c.ci_low, c.ci_high, c.p_value) == (
        r["estimate"], r["ci_low"], r["ci_high"], r["p_value"])


def test_holm_is_the_frozen_procedure():
    p = {"a": 0.01, "b": 0.04, "c": 0.03}
    assert S.holm(p) == frozen_stats.holm(p)
    assert S.holm(p)["a"]["p_holm"] == pytest.approx(0.03)
