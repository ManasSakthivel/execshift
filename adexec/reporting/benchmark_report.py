"""Markdown report for a recorded benchmark run (from ``results.json``)."""

from __future__ import annotations

from typing import Any


def _f(x: float, d: int = 2) -> str:
    return f"{x:+.{d}f}"


def render_benchmark_report(res: dict[str, Any], experiment_id: str) -> str:
    s = res["suite"]
    st = s["statistics"]
    L = [f"# Benchmark run `{experiment_id}`", "",
         f"> **{res['label']}.** Episodes come from the post-study benchmark stream "
         "(split tag 66). They are not the pre-registered confirmatory test set, and "
         "nothing here changes or extends the study's findings "
         "(`results/authoritative.json`).", "",
         f"**Suite:** `{s['name']}` ({s['category']}). {s['description']}", "",
         f"**Episodes:** {s['n_episodes']} per scenario per seed; seeds {s['seeds']}. "
         f"**Reference policy:** {s['reference']}.", "",
         f"**Statistics:** {st['design']} comparisons on {st['estimator']}, percentile "
         f"bootstrap B = {st['bootstrap_B']} (seed {st['bootstrap_seed']}), practical "
         f"threshold {st['threshold_bp']:g} bp ({st['threshold_source']}). Unit: "
         f"{st['unit']}.", ""]
    for n in s.get("notes", []):
        L += [f"*Note:* {n}", ""]
    if res["skipped"]:
        L += ["**Skipped policies** (artifacts unavailable):", ""]
        L += [f"- `{k}`: {v}" for k, v in res["skipped"].items()]
        L.append("")
    metrics = s["metrics"]
    L += ["## Cost metrics (realised IS, bp; lower is better)", "",
          "| scenario | policy | n | mean (CV) | " + " | ".join(metrics) + " | fallback |",
          "|---|---|---|---|" + "---|" * len(metrics) + "---|"]
    for r in res["rows"]:
        fb = f"{r['fallback_rate']:.1%}" if "fallback_rate" in r else ""
        L.append(f"| {r['scenario']} | {r['policy']} | {r['n']} | {r['mean_cv']:.3f} | "
                 + " | ".join(f"{r['metrics'][m]:.3f}" for m in metrics) + f" | {fb} |")
    L += ["", f"## Paired differences vs {s['reference']} (control-variate IS, bp)", "",
          "Negative favours the policy. *Verdict* combines the CI with the practical "
          "threshold: a CI that excludes zero with |estimate| below the threshold is "
          "reported as `significant but below threshold`, not as a win.", "",
          "| scenario | policy | estimate [95% CI] | p | d_z | win rate | realised est. | "
          "verdict |", "|---|---|---|---|---|---|---|---|"]
    for c in res["comparisons"]:
        L.append(f"| {c['scenario']} | {c['a']} | {_f(c['estimate'], 3)} "
                 f"[{_f(c['ci_low'], 3)}, {_f(c['ci_high'], 3)}] | {c['p_value']:.3g} | "
                 f"{c['effect_size']:+.3f} | {c['win_rate']:.1%} | "
                 f"{_f(c['realised_estimate'], 3)} | {c['verdict']} |")
    L += ["", "Figures: `figures/` in this run directory. Per-episode costs: "
          "`episodes.npz` (keys `<scenario>|<policy>|IS` and `...|IScv`).", ""]
    return "\n".join(L)


__all__ = ["render_benchmark_report"]
