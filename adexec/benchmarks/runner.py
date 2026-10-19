"""Execute a ``BenchmarkSuite`` and record it as a registered experiment run.

For every scenario and seed the runner generates one batch of episodes from the post-study
stream and rolls out every policy on that same batch (common random numbers). It then
computes, per (scenario, policy):

* the suite's sample metrics on **realised** IS (risk metrics describe what an order
  actually pays), plus the mean of the control-variate IS (``mean_cv``, lower variance);
* a paired comparison against the reference policy on the control-variate IS: estimate,
  bootstrap CI, p-value, d_z and a practical verdict against the threshold, plus the
  realised-IS estimate and the paired win rate;
* any diagnostic a policy exposes as ``fallback_rate`` (the gated policies), read with
  ``getattr``, so the runner needs no policy-specific code.

Policies are built per impact model, because tuned baselines and PPO checkpoints are
impact-specific. A policy whose checkpoints are missing is skipped and recorded under
``skipped`` (or raises, with ``skip_unavailable=False``). The reference policy is never
skipped.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..agents import PolicyContext, make_policy
from ..evaluation.engine import evaluate_policies
from ..evaluation.metrics import METRICS, win_rate
from ..evaluation.statistics import paired_comparison
from ..market import DEFAULT_SIMULATOR, benchmark_rngs
from .suite import BenchmarkSuite

POST_STUDY_LABEL = "POST-STUDY / EXPLORATORY"


@dataclass
class BenchmarkResult:
    suite: dict[str, Any]
    label: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    comparisons: list[dict[str, Any]] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    episodes: dict[str, np.ndarray] = field(default_factory=dict)
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "suite": self.suite, "rows": self.rows,
                "comparisons": self.comparisons, "skipped": self.skipped,
                "seconds": round(self.seconds, 2)}

    def row(self, scenario: str, policy: str) -> dict[str, Any]:
        for r in self.rows:
            if r["scenario"] == scenario and r["policy"] == policy:
                return r
        raise KeyError((scenario, policy))

    def headline(self) -> dict[str, Any]:
        """Compact metrics for the manifest: mean_cv per scenario/policy."""
        return {f"{r['scenario']}|{r['policy']}": round(r["mean_cv"], 4) for r in self.rows}


def _build_policies(suite: BenchmarkSuite, ctx: PolicyContext, skip_unavailable: bool,
                    skipped: dict[str, str]) -> dict[str, Any]:
    pols = {}
    for name, spec in suite.policies.items():
        try:
            pol = make_policy(spec, ctx)
        except FileNotFoundError as e:
            if not skip_unavailable or name == suite.reference:
                raise
            skipped[name] = str(e).split(". ")[0]
            continue
        pol.name = name
        pols[name] = pol
    return pols


def run_suite(suite: BenchmarkSuite, *, n_episodes: int | None = None, seeds=None,
              models_dir: Path | None = None, baselines_dir: Path | None = None,
              skip_unavailable: bool = True, simulator=DEFAULT_SIMULATOR,
              progress: Callable[[str], None] | None = None) -> BenchmarkResult:
    n = int(n_episodes or suite.n_episodes)
    seeds = tuple(int(s) for s in (seeds if seeds is not None else suite.seeds))
    stat = suite.statistics
    t0 = time.time()
    res = BenchmarkResult(suite=suite.to_dict() | {"n_episodes": n, "seeds": list(seeds)},
                          label=POST_STUDY_LABEL)
    cache: dict[str, dict[str, Any]] = {}
    for scen in suite.scenarios:
        im = scen.impact_model
        if im not in cache:
            ctx = PolicyContext(impact_model=im)
            if models_dir is not None:
                ctx.models_dir = Path(models_dir)
            if baselines_dir is not None:
                ctx.baselines_dir = Path(baselines_dir)
            cache[im] = _build_policies(suite, ctx, skip_unavailable, res.skipped)
        pols = cache[im]
        is_r: dict[str, list[np.ndarray]] = {k: [] for k in pols}
        is_c: dict[str, list[np.ndarray]] = {k: [] for k in pols}
        fb_rate: dict[str, list[float]] = {}
        for seed in seeds:
            market = simulator.generate(scen.params(), benchmark_rngs(scen, n, seed))
            for name, pol in pols.items():
                run = evaluate_policies(market, {name: pol}, simulator)[name]
                is_r[name].append(run.is_bps)
                is_c[name].append(run.is_cv_bps)
                rate = getattr(pol, "fallback_rate", None)  # optional diagnostic hook
                if rate is not None and np.isfinite(rate):
                    fb_rate.setdefault(name, []).append(float(rate))
        if progress:
            progress(f"{scen.name}: {len(pols)} policies x {n * len(seeds)} episodes")
        R = {k: np.concatenate(v) for k, v in is_r.items()}
        C = {k: np.concatenate(v) for k, v in is_c.items()}
        for name in pols:
            res.episodes[f"{scen.name}|{name}|IS"] = R[name]
            res.episodes[f"{scen.name}|{name}|IScv"] = C[name]
            row = {"scenario": scen.name, "condition": scen.condition, "family": scen.family,
                   "impact_model": im, "is_preregistered_scenario": scen.is_preregistered,
                   "policy": name, "n": int(R[name].size),
                   "mean_cv": float(C[name].mean()),
                   "metrics": {m: METRICS[m](R[name]) for m in suite.metrics}}
            if name in fb_rate:
                row["fallback_rate"] = float(np.mean(fb_rate[name]))
            res.rows.append(row)
            if name == suite.reference:
                continue
            ref = suite.reference
            cmp_ = paired_comparison(C[name], C[ref], (name, ref), stat.threshold_bp,
                                     stat.threshold_source, stat.bootstrap_B,
                                     stat.bootstrap_seed)
            d = cmp_.to_dict()
            d.update(scenario=scen.name, impact_model=im,
                     realised_estimate=float((R[name] - R[ref]).mean()),
                     win_rate=win_rate(C[name], C[ref]))
            res.comparisons.append(d)
    res.seconds = time.time() - t0
    return res


# --------------------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------------------


def record_run(result: BenchmarkResult, *, name: str, config: Mapping[str, Any],
               config_hash: str, kind: str = "benchmark", store=None,
               config_text: str | None = None, figures: bool = True) -> Path:
    """Write results, per-episode arrays, figures and the manifest; return the run dir."""
    from ..experiments.registry import FileArtifactStore, build_manifest, make_experiment_id

    store = store or FileArtifactStore()
    exp_id = make_experiment_id(name, config_hash)
    d = store.create_run(exp_id)
    if config_text is not None:
        (d / "config.toml").write_text(config_text)
    (d / "config.json").write_text(json.dumps(config, indent=1, sort_keys=True) + "\n")
    (d / "results.json").write_text(json.dumps(result.to_dict(), indent=1) + "\n")
    np.savez_compressed(d / "episodes.npz", **result.episodes)  # type: ignore[arg-type]
    if figures:
        from ..reporting.benchmark_figures import make_benchmark_figures

        make_benchmark_figures(result.to_dict(), d / "figures")
    from ..reporting.benchmark_report import render_benchmark_report

    (d / "REPORT.md").write_text(render_benchmark_report(result.to_dict(), exp_id))
    suite = result.suite
    man = build_manifest(
        experiment_id=exp_id, label=result.label, kind=kind, config=config,
        config_hash=config_hash, seeds=suite["seeds"],
        policies=[{"name": k, "spec": v} for k, v in suite["policies"].items()
                  if k not in result.skipped],
        impact_model=sorted({s["impact_model"] for s in suite["scenarios"]}),
        scenarios=[s["name"] for s in suite["scenarios"]], metrics=result.headline(),
        run_dir=d, extra={"suite": suite["name"], "skipped_policies": result.skipped,
                          "seconds": round(result.seconds, 2),
                          "episode_stream": "SeedSequence([66, condition_index, seed, i])"})
    store.write_manifest(d, man)
    return d


def suite_from_config(cfg) -> BenchmarkSuite:
    """An ``ExperimentConfig`` -> suite: a named suite (``[benchmark] suite = ...``) with
    optional ``n_episodes`` / ``seeds`` overrides, or an ad-hoc suite from ``[market]``,
    ``[[policies]]`` and ``[evaluation]``."""
    from dataclasses import replace

    from .catalog import get_suite
    from .suite import StatisticalProcedure, named_policies

    ev = cfg.evaluation
    stat = StatisticalProcedure(bootstrap_B=ev.bootstrap_B, threshold_bp=ev.threshold_bp,
                                threshold_source=ev.threshold_source)
    if cfg.benchmark.get("suite"):
        s = get_suite(cfg.benchmark["suite"])
        kw: dict[str, Any] = {"statistics": stat}
        if "n_episodes" in cfg.benchmark:
            kw["n_episodes"] = int(cfg.benchmark["n_episodes"])
        if "seeds" in cfg.benchmark:
            kw["seeds"] = tuple(int(x) for x in cfg.benchmark["seeds"])
        return replace(s, **kw)
    return BenchmarkSuite(cfg.name, "custom", cfg.description or cfg.name,
                          named_policies(cfg.policies), tuple(cfg.market.scenarios()),
                          seeds=tuple(ev.seeds), n_episodes=ev.n_episodes,
                          metrics=tuple(ev.metrics), reference=ev.reference,
                          statistics=stat)


__all__ = ["BenchmarkResult", "record_run", "run_suite", "suite_from_config"]
