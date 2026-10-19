"""``execshift`` (alias ``quant-exec``): command-line interface to the platform.

Every subcommand calls library functions (``benchmarks``, ``experiments``, ``agents``,
``pipeline``, ``reporting``); none shells out to scripts.

    execshift benchmark --suite robustness            run a standard suite, record a run
    execshift benchmark --list                        list suites
    execshift evaluate  --config configs/examples/evaluate_liquidity.toml
    execshift evaluate  --experiment <id>             re-run a recorded run, compare numbers
    execshift train     --config configs/examples/train_small.toml
    execshift reproduce --fast                        make reproduce-paper-fast
    execshift reproduce --verify                      check results against the reference
    execshift report    --experiment <id>             re-render a run's report and figures
    execshift report    --study                       re-render study docs and figures
    execshift inspect                                 list recorded runs
    execshift inspect   <id>                          manifest + artifact integrity
    execshift inspect   --claim H4                    provenance of a published number
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

PROG = "execshift"


def _store(args):
    from ..experiments.registry import FileArtifactStore

    return FileArtifactStore(Path(args.runs_dir) if args.runs_dir else None)


def _print(obj) -> None:
    print(json.dumps(obj, indent=1, default=str))


def _ok(msg: str) -> None:
    print(msg, file=sys.stderr)


# --------------------------------------------------------------------------------------
# benchmark / evaluate
# --------------------------------------------------------------------------------------


def cmd_benchmark(args) -> int:
    from ..benchmarks import SUITES, get_suite, record_run, run_suite
    from ..experiments.config import hash_config

    if args.list or not args.suite:
        for name, fn in SUITES.items():
            s = fn()
            rl = "needs PPO checkpoints" if s.requires_checkpoints else "no checkpoints"
            print(f"{name:<11} {s.category:<21} {len(s.scenarios):>3} scenarios  "
                  f"{s.n_episodes:>5} ep  ({rl})\n            {s.description}")
        return 0 if args.list else 2
    suite = get_suite(args.suite)
    res = run_suite(suite, n_episodes=args.episodes, seeds=args.seeds,
                    models_dir=args.models_dir, skip_unavailable=not args.strict,
                    progress=None if args.quiet else _ok)
    cfg = res.suite
    d = record_run(res, name=suite.name, config=cfg, config_hash=hash_config(cfg),
                   store=_store(args), figures=not args.no_figures)
    if res.skipped:
        _ok(f"skipped (unavailable): {', '.join(res.skipped)}")
    _ok(f"{res.label}: {len(res.rows)} rows in {res.seconds:.1f} s")
    print(d.name)
    return 0


def cmd_evaluate(args) -> int:
    from ..benchmarks import record_run, run_suite, suite_from_config
    from ..experiments.config import config_from_dict, load_config

    store = _store(args)
    if args.experiment:
        man = store.load_manifest(args.experiment)
        if man["kind"] not in ("evaluate", "benchmark"):
            _ok(f"run {man['experiment_id']} is a {man['kind']} run; nothing to re-evaluate")
            return 2
        old = json.loads((store.run_dir(args.experiment) / "results.json").read_text())
        if "suite" in man.get("extra", {}) and man["kind"] == "benchmark":
            from dataclasses import replace

            from ..benchmarks import get_suite

            s = get_suite(man["extra"]["suite"])
            suite = replace(s, n_episodes=old["suite"]["n_episodes"],
                            seeds=tuple(old["suite"]["seeds"]))
        else:
            suite = suite_from_config(config_from_dict(man["config"]))
        res = run_suite(suite, models_dir=args.models_dir, progress=None if args.quiet else _ok)
        a = {(r["scenario"], r["policy"]): r["mean_cv"] for r in old["rows"]}
        b = {(r["scenario"], r["policy"]): r["mean_cv"] for r in res.rows}
        diff = {f"{k[0]}|{k[1]}": (a[k], b.get(k)) for k in a if b.get(k) != a[k]}
        print(f"re-evaluated {man['experiment_id']}: {len(a)} values, "
              f"{len(a) - len(diff)} identical, {len(diff)} different")
        for k, (x, y) in list(diff.items())[:20]:
            print(f"  {k}: {x} -> {y}")
        return 0 if not diff else 1
    if not args.config:
        _ok("evaluate needs --config <file.toml> or --experiment <id>")
        return 2
    cfg = load_config(args.config)
    if args.episodes:  # record the override in the config (and hence the hash)
        from dataclasses import replace

        cfg = replace(cfg, evaluation=replace(cfg.evaluation, n_episodes=args.episodes))
    suite = suite_from_config(cfg)
    res = run_suite(suite, models_dir=args.models_dir,
                    skip_unavailable=not args.strict, progress=None if args.quiet else _ok)
    d = record_run(res, name=cfg.name, config=cfg.to_dict(), config_hash=cfg.config_hash,
                   kind=cfg.kind if cfg.kind != "train" else "evaluate", store=store,
                   config_text=Path(args.config).read_text(), figures=not args.no_figures)
    print(d.name)
    return 0


# --------------------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------------------


def cmd_train(args) -> int:
    if args.study:
        from ..pipeline.training import main as study_train

        argv = ["--impact", *args.impact]
        if args.workers:
            argv += ["--workers", str(args.workers)]
        return study_train(argv)
    if not args.config:
        _ok("train needs --config <file.toml> (post-study) or --study (the paper's 150 members)")
        return 2
    from ..agents.training import train_from_config
    from ..experiments.config import load_config
    from ..experiments.registry import build_manifest, make_experiment_id

    cfg = load_config(args.config)
    if cfg.kind != "train":
        _ok(f"{args.config}: [experiment] kind is {cfg.kind!r}, expected 'train'")
        return 2
    store = _store(args)
    exp_id = make_experiment_id(cfg.name, cfg.config_hash)
    d = store.create_run(exp_id)
    (d / "config.toml").write_text(Path(args.config).read_text())
    recs = train_from_config(cfg, d / "models", progress=_ok)
    (d / "training.json").write_text(json.dumps(recs, indent=1) + "\n")
    scen = cfg.market.scenarios()[0]
    from .. import paths

    models = d / "models"
    # repository-relative when possible: manifests must not carry user-specific paths
    models_ref = (str(models.relative_to(paths.ROOT)) if models.is_relative_to(paths.ROOT)
                  else f"<runs-dir>/{exp_id}/models")
    man = build_manifest(
        experiment_id=exp_id, label=cfg.label, kind="train", config=cfg.to_dict(),
        config_hash=cfg.config_hash, seeds=list(cfg.training.seeds),
        policies=[{"name": "PPO", "members": len(recs)}], impact_model=scen.impact_model,
        scenarios=[scen.name], metrics={r["path"]: r["weights_sha256"] for r in recs},
        run_dir=d, extra={"policy_spec": {"kind": "ppo", "models_dir": models_ref,
                                          "impact_model": scen.impact_model,
                                          "ensemble_size": len(recs)}})
    store.write_manifest(d, man)
    print(d.name)
    return 0


# --------------------------------------------------------------------------------------
# reproduce
# --------------------------------------------------------------------------------------

STAGES = ("validate-sim", "tune", "train", "val", "confirm", "figures", "report", "verify")


def _stage(name: str, workers: int | None) -> int:
    from .. import paths
    from ..pipeline import (
        baselines,
        confirmatory,
        documents,
        figure_data,
        simulator_validation,
        training,
        validation,
        verify,
    )
    from ..reporting import study_figures

    w = ["--workers", str(workers)] if workers else []
    if name == "validate-sim":
        return simulator_validation.main([])
    if name == "tune":
        return baselines.main([])
    if name == "train":
        return training.main(["--impact", "sqrt", "linear", "transient", *w])
    if name == "val":
        return validation.main(w)
    if name == "confirm":
        return confirmatory.main(w)
    if name == "figures":
        return figure_data.main([]) or study_figures.main([])
    if name == "report":
        return documents.main([])
    if name == "verify":
        return verify.main([str(paths.RESULTS / "authoritative.reference.json"),
                            str(paths.RESULTS / "authoritative.json")])
    raise ValueError(name)


def cmd_reproduce(args) -> int:
    if args.fast:
        from ..pipeline.fast import main as fast

        return fast([])
    if args.verify:
        from .. import paths
        from ..pipeline.verify import main as verify

        argv = [str(paths.RESULTS / "authoritative.reference.json"),
                str(paths.RESULTS / "authoritative.json")]
        return verify(argv + (["--strict"] if args.strict else []))
    if args.stage:
        return _stage(args.stage, args.workers)
    if args.full:
        if not args.yes:
            _ok("`reproduce --full` deletes models/ and results/raw/, retrains 150 PPO members "
                "(~76 min on 14 cores) and appends 780 entries to "
                "results/test_access_log.jsonl. Re-run with --yes to proceed.")
            return 2
        import shutil

        from .. import paths

        for d in (paths.models_dir(), paths.raw_dir()):
            shutil.rmtree(d, ignore_errors=True)
        for s in STAGES:
            _ok(f"== {s}")
            rc = _stage(s, args.workers)
            if rc:
                _ok(f"stage {s} failed (exit {rc})")
                return rc
        return 0
    _ok("choose one of --fast, --full, --verify, --stage")
    return 2


# --------------------------------------------------------------------------------------
# report / inspect
# --------------------------------------------------------------------------------------


def cmd_report(args) -> int:
    if args.study:
        from ..pipeline import documents
        from ..reporting import study_figures, supplementary

        rc = study_figures.main([]) or documents.main([])
        supplementary.make_supplementary_figures()
        return rc
    if not args.experiment:
        _ok("report needs --experiment <id> or --study")
        return 2
    from ..reporting.benchmark_figures import make_benchmark_figures
    from ..reporting.benchmark_report import render_benchmark_report

    store = _store(args)
    exp_id = store.resolve(args.experiment)
    d = store.run_dir(exp_id)
    if not (d / "results.json").exists():
        _ok(f"run {exp_id} has no results.json (training runs have no report)")
        return 2
    res = json.loads((d / "results.json").read_text())
    figs = make_benchmark_figures(res, d / "figures")
    (d / "REPORT.md").write_text(render_benchmark_report(res, exp_id))
    print(d / "REPORT.md")
    for f in figs:
        print(f)
    return 0


def cmd_inspect(args) -> int:
    if args.claim:
        from ..experiments.study import explain

        _print(explain(args.claim))
        return 0
    if args.study:
        from ..experiments.study import study_manifest

        _print(study_manifest())
        return 0
    if args.conditions:
        from ..config import CONDITIONS, NOMINAL

        for c in CONDITIONS:
            diff = {k: v for k, v in c.params.__dict__.items() if v != getattr(NOMINAL, k)}
            print(f"{c.name:<11} {c.family:<8} {c.severity:<9} {diff}")
        return 0
    if args.lock:
        from ..guard import check_lock

        lock = check_lock()
        print(f"pre-registration lock OK (locked {lock['locked_at']}; "
              f"{len(lock['files'])} frozen files)")
        return 0
    store = _store(args)
    if args.experiment:
        man = store.load_manifest(args.experiment)
        integ = store.verify(args.experiment)
        if args.json:
            _print(man | {"integrity": integ})
            return 0
        bad = {k: v for k, v in integ.items() if v != "ok"}
        print(f"experiment   {man['experiment_id']}\nlabel        {man['label']}\n"
              f"kind         {man['kind']}\ncreated      {man['created_utc']}\n"
              f"git          {man['git_commit']} (dirty: {man['git_dirty']})\n"
              f"config hash  {man['config_hash']}\nseeds        {man['seeds']}\n"
              f"impact       {man['impact_model']}\nscenarios    {len(man['scenarios'])}\n"
              f"policies     {[p.get('name') for p in man['policies']]}\n"
              f"python       {man['environment']['python']} "
              f"({man['environment']['platform']} {man['environment']['machine']})\n"
              f"software     adexec {man['software_version']}\n"
              f"artifacts    {len(integ)} recorded, "
              f"{'all intact' if not bad else f'{len(bad)} changed: {bad}'}")
        return 0 if not bad else 1
    runs = store.list_runs()
    if not runs:
        print(f"no runs in {store.root} (create one with `{PROG} benchmark --suite smoke`)")
        return 0
    for r in runs:
        m = store.load_manifest(r)
        print(f"{r}  {m['kind']:<9} {m['label']}")
    return 0


# --------------------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=PROG, description="Robustness benchmarking for RL trade execution under "
        "distribution shift (synthetic markets). All new runs are POST-STUDY / EXPLORATORY; "
        "the pre-registered study is frozen.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n  execshift benchmark --suite smoke\n"
               "  execshift inspect --claim H4\n  execshift reproduce --fast")
    p.add_argument("--runs-dir", help="experiment registry directory (default: runs/, "
                   "or $ADEXEC_RUNS_DIR)")
    sub = p.add_subparsers(dest="command", metavar="COMMAND")

    b = sub.add_parser("benchmark", help="run a standard benchmark suite",
                       description="Run a standard benchmark suite on the post-study "
                       "episode stream and record it as an experiment run.")
    b.add_argument("--suite", help="suite name (see --list)")
    b.add_argument("--list", action="store_true", help="list available suites")
    b.add_argument("--episodes", type=int, help="episodes per scenario per seed (override)")
    b.add_argument("--seeds", type=int, nargs="+", help="benchmark seeds (override)")
    b.add_argument("--models-dir", type=Path, help="PPO checkpoints (default: models/)")
    b.add_argument("--strict", action="store_true",
                   help="fail instead of skipping policies whose checkpoints are missing")
    b.add_argument("--no-figures", action="store_true")
    b.add_argument("--quiet", action="store_true")
    b.set_defaults(func=cmd_benchmark)

    e = sub.add_parser("evaluate", help="evaluate policies from a config, or re-run a run",
                       description="Evaluate the policies and scenarios of an experiment "
                       "config, or re-execute a recorded run and compare its numbers.")
    g = e.add_mutually_exclusive_group()
    g.add_argument("--config", type=Path, help="experiment TOML (kind = evaluate)")
    g.add_argument("--experiment", help="recorded run id (or unique prefix) to re-execute")
    e.add_argument("--episodes", type=int, help="override [evaluation] n_episodes")
    e.add_argument("--models-dir", type=Path)
    e.add_argument("--strict", action="store_true")
    e.add_argument("--no-figures", action="store_true")
    e.add_argument("--quiet", action="store_true")
    e.set_defaults(func=cmd_evaluate)

    t = sub.add_parser("train", help="train PPO members (post-study config or the study)",
                       description="Train PPO members from a config (kind = train) into a "
                       "new run directory, or retrain the study's members with --study.")
    t.add_argument("--config", type=Path)
    t.add_argument("--study", action="store_true",
                   help="train the pre-registered members into models/ (skips existing)")
    t.add_argument("--impact", nargs="+", default=["sqrt"], help="with --study")
    t.add_argument("--workers", type=int)
    t.set_defaults(func=cmd_train)

    r = sub.add_parser("reproduce", help="reproduce the pre-registered study",
                       description="Reproduce the paper. --fast is the reduced sanity "
                       "pipeline (minutes, never touches the test split); --full is the "
                       "whole paper (~76 min on 14 cores); --verify compares "
                       "results/authoritative.json with the committed reference.")
    m = r.add_mutually_exclusive_group()
    m.add_argument("--fast", action="store_true")
    m.add_argument("--full", action="store_true")
    m.add_argument("--verify", action="store_true")
    m.add_argument("--stage", choices=STAGES, help="run a single pipeline stage")
    r.add_argument("--strict", action="store_true", help="with --verify: require byte-equal "
                   "numeric leaves")
    r.add_argument("--yes", action="store_true", help="confirm --full")
    r.add_argument("--workers", type=int)
    r.set_defaults(func=cmd_reproduce)

    rp = sub.add_parser("report", help="render reports and figures",
                        description="Re-render a recorded run's REPORT.md and figures from "
                        "its results.json, or the study's documents and figures from the "
                        "committed artifacts.")
    g2 = rp.add_mutually_exclusive_group()
    g2.add_argument("--experiment", help="run id or unique prefix")
    g2.add_argument("--study", action="store_true")
    rp.set_defaults(func=cmd_report)

    i = sub.add_parser("inspect", help="list runs, show a manifest, trace a study number",
                       description="Without arguments: list recorded runs. With a run id: "
                       "show its manifest and re-hash its artifacts. --claim traces a "
                       "published study number to its source.")
    i.add_argument("experiment", nargs="?", help="run id or unique prefix")
    i.add_argument("--claim", help="study claim: H1 ... H6")
    i.add_argument("--study", action="store_true", help="the study's reconstructed manifest")
    i.add_argument("--conditions", action="store_true",
                   help="the 13 pre-registered conditions")
    i.add_argument("--lock", action="store_true", help="check the pre-registration lock")
    i.add_argument("--json", action="store_true", help="full manifest as JSON")
    i.set_defaults(func=cmd_inspect)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    from ..experiments.config import ConfigError
    from ..experiments.registry import RunNotFoundError

    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    try:
        return int(args.func(args) or 0)
    except (ConfigError, RunNotFoundError, FileNotFoundError, KeyError) as e:
        print(f"{PROG}: error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
