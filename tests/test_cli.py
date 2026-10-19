"""CLI smoke tests (in-process, through ``adexec.cli.main``)."""

import json
import subprocess
import sys

import pytest

from adexec.cli import build_parser, main
from adexec.paths import ROOT

COMMANDS = ("benchmark", "evaluate", "train", "reproduce", "report", "inspect")


def test_top_level_help(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert all(c in out for c in COMMANDS)


@pytest.mark.parametrize("cmd", COMMANDS)
def test_subcommand_help(cmd, capsys):
    with pytest.raises(SystemExit) as e:
        main([cmd, "--help"])
    assert e.value.code == 0 and "usage:" in capsys.readouterr().out


def test_no_command_prints_help(capsys):
    assert main([]) == 2


def test_benchmark_list(capsys):
    assert main(["benchmark", "--list"]) == 0
    assert "robustness" in capsys.readouterr().out


def test_benchmark_inspect_report_evaluate_cycle(runs_dir, capsys):
    assert main(["benchmark", "--suite", "smoke", "--episodes", "40", "--quiet"]) == 0
    run_id = capsys.readouterr().out.strip().splitlines()[-1]
    assert (runs_dir / run_id / "manifest.json").exists()
    assert main(["inspect"]) == 0 and run_id in capsys.readouterr().out
    assert main(["inspect", run_id[:14]]) == 0
    assert "all intact" in capsys.readouterr().out
    assert main(["report", "--experiment", run_id]) == 0
    assert main(["evaluate", "--experiment", run_id, "--quiet"]) == 0
    assert "0 different" in capsys.readouterr().out


def test_evaluate_config(runs_dir, tmp_path, capsys):
    cfg = tmp_path / "c.toml"
    cfg.write_text('[experiment]\nname = "cli-test"\n[market]\nconditions = ["ID", "VOL-mod"]\n'
                   '[[policies]]\nkind = "twap"\n[[policies]]\nkind = "ac"\nkappa = 1.0\n'
                   '[evaluation]\nn_episodes = 30\nbootstrap_B = 200\n')
    assert main(["evaluate", "--config", str(cfg), "--quiet", "--no-figures"]) == 0
    run_id = capsys.readouterr().out.strip().splitlines()[-1]
    man = json.loads((runs_dir / run_id / "manifest.json").read_text())
    assert man["config"]["experiment"]["name"] == "cli-test"
    assert (runs_dir / run_id / "config.toml").read_text() == cfg.read_text()


def test_errors_are_reported_not_raised(runs_dir, tmp_path, capsys):
    assert main(["inspect", "no-such-run"]) == 1
    assert main(["benchmark", "--suite", "nope"]) == 1
    bad = tmp_path / "bad.toml"
    bad.write_text('[experiment]\nname = "x"\n[market]\nconditions = ["MOON"]\n')
    assert main(["evaluate", "--config", str(bad)]) == 1
    assert "unknown condition" in capsys.readouterr().err


def test_reproduce_full_requires_confirmation(capsys):
    assert main(["reproduce", "--full"]) == 2
    assert "--yes" in capsys.readouterr().err
    assert main(["reproduce"]) == 2


def test_study_inspection(capsys):
    assert main(["inspect", "--claim", "H4"]) == 0
    e = json.loads(capsys.readouterr().out)
    assert e["verdict"] == "FAIL" and round(e["value"]["estimate"], 3) == -0.013
    assert main(["inspect", "--lock"]) == 0
    assert main(["inspect", "--conditions"]) == 0
    assert "LIQ-sev" in capsys.readouterr().out


def test_reproduce_verify_passes_on_committed_results():
    assert main(["reproduce", "--verify", "--strict"]) == 0


def test_module_entry_point():
    r = subprocess.run([sys.executable, "-m", "adexec", "inspect", "--lock"], cwd=ROOT,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "lock OK" in r.stdout


def test_parser_has_expected_options():
    p = build_parser()
    a = p.parse_args(["benchmark", "--suite", "risk", "--episodes", "10", "--seeds", "0", "1"])
    assert a.suite == "risk" and a.episodes == 10 and a.seeds == [0, 1]
