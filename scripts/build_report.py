"""Render every generated document from results/*.json."""

import json

from adexec.guard import ROOT
from adexec.report import render_readiness, render_report
from adexec.report_sim import render_baselines, render_sim_validation
from adexec.report_tables import render_tables


def main():
    res = ROOT / "results"
    sv = json.loads((res / "simulator_validation.json").read_text())
    (ROOT / "docs" / "SIMULATOR_VALIDATION.md").write_text(render_sim_validation(sv))
    bl = {im: json.loads((res / f"baselines_{im}.json").read_text())
          for im in ("sqrt", "linear", "transient")}
    (ROOT / "docs" / "BASELINES.md").write_text(render_baselines(bl))
    auth = res / "authoritative.json"
    if auth.exists():
        a = json.loads(auth.read_text())
        (res / "tables.md").write_text(render_tables(a))
        (ROOT / "docs" / "FINAL_RESEARCH_REPORT.md").write_text(render_report(a))
        (ROOT / "FINAL_STANFORD_READINESS_REPORT.md").write_text(render_readiness(a))
    print("documents rendered")


if __name__ == "__main__":
    main()
