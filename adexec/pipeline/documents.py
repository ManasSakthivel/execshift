"""Stage 7: render every generated document from results/*.json."""

from __future__ import annotations

import json

from .. import paths
from ..reporting.sim_report import render_baselines, render_sim_validation
from ..reporting.study_report import render_readiness, render_report
from ..reporting.tables import render_tables


def main(argv=None) -> int:
    res = paths.RESULTS
    sv = json.loads((res / "simulator_validation.json").read_text())
    (paths.DOCS / "SIMULATOR_VALIDATION.md").write_text(render_sim_validation(sv))
    bl = {im: json.loads((res / f"baselines_{im}.json").read_text())
          for im in ("sqrt", "linear", "transient")}
    (paths.DOCS / "BASELINES.md").write_text(render_baselines(bl))
    auth = res / "authoritative.json"
    if auth.exists():
        a = json.loads(auth.read_text())
        (res / "tables.md").write_text(render_tables(a))
        (paths.DOCS / "FINAL_RESEARCH_REPORT.md").write_text(render_report(a))
        (paths.ROOT / "FINAL_STANFORD_READINESS_REPORT.md").write_text(render_readiness(a))
    print("documents rendered")
    return 0
