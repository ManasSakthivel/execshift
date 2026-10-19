"""Compare a regenerated authoritative.json with the committed reference.

Primary check (pre-declared before the confirmatory run; unchanged)
-------------------------------------------------------------------
Every reported point estimate and CI bound (keys ``estimate``, ``ci_low``, ``ci_high``)
must agree within ``max(0.02 bp, 10% of the committed CI half-width)``. Every hypothesis
decision (``supported`` / ``reject`` and every boolean) must be identical. On the same
platform the pipeline is bit-for-bit deterministic, so differences there indicate a
reproduction failure. Across platforms, floating-point differences in PPO training can move
estimates slightly, and the tolerance is designed to absorb them.

Post-study additions (2026-10-03; they make the check stricter, never looser)
-----------------------------------------------------------------------------
* **List-length mismatches now fail.** The original walker used ``zip(strict=False)`` and
  silently ignored truncated lists.
* **Supplementary coverage report.** The primary check covers 1,063 of ~14,400 numeric
  leaves and does *not* include the H3 point metrics (Spearman rho, AUROC). Every other
  numeric leaf is now compared and summarised (count identical, max absolute difference,
  worst paths). The report is informational and does not change the exit code, because no
  tolerance was pre-declared for these leaves.
* ``--strict``: require every numeric leaf to be exactly equal. This is the right check for
  a same-platform reproduction, where the claim is byte-identity.

usage: python scripts/verify_reproduction.py <reference.json> <regenerated.json> [--strict]
"""

from __future__ import annotations

import argparse
import json
import math

ABS_TOL = 0.02
MISSING = object()
PRIMARY_KEYS = ("estimate", "ci_low", "ci_high")


def walk(a, b, path=""):
    """Yield (path, ref, new, ci_halfwidth) for every leaf of the reference."""
    if isinstance(a, dict):
        hw = None
        if "ci_low" in a and "ci_high" in a:
            hw = (a["ci_high"] - a["ci_low"]) / 2
        for k in a:
            if not isinstance(b, dict) or k not in b:
                yield f"{path}/{k}", a[k], MISSING, hw
                continue
            if isinstance(a[k], (dict, list)):
                yield from walk(a[k], b[k], f"{path}/{k}")
            else:
                yield f"{path}/{k}", a[k], b[k], hw
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            yield f"{path}#len", len(a), (len(b) if isinstance(b, list) else MISSING), None
            return
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            if isinstance(x, (dict, list)):
                yield from walk(x, y, f"{path}[{i}]")
            else:
                yield f"{path}[{i}]", x, y, None


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def compare(ref: dict, new: dict) -> dict:
    fails, checked = [], 0
    supp_n = supp_equal = 0
    supp_max, worst = 0.0, []
    for path, a, b, hw in walk(ref, new):
        leaf = path.rsplit("/", 1)[-1]
        if b is MISSING:
            fails.append(f"missing {path}")
        elif path.endswith("#len"):
            checked += 1
            fails.append(f"list length differs {path[:-4]}: {a} -> {b}")
        elif a is None or b is None:
            checked += 1
            if a is not b:
                fails.append(f"null mismatch {path}: {a} -> {b}")
        elif isinstance(a, bool) or leaf in ("supported", "reject"):
            checked += 1
            if a != b:
                fails.append(f"decision differs {path}: {a} -> {b}")
        elif leaf in PRIMARY_KEYS and _is_num(a):
            checked += 1
            tol = max(ABS_TOL, 0.1 * hw) if hw else ABS_TOL
            if abs(a - b) > tol:
                fails.append(f"{path}: {a:.4f} -> {b:.4f} (tol {tol:.4f})")
        elif _is_num(a) and _is_num(b):
            supp_n += 1
            both_nan = isinstance(a, float) and isinstance(b, float) and math.isnan(a) and \
                math.isnan(b)
            if a == b or both_nan:
                supp_equal += 1
            else:
                d = abs(a - b)
                supp_max = max(supp_max, d)
                worst.append((d, path, a, b))
    worst.sort(reverse=True)
    return {"primary_checked": checked, "primary_failures": fails,
            "supplementary_compared": supp_n, "supplementary_identical": supp_equal,
            "supplementary_max_abs_diff": supp_max,
            "supplementary_worst": [f"{p}: {a!r} -> {b!r}" for _, p, a, b in worst[:20]]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Verify a regenerated authoritative.json "
                                 "against the committed reference.")
    ap.add_argument("reference")
    ap.add_argument("regenerated")
    ap.add_argument("--strict", action="store_true",
                    help="also require every other numeric leaf to be exactly equal "
                         "(same-platform byte-identity)")
    a = ap.parse_args(argv)
    with open(a.reference) as fh:
        ref = json.load(fh)
    with open(a.regenerated) as fh:
        new = json.load(fh)
    r = compare(ref, new)
    fails = r["primary_failures"]
    print(f"checked {r['primary_checked']} values / decisions; {len(fails)} outside tolerance")
    for f in fails[:50]:
        print("  ", f)
    n, eq = r["supplementary_compared"], r["supplementary_identical"]
    print(f"supplementary (informational): {n} other numeric leaves; {eq} identical; "
          f"max |diff| {r['supplementary_max_abs_diff']:.3g}")
    for w in r["supplementary_worst"][:10]:
        print("   ~", w)
    if fails:
        return 1
    if a.strict and eq != n:
        print(f"--strict: {n - eq} numeric leaves differ")
        return 1
    return 0
