"""Compare a regenerated authoritative.json with the committed one.

Tolerance (pre-declared): every reported point estimate and CI bound must agree within
  max(0.02 bp, 10% of the committed CI half-width)
and every hypothesis decision (supported / reject) must be identical.  On the same
platform the pipeline is bit-for-bit deterministic, so differences there indicate a
reproduction failure; across platforms floating-point differences in PPO training can move
estimates slightly, which the tolerance is designed to absorb.

usage: python scripts/verify_reproduction.py <reference.json> <regenerated.json>
"""

import json
import sys

ABS_TOL = 0.02


def walk(a, b, path=""):
    """Yield (path, ref, new, ci_halfwidth) for comparable numeric leaves."""
    if isinstance(a, dict):
        hw = None
        if "ci_low" in a and "ci_high" in a:
            hw = (a["ci_high"] - a["ci_low"]) / 2
        for k in a:
            if k not in b:
                yield f"{path}/{k}", a[k], None, hw
                continue
            if isinstance(a[k], (dict, list)):
                yield from walk(a[k], b[k], f"{path}/{k}")
            else:
                yield f"{path}/{k}", a[k], b[k], hw
    elif isinstance(a, list):
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            yield from walk(x, y, f"{path}[{i}]") if isinstance(x, (dict, list)) else iter(
                [(f"{path}[{i}]", x, y, None)])


def main(ref_path, new_path):
    ref, new = json.loads(open(ref_path).read()), json.loads(open(new_path).read())
    fails, checked = [], 0
    for path, a, b, hw in walk(ref, new):
        leaf = path.rsplit("/", 1)[-1]
        if b is None:
            fails.append(f"missing {path}")
        elif isinstance(a, bool) or leaf in ("supported", "reject"):
            checked += 1
            if a != b:
                fails.append(f"decision differs {path}: {a} -> {b}")
        elif leaf in ("estimate", "ci_low", "ci_high") and isinstance(a, (int, float)):
            checked += 1
            tol = max(ABS_TOL, 0.1 * hw) if hw else ABS_TOL
            if abs(a - b) > tol:
                fails.append(f"{path}: {a:.4f} -> {b:.4f} (tol {tol:.4f})")
    print(f"checked {checked} values / decisions; {len(fails)} outside tolerance")
    for f in fails[:50]:
        print("  ", f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
