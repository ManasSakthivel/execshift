# FAQ

**Does the uncertainty gate work?**
Not in any practically useful sense. It lowers deployment-mixture cost relative to always
trusting RL by -0.013 [-0.024, -0.004] bp, against a pre-registered 0.25 bp threshold. That
makes H4, the central claim, not supported ([`RESULTS.md`](RESULTS.md) §4).

**Then why is H3 marked "supported"?**
H3's pre-registered rule only asks whether the score predicts regret better than chance:
the Spearman ρ CI must exclude 0 and the AUROC CI must exceed 0.5. With 160,000
episode-replicates, ρ = +0.108 and AUROC 0.542 clear that bar. The practical test is H4.

**Is this a failure of tuning, or a real finding?**
A real finding, with a mechanism. A gate that can only tell market conditions apart is
bounded by the *headroom*, the cost RL incurs in the conditions where it loses to TWAP. Under
the primary impact model that is 0.065 bp. The costliest hidden failure, alpha decay,
leaves no trace in pre-trade data. All of this was measured, not assumed
([`METHODOLOGY.md`](METHODOLOGY.md) §12).

**Could a better gate have succeeded?**
Possibly. An unachievable per-order oracle beats always-RL by 0.31 bp, so the threshold is
reachable by a gate that discriminates *within* conditions. The measured within-condition
signal is weak (ρ ≤ +0.103). [`FUTURE_WORK.md`](FUTURE_WORK.md) lists the directions:
direct regret prediction and mid-order intervention.

**Is any of this evidence about real markets?**
No. Every market is synthetic, from one parametric simulator. Nothing here shows real-market
performance, profitability or deployability.

**Why is the package called `adexec` and not `execshift`?**
The pre-registration lock records the frozen files under `adexec/`. Renaming the package
would change every locked path and count as a protocol deviation.

**Why are some files never reformatted or type-checked?**
They are byte-for-byte hash-locked. See [`DEVELOPMENT.md`](DEVELOPMENT.md) §1.

**Can I reproduce the study without 76 minutes of compute?**
`make check` verifies the committed results, their hashes and the derived documents in under
a minute. `make reproduce-paper-fast` runs every stage at toy scale in about a minute. Only
`make reproduce-paper` retrains the 150 PPO members and regenerates the confirmatory outputs
([`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)).

**Why are the benchmark results labelled exploratory?**
They come from a separate episode stream (seed tag 66) created after the study. They can
probe new scenarios, but they cannot change a pre-registered verdict.

**What is the relationship to `rl-market-maker`?**
This repository began as a local clone of that unlicensed market-making project. None of its
code is included. ExecShift is an independent codebase on an orphan branch
([`../PROVENANCE.md`](../PROVENANCE.md)).
