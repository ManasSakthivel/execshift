"""Stage 1: simulator validation (stylised facts) -> results/simulator_validation.json + Fig. 1."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
from scipy import stats

from .. import paths
from ..config import IMPACT_MODELS, NOMINAL
from ..market.validation import run_checks
from ..reporting.style import COLOR, INK2, PALETTE, plt, setup


def main(argv=None) -> int:
    out, arts = {}, {}
    for im in IMPACT_MODELS:
        checks, art = run_checks(replace(NOMINAL, impact_model=im))
        arts[im] = art
        out[im] = [{"check": c.name, "value": round(float(c.value), 4), "target_low": c.lo,
                    "target_high": c.hi, "status": c.status, "source": c.source} for c in checks]
    out["not_modelled"] = [
        "Intraday U-shaped volume/volatility seasonality (single 30-min window, no time of day)",
        "Leverage effect (return-volatility asymmetry)",
        "Long memory of volatility beyond ~30 min (log-vol is AR(1): exponential, not power-law, "
        "decay)",
        "Discrete tick size, queue dynamics and limit-order fills (execution is by marketable "
        "orders against a parametric cost model)",
        "Reaction of other market participants to our orders beyond the impact model",
    ]
    out["kurtosis_by_aggregation"] = arts["sqrt"]["kurtosis_by_agg"]
    paths.RESULTS.mkdir(exist_ok=True)
    (paths.RESULTS / "simulator_validation.json").write_text(json.dumps(out, indent=1))
    n_fail = sum(c["status"] == "FAIL" for im in IMPACT_MODELS for c in out[im])
    print(f"simulator validation: {n_fail} FAIL across {len(IMPACT_MODELS)} impact models")
    figure(arts)
    return 1 if n_fail else 0


def figure(arts):
    setup()
    art = arts["sqrt"]
    m = art["series"]
    r = (m.ret[:, 1:] / m.sigma_nom[:, None]).ravel()
    r = r / r.std()
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.0))
    bins = np.linspace(-10, 10, 121)
    h, e = np.histogram(r, bins=bins, density=True)
    c = 0.5 * (e[1:] + e[:-1])
    ax[0].semilogy(c, np.where(h > 0, h, np.nan), color=COLOR["RL"], label="simulated 30 s")
    ax[0].semilogy(c, stats.norm.pdf(c), color=INK2, lw=1.2, ls="--", label="Gaussian")
    ax[0].set(title="(a) Heavy-tailed returns", xlabel="standardised return",
              ylabel="density (log)", ylim=(1e-6, 1))
    ax[0].legend(loc="lower center")
    lags = art["lags"]
    ax[1].plot(lags, art["acf_abs"], color=PALETTE[1], label="ACF |r|")
    ax[1].plot(lags, art["acf_r"], color=PALETTE[0], label="ACF r")
    ax[1].axhline(0, color=INK2, lw=0.6)
    ax[1].set(title="(b) Volatility clustering", xlabel="lag (30 s bins)", ylabel="autocorrelation")
    ax[1].legend()
    k = art["kurtosis_by_agg"]
    ax[2].bar(["30 s", "5 min", "30 min"], [k[1], k[10], k[60]], color=COLOR["RL"], width=0.5)
    ax[2].set(title="(c) Aggregational Gaussianity", ylabel="excess kurtosis")
    for im in IMPACT_MODELS:
        a = arts[im]
        ax[3].loglog(a["impact_parts"], a["impact_cost"], marker="o", color=COLOR[im], label=im)
    p = arts["sqrt"]["impact_parts"]
    ref = arts["sqrt"]["impact_cost"][2] * np.sqrt(p / p[2])
    ax[3].loglog(p, ref, color=INK2, ls="--", lw=1.0, label="slope 1/2")
    ax[3].set(title="(d) Metaorder impact", xlabel="participation rate",
              ylabel="impact cost (bp)")
    ax[3].set_xticks(p, [f"{x:.0%}" for x in p])
    ax[3].set_yticks([2, 5, 10, 20], ["2", "5", "10", "20"])
    ax[3].minorticks_off()
    ax[3].legend()
    fig.tight_layout()
    paths.FIGURES.mkdir(exist_ok=True)
    fig.savefig(paths.FIGURES / "fig1_simulator_validation.png")
