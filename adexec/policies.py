"""Classical execution policies (all newly implemented in this project).

Every policy maps the execution state to the fraction of the original order X to sell in
the current bin.  Policies only read *observable* information through ``ex.obs()`` plus
the time index; none of them can see the simulator's latent state.
"""

from __future__ import annotations

import numpy as np

from .config import N_STEPS

ACTION_LOG_RANGE = np.log(4.0)  # multipliers of the TWAP-on-remaining pace lie in [1/4, 4]


def twap_pace(ex) -> np.ndarray:
    return ex.q / (N_STEPS - ex.t)


def pace_multiplier(ex, a: np.ndarray) -> np.ndarray:
    """Map a in [-1, 1] to a sell quantity: multiplier exp(a ln 4) of the remaining-TWAP pace."""
    return twap_pace(ex) * np.exp(ACTION_LOG_RANGE * np.clip(a, -1.0, 1.0))


class TWAP:
    name = "TWAP"

    def __call__(self, ex):
        return twap_pace(ex)


class AlmgrenChriss:
    """Static Almgren-Chriss (2000) trajectory q(t) = sinh(k(T-t)) / sinh(kT).

    ``kappa`` (per horizon) is the urgency; kappa -> 0 recovers TWAP and kappa < 0 gives the
    mirrored, back-loaded schedule.  The schedule is static (open-loop), which is what makes it a
    robust fallback: it does not consume any market feature that could shift.
    """

    def __init__(self, kappa: float):
        self.kappa = float(kappa)
        self.name = f"AC(kappa={kappa:g})"
        tau = np.arange(N_STEPS + 1) / N_STEPS
        if abs(kappa) < 1e-9:
            traj = 1.0 - tau
        elif kappa > 0:
            traj = np.sinh(kappa * (1 - tau)) / np.sinh(kappa)
        else:  # mirrored (back-loaded) trajectory
            traj = 1.0 - np.sinh(-kappa * tau) / np.sinh(-kappa)
        self.schedule = -np.diff(traj)

    def __call__(self, ex):
        return np.full(ex.q.shape, self.schedule[ex.t])


class Adaptive:
    """Opportunistic adaptive heuristic (tuned): scales the remaining-TWAP pace by
    exp(-a*signal + b*log_depth - c*log_spread - d*log_rv + e*log_volume), clipped to [1/4, 4].

    This is the strongest feature-driven classical baseline we consider: it uses exactly
    the same observable features as the RL agent, with a hand-specified functional form.
    """

    def __init__(self, a=0.0, b=0.0, c=0.0, d=0.0, e=0.0):
        self.w = np.array([a, b, c, d, e], dtype=float)
        self.name = "ADAPT(" + ",".join(f"{x:g}" for x in self.w) + ")"

    def __call__(self, ex):
        o = ex.obs()
        z = (-self.w[0] * o[:, 2] + self.w[1] * o[:, 5] - self.w[2] * o[:, 4]
             - self.w[3] * o[:, 3] + self.w[4] * o[:, 6])
        z = np.clip(z, -ACTION_LOG_RANGE, ACTION_LOG_RANGE)
        return twap_pace(ex) * np.exp(z)


class Blend:
    """A3 control: per-step convex combination of two policies' sell quantities."""

    def __init__(self, p1, p2, w: float):
        self.p1, self.p2, self.w = p1, p2, float(w)
        self.name = f"BLEND(w={w:g})"

    def reset(self, ex):
        for p in (self.p1, self.p2):
            if hasattr(p, "reset"):
                p.reset(ex)

    def __call__(self, ex):
        return self.w * self.p1(ex) + (1 - self.w) * self.p2(ex)
