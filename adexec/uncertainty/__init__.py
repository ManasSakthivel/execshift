"""Pre-trade uncertainty estimators behind ``interfaces.UncertaintyEstimator``.

Every estimator delegates to the frozen feature code in ``adexec/features.py``. Its scores
are therefore exactly the corresponding gate-feature columns used in the study
(``tests/test_uncertainty.py`` checks this). The point of the wrappers is that a single
score can be computed, thresholded, benchmarked and plotted without the whole 15-column
feature pipeline.

What the study found about these signals (``results/authoritative.json``, primary model):

* ensemble action disagreement: Spearman rho with RL regret = -0.016 (no signal);
* Mahalanobis OOD distance: rho = +0.027. It fires mostly on volatility shifts, where RL is
  strongest;
* shadow regret: rho = +0.062;
* RCG predicted regret (learned from all features): rho = +0.108, AUROC 0.542.

Higher score means less trust in RL.
"""

from __future__ import annotations

import numpy as np

from ..config import N_EPISODES, NOMINAL
from ..features import (
    OODModel,
    gate_features,
    market_summary,
    policy_uncertainty,
    shadow_market,
)
from ..sim import Market, generate_market, run_policy


def _frozen_ensemble(policy):
    """The frozen ``PPOEnsemble`` behind an ``agents.PPOPolicy`` (or the object itself)."""
    return getattr(policy, "ensemble", policy)


class _TrainQuantileMixin:
    impact_model: str = "sqrt"
    _train_scores: np.ndarray | None = None

    def score(self, market: Market) -> np.ndarray:  # pragma: no cover - overridden
        raise NotImplementedError

    def train_scores(self) -> np.ndarray:
        """Scores on the 4,000 in-distribution TRAIN episodes (the study's reference set)."""
        if self._train_scores is None:
            from dataclasses import replace

            from ..guard import split_rngs

            p = replace(NOMINAL, impact_model=self.impact_model)
            m = generate_market(p, split_rngs("train", "ID", N_EPISODES["train"]["ID"]))
            self._train_scores = self.score(m)
        return self._train_scores

    def train_quantile(self, q: float) -> float:
        return float(np.quantile(self.train_scores(), q))


class EnsembleDisagreement(_TrainQuantileMixin):
    """Std of PPO-member actions at the arrival state (study feature ``u_ens_s0``)."""

    name = "ensemble_action_std"

    def __init__(self, rl_policy, impact_model: str = "sqrt") -> None:
        self.ens = _frozen_ensemble(rl_policy)
        self.impact_model = impact_model

    def score(self, market: Market) -> np.ndarray:
        return policy_uncertainty(market, self.ens)[:, 0]


class OODDistance(_TrainQuantileMixin):
    """Mahalanobis distance of the pre-trade market summary from TRAIN-ID summaries
    (study feature ``ood_maha``)."""

    name = "ood_mahalanobis"

    def __init__(self, impact_model: str = "sqrt", reference: Market | None = None) -> None:
        self.impact_model = impact_model
        if reference is None:
            from dataclasses import replace

            from ..guard import split_rngs

            p = replace(NOMINAL, impact_model=impact_model)
            reference = generate_market(p, split_rngs("train", "ID",
                                                      N_EPISODES["train"]["ID"]))
        self.model = OODModel.fit(market_summary(reference))

    def score(self, market: Market) -> np.ndarray:
        return self.model.score(market_summary(market))


class ShadowRegret(_TrainQuantileMixin):
    """RL minus fallback cost, replayed over the pre-trade window with the trader's nominal
    cost model (study feature ``shadow_regret``)."""

    name = "shadow_regret"

    def __init__(self, rl_policy, fallback, impact_model: str = "sqrt") -> None:
        self.ens = _frozen_ensemble(rl_policy)
        self.fallback = fallback
        self.impact_model = impact_model

    def score(self, market: Market) -> np.ndarray:
        sm = shadow_market(market, self.impact_model)
        is_rl, _ = run_policy(sm, self.ens)
        is_fb, _ = run_policy(sm, self.fallback)
        return is_rl - is_fb


class RCGScore(_TrainQuantileMixin):
    """The study's regret-calibrated gate score: a regressor's predicted counterfactual
    regret (IS_cv,RL - IS_cv,TWAP) from all 15 gate features. Fall back iff score > 0.

    ``regressor`` must already be fitted (on validation data; see ``from_validation``).
    """

    name = "rcg_predicted_regret"

    def __init__(self, regressor, rl_policy, fallback, ood: OODModel,
                 impact_model: str = "sqrt") -> None:
        self.reg, self.ens, self.fallback, self.ood = (regressor, _frozen_ensemble(rl_policy),
                                                       fallback, ood)
        self.impact_model = impact_model

    @classmethod
    def from_validation(cls, val_conds: dict, rl_policy, fallback, impact_model: str = "sqrt",
                        kind: str = "hgb", held_out_family: str | None = None) -> RCGScore:
        """Fit exactly as the study does (``gates.fit_regressor`` with LOFO conditions)."""
        from ..gates import fit_regressor, training_conds

        reg = fit_regressor(val_conds, training_conds(list(val_conds), held_out_family), kind)
        return cls(reg, rl_policy, fallback, OODDistance(impact_model).model, impact_model)

    def score(self, market: Market) -> np.ndarray:
        F = gate_features(market, self.ens, self.fallback, self.ood, self.impact_model)
        return self.reg.predict(F)


ESTIMATORS = ("ensemble", "ood", "shadow")


def make_estimator(name: str, primary=None, fallback=None, impact_model: str = "sqrt"):
    if name == "ensemble":
        return EnsembleDisagreement(primary, impact_model)
    if name == "ood":
        return OODDistance(impact_model)
    if name == "shadow":
        return ShadowRegret(primary, fallback, impact_model)
    raise ValueError(f"unknown estimator {name!r}; expected one of {ESTIMATORS} "
                     "(RCG needs fitted validation data: use RCGScore.from_validation)")


__all__ = ["ESTIMATORS", "EnsembleDisagreement", "OODDistance", "RCGScore", "ShadowRegret",
           "make_estimator"]
