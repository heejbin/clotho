"""Shared constants and small helpers for Clotho iteration runs."""
from __future__ import annotations

import logging
import random
from dataclasses import replace

import numpy as np
import torch

from clotho.preprocessing.feature_reduction import PCAFeatureReducer
from utils import get_gmm_components_perplexity

PASS_THRESH = 0.5
MIN_CLUSTER_SIZE = 5
MIN_CLUSTER_SIZE_MCMC = 1
# Scoring models that share the PyMC GMM path (NUTS or ADVI posterior draws).
BAYES_GMM_MODELS = ('MCMCGMM', 'ADVIGMM')
# Minimum labeled refs per class before fitting (single: pass only; dual: both).
# Bypassed when --random_start_weighted / --random_start_double_hard_weighted
# or the *_sum_* variants: n0 is the only count gate, then fit immediately.
MIN_CLASS_REFS_TO_FIT = 10
# Points added to the reference set each AL iteration (BatchBALD batch size).
NUM_ADD = 10
# Label grid for pairwise epig: snap η to {0, 1/n, ..., 1}.
# rankvar and epigspearman rank raw η (average ties), matching spearmanr
# on continuous P(pass|x). For epigspearman, n_trials is the binomial label size.
# Override with --n_trials.
DEFAULT_N_TRIALS = 10
# --random_100_start: random acquisition until this many refs, then the requested method.
RANDOM_START_N_REFS = 100
# --fast: cheaper NUTS / BALD thinning / warmup adds.
FAST_POSTERIOR_EVERY = 20  # iterations
FAST_THIN_STRIDE = 5
# Random/fallback adds while a class is still below MIN_CLASS_REFS_TO_FIT.
FAST_WARMUP_ADD = 20
# Intermediate figures: every N refs, unless --verbose_figure (every iter).
DEFAULT_PLOT_EVERY_POINTS = 50
# PCA feature-search schedule (shared by GMM and MCMCGMM).
INIT_N_FEATURES = 10
# --d_schedule: start at the floor and only step up (never down).
D_SCHEDULE_INIT_N_FEATURES = 5
D_SCHEDULE_FEATURE_STEP = 5
FEATURE_STEP = 10
MIN_N_FEATURES = 5
MAX_N_FEATURES_GMM = 50
MAX_N_FEATURES_MCMC = 50

# sklearn EM defaults (current / strict, matched to Clotho GMM) vs --generous
# (pre-tightening MCMCGMM EM: n_init=3, reg_covar=0.05; max_iter/tol were
# sklearn defaults because they were not passed through).
DEFAULT_EM = {
    'n_init': 30,
    'max_iter': 500,
    'tol': 1e-3,
    'reg_covar': 1e-6,
}
GENEROUS_EM = {
    'n_init': 3,
    'max_iter': 100,
    'tol': 1e-3,
    'reg_covar': 0.05,
}
# Cheap probe / CV EM restarts: never exceed this, and never exceed the run's EM n_init.
PROBE_N_INIT_CAP = 10
# Adaptive --reg_decay: start at 1e-1 and only step λ → λ/10.
# Dual: 3-fold Pearson of Bayes p(y=pass|x) vs labels (same functional as eval).
# Single: 3-fold Pearson of GMM rank-scores.
# Decay uses the same min-fold / 0.1 gate as PCA-dim LD and --d_schedule HD.
# Monotone: cannot go back up.
INIT_REG_COVAR_CV = 1e-1
REG_COVAR_CV_FACTOR = 10.0
MIN_REG_COVAR_CV = 1e-6
MAX_REG_COVAR_CV = 1.0
# Shared CV stability floor: min fold must beat max(current min fold, this).
CV_MIN_FOLD_FLOOR = 0.1

# Valid EM (GMM) ablation grid for --all_poss_comb (RQ1-1 inventory):
#   - oracle_prior ⇒ dual_gmm
#   - full cov only under rigorous + ¬oracle
ALL_POSS_COMB = [
    # covariance, dual, generous, oracle_prior
    dict(covariance_type='diag', dual_gmm=False, generous=True,  oracle_prior=False),
    dict(covariance_type='diag', dual_gmm=False, generous=False, oracle_prior=False),
    dict(covariance_type='diag', dual_gmm=True,  generous=True,  oracle_prior=False),
    dict(covariance_type='diag', dual_gmm=True,  generous=True,  oracle_prior=True),
    dict(covariance_type='diag', dual_gmm=True,  generous=False, oracle_prior=False),
    dict(covariance_type='diag', dual_gmm=True,  generous=False, oracle_prior=True),
    dict(covariance_type='full', dual_gmm=False, generous=False, oracle_prior=False),
    dict(covariance_type='full', dual_gmm=True,  generous=False, oracle_prior=False),
]


def _em_settings(generous: bool, reg_covar: float | None = None) -> dict:
    """Return EM hyperparams; optional ``reg_covar`` overrides the preset value."""
    em = dict(GENEROUS_EM if generous else DEFAULT_EM)
    if reg_covar is not None:
        em['reg_covar'] = float(reg_covar)
    return em


def _format_reg_covar_tag(reg_covar: float) -> str:
    """Compact path tag, e.g. 1e-6 -> 'reg1e-6', 0.01 -> 'reg1e-2'."""
    s = f"{float(reg_covar):.0e}"  # e.g. '1e-06'
    mant, exp = s.split('e')
    return f"reg{mant}e{int(exp)}"


def _results_clotho_root(scoring_model: str) -> str:
    """GMM runs live under ``results_clotho/EM/``; MCMC/ADVI stay at the top level."""
    if scoring_model == 'GMM':
        return './results_clotho/EM'
    return './results_clotho'


def _is_gmm_collapse_error(exc: BaseException) -> bool:
    """sklearn GMM EM died on empty/collapsed components."""
    msg = str(exc).lower()
    return any(
        needle in msg
        for needle in (
            'ill-defined empirical covariance',
            'fitting the mixture model failed',
            'n_samples >= n_components',
            'must be >= n_components',
        )
    )


def _apply_em_settings(scorer_config, generous: bool, reg_covar: float | None = None):
    """Apply strict or generous EM hyperparameters onto a scorer config."""
    em = _em_settings(generous, reg_covar=reg_covar)
    if isinstance(scorer_config, MCMCGMMScorerConfig):
        return replace(
            scorer_config,
            sklearn_n_init=em['n_init'],
            sklearn_max_iter=em['max_iter'],
            sklearn_tol=em['tol'],
            reg_covar=em['reg_covar'],
        )
    return replace(
        scorer_config,
        n_init=em['n_init'],
        max_iter=em['max_iter'],
        tol=em['tol'],
        reg_covar=em['reg_covar'],
    )


def _cheap_probe_n_init(scorer_config) -> int:
    """Fewer EM restarts for k-probe / PCA CV than the main fit (capped)."""
    if isinstance(scorer_config, MCMCGMMScorerConfig):
        n = int(scorer_config.sklearn_n_init)
    else:
        n = int(scorer_config.n_init)
    return max(1, min(n, PROBE_N_INIT_CAP))
# Fixed run-level color vmax for BALD plots (theory max is log(2)≈0.693;
# use a lower clip so typical small scores still show color contrast).
BALD_UNC_VMAX = 0.2
# Aleatoric E_t[H[p_t]] uses the full binary-entropy range [0, log 2].
ALEATORIC_UNC_VMAX = float(np.log(2.0))


def _diag_vars_to_covmats(variances):
    """Expand diagonal variances to full cov matrices for GMM plot overlays."""
    variances = np.asarray(variances, dtype=float)
    if variances.ndim == 2:
        return np.stack([np.diag(v) for v in variances], axis=0)
    if variances.ndim == 3:
        n, k, d = variances.shape
        out = np.zeros((n, k, d, d), dtype=float)
        idx = np.arange(d)
        out[:, :, idx, idx] = variances
        return out
    raise ValueError(f"Expected variances ndim 2 or 3, got {variances.ndim}")



def _uncertainty_color_limits_for_method(refset_extension_method, scoring_model=None):
    """Return (vmin, vmax) for absolute uncertainty coloring within a run.

    BALD (and MCMCGMM ``balanced``, which uses BALD for the unc half) uses a
    fixed clip. Other methods leave limits open and are finalized from observed
    values at the end of the run.
    """
    if refset_extension_method in ('bald', 'batchbald'):
        return 0.0, BALD_UNC_VMAX
    if refset_extension_method == 'balanced' and scoring_model in BAYES_GMM_MODELS:
        return 0.0, BALD_UNC_VMAX
    return None, None


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducible AL sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_feature_rep(LIH, LIH_rep, n_features, dim_reduction_method):
    if n_features in LIH_rep:
        return
    if dim_reduction_method != 'pca':
        raise NotImplementedError(
            f"Dimensionality reduction method '{dim_reduction_method}' is not implemented."
        )
    dim_reducer = PCAFeatureReducer(n_features=n_features)
    LIH_rep[n_features] = np.asarray(dim_reducer.fit_transform(LIH))


def _is_recoverable_mcmc_fit_error(exc: BaseException) -> bool:
    """True for init/sampling failures that may be fixed by lowering PCA dim."""
    try:
        from pymc.exceptions import SamplingError
    except Exception:  # pragma: no cover
        SamplingError = ()  # type: ignore

    if isinstance(exc, SamplingError):
        return True
    type_name = type(exc).__name__.lower()
    if 'sampling' in type_name:
        return True
    msg = str(exc).lower()
    return any(n in msg for n in ('-inf', 'bad initial energy', 'initial evaluation'))


def _gmm_warm_start_init_params(scorer):
    """Copy last GMM params for PyMC warm-start (posterior mean if available)."""
    if scorer is None or not getattr(scorer, "check_is_fitted", lambda: False)():
        return None
    model = getattr(scorer, "model", None)
    if model is None:
        return None
    dtype = getattr(model, "_dtype", np.dtype(np.float32))
    if getattr(model, "mcmc_means_", None) is not None:
        mu = np.array(model.mcmc_means_, dtype=dtype, copy=True)
        weights = np.array(model.mcmc_weights_, dtype=dtype, copy=True)
        cov = np.array(model.mcmc_covariances_, dtype=dtype, copy=True)
        reg = float(model.reg_covar)
        sigma = np.sqrt(np.maximum(cov - reg, 1e-12)).astype(dtype)
        return {"mu": mu, "w": weights, "cov": cov, "sigma": sigma}
    getter = getattr(scorer, "get_init_params", None)
    params = getter() if getter is not None else None
    if not params:
        return None
    return {key: np.array(val, copy=True) for key, val in params.items()}


def _serialize_fit_profile(fit_profile: dict) -> dict:
    """Keep numeric timings rounded; pass through non-numeric tags (e.g. inference='advi')."""
    out = {}
    for key, val in fit_profile.items():
        if isinstance(val, (bool, np.bool_)):
            out[key] = bool(val)
        elif isinstance(val, (int, float, np.integer, np.floating)):
            out[key] = round(float(val), 3)
        else:
            out[key] = val
    return out


def _maybe_update_n_components_from_perplexity(
    current_n_components: int,
    perplexity: float,
    num_effective_references: int,
    max_n_components: int,
    iteration: int,
    cluster_update_size: int = 1,
    min_n_components: int = MIN_CLUSTER_SIZE,
) -> int:
    """Return updated k from weight-perplexity heuristic (may be unchanged)."""
    saturated_n_comp_thresh = current_n_components - (cluster_update_size * 0.5)
    understuffed_n_comp_thresh = current_n_components - (cluster_update_size * 1.5)

    if perplexity > saturated_n_comp_thresh:
        next_n_components = min(current_n_components + cluster_update_size, max_n_components)
        if num_effective_references // 10 > next_n_components:
            logging.debug(
                f"[n_components probe (Iteration: {iteration})] saturated "
                f"(perplexity={perplexity:.4f} > {saturated_n_comp_thresh:.4f}) "
                f"-> k {current_n_components} -> {next_n_components}"
            )
            return next_n_components
    elif perplexity < understuffed_n_comp_thresh:
        next_n_components = max(current_n_components - cluster_update_size, min_n_components)
        logging.debug(
            f"[n_components probe (Iteration: {iteration})] understuffed "
            f"(perplexity={perplexity:.4f} < {understuffed_n_comp_thresh:.4f}) "
            f"-> k {current_n_components} -> {next_n_components}"
        )
        return next_n_components
    return current_n_components


def bayes_pass_posterior(logp_pass, logp_fail, prior_pass=0.5, prior_fail=0.5):
    """Class-1 posterior from two class-conditional log-likelihoods (Bayes).

    p(y=1|x) = p(x|y=1)p(y=1) / [p(x|y=1)p(y=1) + p(x|y=0)p(y=0)]
    computed in log-space for numerical stability.
    """
    logp_pass = np.asarray(logp_pass, dtype=float)
    logp_fail = np.asarray(logp_fail, dtype=float)
    prior_pass = float(prior_pass)
    prior_fail = float(prior_fail)
    if prior_pass <= 0.0 or prior_fail <= 0.0:
        raise ValueError(
            f"Class priors must be positive (got prior_pass={prior_pass}, prior_fail={prior_fail})"
        )
    log_joint_pass = logp_pass + np.log(prior_pass)
    log_joint_fail = logp_fail + np.log(prior_fail)
    return np.exp(log_joint_pass - np.logaddexp(log_joint_pass, log_joint_fail))


def _fit_labels_for_class(reference_set_labels, target_class: int):
    """Map binary labels so GMMScorer.fit keeps rows with value 1 for ``target_class``.

    Scorers always train on ``labels == 1``. For the fail GMM (target_class=0),
    invert so failing references become the effective training set.
    """
    labels = np.asarray(reference_set_labels, dtype=int)
    if target_class == 1:
        return labels
    if target_class == 0:
        return 1 - labels
    raise ValueError(f"target_class must be 0 or 1, got {target_class}")


def _success_rate_weights(scores, target_class: int):
    """Class-conditional EM/MCMC weights from repeated-eval success rate k/10.

    Pass GMM uses ``k/10``; fail GMM uses ``1 - k/10``. Every reference is kept
    (fit labels are all 1); a 0/10 point has ~0 mass in the pass density.
    """
    if target_class not in (0, 1):
        raise ValueError(f"target_class must be 0 or 1, got {target_class}")
    rate = np.clip(np.asarray(scores, dtype=float).ravel(), 0.0, 1.0)
    return rate if target_class == 1 else (1.0 - rate)


def _class_fit_labels_and_weight(
    reference_set_labels,
    reference_scores,
    target_class: int,
    sample_weight=None,
    success_rate_weight=False,
):
    """Labels + ``sample_weight`` for one class-conditional GMM fit.

    Default: hard 0/1 split; optional IPW/lambda* weights are sliced to the
    class rows inside the scorer.

    ``success_rate_weight``: do not drop points. Weight pass by k/10 and fail
    by 1-k/10. Incompatible with IPW/lambda* ``sample_weight``.
    """
    if success_rate_weight:
        if sample_weight is not None:
            raise ValueError(
                "success_rate_weight cannot be combined with IPW/lambda* sample_weight"
            )
        n = int(np.asarray(reference_scores).ravel().shape[0])
        return np.ones(n, dtype=int), _success_rate_weights(reference_scores, target_class)
    return _fit_labels_for_class(reference_set_labels, target_class), sample_weight


