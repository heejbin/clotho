import json
import math
import numpy as np
import random
import torch
import warnings

import logging
from pathlib import Path

from scipy.spatial.distance import cdist
from scipy.special import gammaln
from scipy.stats import norm, rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

def extend_reference_set_uncertainty(scoring_model, refset_indices, target_vectors, num_add=10):
    candidate_indices = [i for i in range(len(target_vectors)) if i not in refset_indices]
    if len(candidate_indices) < num_add:
        raise ValueError("Not enough selectable indices to add {} new indices.".format(num_add))

    uncertainties = scoring_model.get_uncertainties(target_vectors)
    logging.info(f"Uncertainty stats: mean={np.mean(uncertainties):.4f}, std={np.std(uncertainties):.4f}, min={np.min(uncertainties):.4f}, max={np.max(uncertainties):.4f}")
    logging.info(f"Uncertainties: {uncertainties}")

    top_n_uncertain_indices = uncertainties[candidate_indices].argsort()[::-1][:num_add]
    top_n_uncertain_indices = [candidate_indices[i] for i in top_n_uncertain_indices]
    
    assert set(top_n_uncertain_indices).isdisjoint(set(refset_indices)), "Selected indices must not overlap with reference set indices. Current reference set indices: {}, selected indices: {}".format(refset_indices, top_n_uncertain_indices)
    
    return top_n_uncertain_indices, uncertainties


def _select_top_by_score(refset_indices, scores, num_add=10):
    """Return top-``num_add`` candidate indices by descending ``scores``."""
    scores = np.asarray(scores, dtype=float)
    candidate_indices = [i for i in range(len(scores)) if i not in refset_indices]
    if len(candidate_indices) < num_add:
        raise ValueError(f"Not enough selectable indices to add {num_add} new indices.")
    cand = np.asarray(candidate_indices, dtype=int)
    order = np.argsort(scores[cand])[::-1][:num_add]
    selected = [int(cand[i]) for i in order]
    assert set(selected).isdisjoint(set(refset_indices)), (
        "Selected indices must not overlap with reference set indices. "
        f"Current reference set indices: {refset_indices}, selected indices: {selected}"
    )
    return selected

def _binary_shannon_entropy(p):
    """Binary Shannon entropy H(p) with 0 log 0 = 0. ``p`` in [0, 1]."""
    p = np.asarray(p, dtype=float)
    q = 1.0 - p
    with np.errstate(divide="ignore", invalid="ignore"):
        term_p = np.where(p > 0.0, p * np.log(p), 0.0)
        term_q = np.where(q > 0.0, q * np.log(q), 0.0)
    return np.maximum(-(term_p + term_q), 0.0)


# Dual-density OOD: E_t[log (π1 p_pass + π0 p_fail)] far below reference support.
DEFAULT_OOD_REF_PERCENTILE = 1.0
DEFAULT_OOD_MARGIN = 50.0


def flag_dual_density_ood(
    log_marginal,
    refset_indices,
    ref_percentile=DEFAULT_OOD_REF_PERCENTILE,
    margin=DEFAULT_OOD_MARGIN,
):
    """Flag points whose dual-class log-marginal is far below the reference set.

    ``log_marginal[i] ≈ E_t[ log(π1 p(x_i|pass,θ_t) + π0 p(x_i|fail,θ_t)) ]``.
    Threshold = percentile(ref log_marginal) − margin (nats).
    """
    log_marginal = np.asarray(log_marginal, dtype=float)
    ref_idx = np.asarray(list(refset_indices), dtype=int)
    if ref_idx.size == 0:
        thresh = float(-np.inf)
        return np.zeros(log_marginal.shape[0], dtype=bool), thresh
    ref_lm = log_marginal[ref_idx]
    ref_lm = ref_lm[np.isfinite(ref_lm)]
    if ref_lm.size == 0:
        thresh = float(-np.inf)
        return np.zeros(log_marginal.shape[0], dtype=bool), thresh
    thresh = float(np.percentile(ref_lm, float(ref_percentile))) - float(margin)
    is_ood = np.isfinite(log_marginal) & (log_marginal < thresh)
    return is_ood, thresh


def dual_gmm_class_posteriors(
    scoring_model_pass,
    scoring_model_fail,
    target_vectors_pass,
    target_vectors_fail,
    prior_pass=0.5,
    prior_fail=0.5,
    n_draws=None,
):
    """Per-draw Bayes pass posteriors from two class-conditional MCMC GMMs.

    For each paired posterior draw ``t``::

        p_t(x) = p(y=1|x, θ_t)
               = [p(x|y=1,θ^pass_t) π_1] /
                 [p(x|y=1,θ^pass_t) π_1 + p(x|y=0,θ^fail_t) π_0]

    Draws are paired by index (min draw count if the two posteriors differ).

    Returns
    -------
    p_t : ndarray, shape (n_draws, n_samples)
        Pass-class posterior under each MCMC draw.
    log_marginal : ndarray, shape (n_samples,)
        Mean over draws of ``logaddexp(log π1 p_pass, log π0 p_fail)``.
    """
    prior_pass = float(prior_pass)
    prior_fail = float(prior_fail)
    if prior_pass <= 0.0 or prior_fail <= 0.0:
        raise ValueError(
            f"Class priors must be positive (got prior_pass={prior_pass}, prior_fail={prior_fail})"
        )

    logp_pass = np.asarray(
        scoring_model_pass.predictive_log_likelihoods_over_draws(
            np.asarray(target_vectors_pass), n_draws=n_draws
        ),
        dtype=float,
    )
    logp_fail = np.asarray(
        scoring_model_fail.predictive_log_likelihoods_over_draws(
            np.asarray(target_vectors_fail), n_draws=n_draws
        ),
        dtype=float,
    )
    if logp_pass.shape[1] != logp_fail.shape[1]:
        raise ValueError(
            f"Pass/fail predictive LL sample counts differ: "
            f"{logp_pass.shape[1]} vs {logp_fail.shape[1]}"
        )
    n_draws_eff = min(logp_pass.shape[0], logp_fail.shape[0])
    if n_draws_eff < 1:
        raise ValueError("Need at least one posterior draw for dual-GMM BALD.")
    logp_pass = logp_pass[:n_draws_eff]
    logp_fail = logp_fail[:n_draws_eff]

    log_joint_pass = logp_pass + np.log(prior_pass)
    log_joint_fail = logp_fail + np.log(prior_fail)
    log_marginal_t = np.logaddexp(log_joint_pass, log_joint_fail)
    p_t = np.exp(log_joint_pass - log_marginal_t)
    log_marginal = log_marginal_t.mean(axis=0)
    return p_t, log_marginal


def dual_gmm_bald_scores(
    scoring_model_pass,
    scoring_model_fail,
    target_vectors_pass,
    target_vectors_fail,
    prior_pass=0.5,
    prior_fail=0.5,
    n_draws=None,
):
    """Canonical binary BALD from two class-conditional MCMC GMMs.

    BALD = H[p̄] − E_t[H[p_t]] with binary Shannon entropy H.

    Returns
    -------
    bald : ndarray, shape (n_samples,)
        Epistemic mutual information H[p̄] − E_t[H[p_t]].
    aleatoric : ndarray, shape (n_samples,)
        Expected predictive entropy E_t[H[p_t]].
    log_marginal : ndarray, shape (n_samples,)
        Mean over draws of ``logaddexp(log π1 p_pass, log π0 p_fail)``,
        i.e. E_t[log p(x|θ_t)] under the class mixture (for OOD diagnostics).
    p_var : ndarray, shape (n_samples,)
        Per-point Var_t[p(y=pass | x, θ_t)] (MCMC disagreement).
    """
    p_t, log_marginal = dual_gmm_class_posteriors(
        scoring_model_pass,
        scoring_model_fail,
        target_vectors_pass,
        target_vectors_fail,
        prior_pass=prior_pass,
        prior_fail=prior_fail,
        n_draws=n_draws,
    )
    p_bar = p_t.mean(axis=0)
    h_total = _binary_shannon_entropy(p_bar)
    h_aleatoric = _binary_shannon_entropy(p_t).mean(axis=0)
    bald = np.maximum(h_total - h_aleatoric, 0.0)
    p_var = np.var(p_t, axis=0)
    return bald, h_aleatoric, log_marginal, p_var


def pool_rank_variance(eta_t, n_trials=None, rank_indices=None):
    """Variance across draws of each point's rank.

    ``eta_t`` has shape ``(n_draws, n_samples)``. Higher score = higher rank.
    Ties use the average rank, the same reduction as ``spearmanr``.

    ``n_trials`` snaps each draw onto ``{0, 1/n, ..., 1}`` before ranking.
    Predictions that fall in the same label bin are tied, so within-bin order
    does not move the rank. ``None`` ranks the raw scores.

    ``rank_indices`` is the set whose order enters the metric (the target
    pool). Points outside it get score 0 and do not affect the ranks.
    Non-finite scores rank last.
    """
    eta_t = np.asarray(eta_t, dtype=float)
    if eta_t.ndim != 2:
        raise ValueError(
            f"eta_t must have shape (n_draws, n_samples), got {eta_t.shape}"
        )
    if eta_t.shape[0] < 1:
        raise ValueError("Need at least one posterior draw for rank variance.")
    n_draws, n = eta_t.shape
    finite = np.isfinite(eta_t)
    scored = np.full(eta_t.shape, -np.inf, dtype=float)
    if n_trials is None:
        scored[finite] = eta_t[finite]
    else:
        scored[finite] = _snap_counts(eta_t[finite], n_trials)
    if rank_indices is None:
        rank_idx = np.arange(n, dtype=int)
    else:
        rank_idx = np.asarray(list(rank_indices), dtype=int)
        if rank_idx.size and (int(rank_idx.min()) < 0 or int(rank_idx.max()) >= n):
            raise ValueError("rank_indices out of range for eta_t.")
    out = np.zeros(n, dtype=float)
    if rank_idx.size == 0:
        return out
    ranks = rankdata(scored[:, rank_idx], method="average", axis=1)
    if ranks.shape != (n_draws, rank_idx.size):
        raise RuntimeError(
            f"rankdata returned shape {ranks.shape}, expected {(n_draws, rank_idx.size)}"
        )
    out[rank_idx] = np.var(ranks, axis=0)
    return out


def dual_gmm_rank_variance(
    scoring_model_pass,
    scoring_model_fail,
    target_vectors_pass,
    target_vectors_fail,
    prior_pass=0.5,
    prior_fail=0.5,
    n_draws=None,
    n_trials=None,
    rank_indices=None,
):
    """Epistemic uncertainty of the rank of raw η.

    η_t(x) = p(y=pass | x, θ_t). Each draw ranks ``rank_indices`` on raw η
    with average ties, the same ranking ``spearmanr`` applies to the
    continuous P(pass|x) score. The score is Var_t[rank_t(x)].

    ``n_trials`` is unused. Pass ``None``. Snapping onto ``{0, 1/n, ..., 1}``
    is still available from ``pool_rank_variance`` but is not the rankvar score.

    Returns
    -------
    rank_var : ndarray, shape (n_samples,)
        Variance of the tied rank of raw η across posterior draws.
    log_marginal : ndarray, shape (n_samples,)
        Mean over draws of ``logaddexp(log π1 p_pass, log π0 p_fail)``.
    p_var : ndarray, shape (n_samples,)
        Per-point Var_t[η_t(x)] (probability-scale disagreement).
    """
    p_t, log_marginal = dual_gmm_class_posteriors(
        scoring_model_pass,
        scoring_model_fail,
        target_vectors_pass,
        target_vectors_fail,
        prior_pass=prior_pass,
        prior_fail=prior_fail,
        n_draws=n_draws,
    )
    rank_var = pool_rank_variance(
        p_t, n_trials=None, rank_indices=rank_indices
    )
    p_var = np.var(p_t, axis=0)
    return rank_var, log_marginal, p_var


def dual_gmm_likelihood_variance(
    scoring_model_pass,
    scoring_model_fail,
    target_vectors_pass,
    target_vectors_fail,
    prior_pass=0.5,
    prior_fail=0.5,
    n_draws=None,
):
    """Var_t of the dual-class mixture likelihood.

    ``v(x) = Var_t[ π1 p(x|pass,θ_t) + π0 p(x|fail,θ_t) ]``, matching
    ``MCMCGMMScorer.get_uncertainties`` (``Var_t[p(x|θ)]``) on the class
    mixture rather than a single class-conditional density.
    """
    prior_pass = float(prior_pass)
    prior_fail = float(prior_fail)
    if prior_pass <= 0.0 or prior_fail <= 0.0:
        raise ValueError(
            f"Class priors must be positive (got prior_pass={prior_pass}, prior_fail={prior_fail})"
        )
    logp_pass = np.asarray(
        scoring_model_pass.predictive_log_likelihoods_over_draws(
            np.asarray(target_vectors_pass), n_draws=n_draws
        ),
        dtype=float,
    )
    logp_fail = np.asarray(
        scoring_model_fail.predictive_log_likelihoods_over_draws(
            np.asarray(target_vectors_fail), n_draws=n_draws
        ),
        dtype=float,
    )
    if logp_pass.shape[1] != logp_fail.shape[1]:
        raise ValueError(
            f"Pass/fail predictive LL sample counts differ: "
            f"{logp_pass.shape[1]} vs {logp_fail.shape[1]}"
        )
    n_draws_eff = min(logp_pass.shape[0], logp_fail.shape[0])
    if n_draws_eff < 1:
        raise ValueError("Need at least one posterior draw for likelihood variance.")
    logp_pass = logp_pass[:n_draws_eff]
    logp_fail = logp_fail[:n_draws_eff]
    log_mix_t = np.logaddexp(
        logp_pass + np.log(prior_pass),
        logp_fail + np.log(prior_fail),
    )
    return np.var(np.exp(log_mix_t), axis=0)


def extend_reference_set_likelihood_variance(
    scoring_model,
    refset_indices,
    target_vectors,
    num_add=10,
    scoring_model_fail=None,
    target_vectors_fail=None,
    prior_pass=None,
    prior_fail=None,
):
    """Top-k by Var_t[p(x|θ)] (pass GMM, or dual mixture if fail scorer given)."""
    if scoring_model_fail is not None:
        if target_vectors_fail is None:
            raise ValueError(
                "extend_reference_set_likelihood_variance requires "
                "target_vectors_fail when scoring_model_fail is set."
            )
        if prior_pass is None:
            prior_pass = 0.5
        if prior_fail is None:
            prior_fail = 0.5
        scores = dual_gmm_likelihood_variance(
            scoring_model,
            scoring_model_fail,
            target_vectors,
            target_vectors_fail,
            prior_pass=prior_pass,
            prior_fail=prior_fail,
        )
        logging.info(
            f"Likelihood-var (dual mix) stats: mean={np.mean(scores):.4g}, "
            f"std={np.std(scores):.4g}, min={np.min(scores):.4g}, max={np.max(scores):.4g}"
        )
    else:
        scores = scoring_model.get_uncertainties(target_vectors)
        logging.info(
            f"Likelihood-var stats: mean={np.mean(scores):.4g}, "
            f"std={np.std(scores):.4g}, min={np.min(scores):.4g}, max={np.max(scores):.4g}"
        )
    selected = _select_top_by_score(refset_indices, scores, num_add=num_add)
    logging.info(f"llvar selected {len(selected)} pts")
    return selected, scores


def dual_gmm_predictive_probs(
    scoring_model_pass,
    scoring_model_fail,
    target_vectors_pass,
    target_vectors_fail,
    prior_pass=0.5,
    prior_fail=0.5,
    n_draws=None,
):
    """Predictive class probabilities under each MCMC draw for BatchBALD.

    Returns
    -------
    probs : ndarray, shape (n_draws, n_samples, 2)
        ``probs[s, n, 0] = p(pass | x_n, θ_s)``,
        ``probs[s, n, 1] = p(fail | x_n, θ_s)``.
    log_marginal : ndarray, shape (n_samples,)
        E_t[log p(x|θ_t)] under the dual-class mixture.
    """
    p_t, log_marginal = dual_gmm_class_posteriors(
        scoring_model_pass,
        scoring_model_fail,
        target_vectors_pass,
        target_vectors_fail,
        prior_pass=prior_pass,
        prior_fail=prior_fail,
        n_draws=n_draws,
    )
    probs = np.stack([p_t, 1.0 - p_t], axis=-1)
    return probs, log_marginal


def _categorical_entropy(p, axis=-1, eps=1e-12):
    """Shannon entropy over a categorical distribution along ``axis``."""
    p = np.asarray(p, dtype=float)
    return -np.sum(p * np.log(p + eps), axis=axis)


def batch_bald_greedy(probs, batch_size, candidate_idx=None):
    """Greedy BatchBALD selection (Kirsch et al., 2019) reference implementation.

    ``probs``: (S, N, C) predictive ``p(y=k | x_n, θ_s)``.
    Optionally restrict the search to ``candidate_idx`` (e.g. a BALD shortlist).

    Returns
    -------
    selected : list[int]
        Selected pool indices in acquisition order.
    mi_scores : list[float]
        Incremental joint mutual information of each selected point given the
        ones already chosen.
    """
    S, N, C = probs.shape
    pool = np.arange(N) if candidate_idx is None else np.asarray(candidate_idx, dtype=int)
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if len(pool) < batch_size:
        raise ValueError(
            f"Not enough candidates ({len(pool)}) to select batch_size={batch_size}."
        )

    cond_entropy_per_point = _categorical_entropy(probs, axis=-1).mean(axis=0)  # (N,)
    selected: list[int] = []
    joint = np.ones((S, 1))  # (S, C^k) running per-sample joint over selected batch
    running_cond_entropy = 0.0
    mi_scores: list[float] = []
    selected_set: set[int] = set()

    for _ in range(batch_size):
        best_score, best_idx, best_joint = -np.inf, -1, None

        for n in pool:
            n = int(n)
            if n in selected_set:
                continue
            cand = probs[:, n, :]  # (S, C)
            new_joint = (joint[:, :, None] * cand[:, None, :]).reshape(S, -1)  # (S, C^{k+1})
            avg_joint = new_joint.mean(axis=0)  # (C^{k+1},)
            h_joint = float(_categorical_entropy(avg_joint, axis=-1))
            mi = h_joint - (running_cond_entropy + cond_entropy_per_point[n])
            if mi > best_score:
                best_score, best_idx, best_joint = mi, n, new_joint

        if best_idx < 0 or best_joint is None:
            raise RuntimeError("BatchBALD greedy failed to select a candidate.")
        selected.append(int(best_idx))
        selected_set.add(int(best_idx))
        joint = best_joint
        running_cond_entropy += float(cond_entropy_per_point[best_idx])
        mi_scores.append(float(best_score))

    return selected, mi_scores


def _log_dual_bald_diagnostics(scores, aleatoric, log_marginal, refset_indices, prior_pass, prior_fail, tag="bald"):
    logging.info(
        f"{tag} (dual) stats: mean={np.mean(scores):.4f}, "
        f"std={np.std(scores):.4f}, min={np.min(scores):.4f}, "
        f"max={np.max(scores):.4f}, prior_pass={float(prior_pass):.4f}, "
        f"prior_fail={float(prior_fail):.4f}"
    )
    logging.info(
        f"aleatoric (dual) stats: mean={np.mean(aleatoric):.4f}, "
        f"std={np.std(aleatoric):.4f}, min={np.min(aleatoric):.4f}, "
        f"max={np.max(aleatoric):.4f}"
    )
    is_ood, ood_thresh = flag_dual_density_ood(log_marginal, refset_indices)
    n_ood = int(np.sum(is_ood))
    logging.info(
        f"dual-density OOD: n={n_ood}/{len(log_marginal)} "
        f"(thresh={ood_thresh:.2f}, "
        f"log_marginal mean={np.mean(log_marginal):.2f}, "
        f"min={np.min(log_marginal):.2f})"
    )


def extend_reference_set_bald(
    scoring_model,
    refset_indices,
    target_vectors,
    num_add=10,
    scoring_model_fail=None,
    target_vectors_fail=None,
    prior_pass=None,
    prior_fail=None,
):
    """Top-k by dual-GMM binary BALD: H[p̄] − mean_t H[p_t].

    Requires pass/fail MCMCGMM scorers. Per-draw Bayes pass posterior from the
    two class-conditional likelihoods.

    Returns
    -------
    selected : list[int]
        Top-k candidate indices by BALD.
    bald_scores : ndarray
        Per-point BALD (epistemic) scores.
    aleatoric : ndarray
        Per-point aleatoric uncertainty E_t[H[p_t]].
    log_marginal : ndarray
        Per-point E_t[log p(x|θ_t)] under the dual-class mixture.
    p_var : ndarray
        Per-point Var_t[p(y=pass | x, θ_t)] (MCMC disagreement).
    """
    if scoring_model_fail is None or target_vectors_fail is None:
        raise ValueError(
            "extend_reference_set_bald requires dual GMM "
            "(scoring_model_fail / target_vectors_fail)."
        )
    if prior_pass is None:
        prior_pass = 0.5
    if prior_fail is None:
        prior_fail = 0.5
    scores, aleatoric, log_marginal, p_var = dual_gmm_bald_scores(
        scoring_model,
        scoring_model_fail,
        target_vectors,
        target_vectors_fail,
        prior_pass=prior_pass,
        prior_fail=prior_fail,
    )
    _log_dual_bald_diagnostics(
        scores, aleatoric, log_marginal, refset_indices, prior_pass, prior_fail, tag="bald"
    )
    selected = _select_top_by_score(refset_indices, scores, num_add=num_add)
    logging.info(f"bald (dual) selected {len(selected)} pts")
    return selected, scores, aleatoric, log_marginal, p_var


def extend_reference_set_rankvar(
    scoring_model,
    refset_indices,
    target_vectors,
    num_add=10,
    scoring_model_fail=None,
    target_vectors_fail=None,
    prior_pass=None,
    prior_fail=None,
    n_trials=None,
    rank_indices=None,
):
    """Top-k by Var_t[rank_t(x)] of raw η.

    The reported Spearman ranks continuous P(pass|x), with average ties. Each
    posterior draw ranks ``rank_indices`` (the eval pool; default the whole
    pool) the same way, and the score is the variance of that rank.
    ``n_trials`` is accepted and ignored so older callers keep working.
    Reference points inside the ranked set stay in the ranking and are
    excluded from selection.

    Returns
    -------
    selected : list[int]
        Top-k candidate indices by rank variance.
    rank_var : ndarray
        Per-point Var_t[rank_t(x)].
    log_marginal : ndarray
        Per-point E_t[log p(x|θ_t)] under the dual-class mixture.
    p_var : ndarray
        Per-point Var_t[η_t(x)].
    """
    if scoring_model_fail is None or target_vectors_fail is None:
        raise ValueError(
            "extend_reference_set_rankvar requires dual GMM "
            "(scoring_model_fail / target_vectors_fail)."
        )
    if prior_pass is None:
        prior_pass = 0.5
    if prior_fail is None:
        prior_fail = 0.5
    scores, log_marginal, p_var = dual_gmm_rank_variance(
        scoring_model,
        scoring_model_fail,
        target_vectors,
        target_vectors_fail,
        prior_pass=prior_pass,
        prior_fail=prior_fail,
        n_trials=None,
        rank_indices=rank_indices,
    )
    n_ranked = (
        scores.shape[0] if rank_indices is None else len(list(rank_indices))
    )
    logging.info(
        f"rankvar stats: mean={np.mean(scores):.4f}, "
        f"std={np.std(scores):.4f}, min={np.min(scores):.4f}, "
        f"max={np.max(scores):.4f}, rank=raw η average ties, "
        f"n_ranked={int(n_ranked)}, prior_pass={float(prior_pass):.4f}, "
        f"prior_fail={float(prior_fail):.4f}"
    )
    if float(np.max(scores)) == 0.0:
        logging.warning(
            "rankvar is 0 everywhere; posterior draws induce the same pool ranking."
        )
    selected = _select_top_by_score(refset_indices, scores, num_add=num_add)
    logging.info(f"rankvar selected {len(selected)} pts")
    return selected, scores, log_marginal, p_var


def extend_reference_set_batchbald(
    scoring_model,
    refset_indices,
    target_vectors,
    num_add=10,
    scoring_model_fail=None,
    target_vectors_fail=None,
    prior_pass=None,
    prior_fail=None,
    n_candidates=None,
):
    """BatchBALD greedy selection on dual-GMM MCMC predictive probabilities.

    Builds ``probs`` of shape ``(S, N, 2)`` from paired pass/fail posterior
    draws, optionally shortlists by pointwise BALD, then greedily maximizes
    joint mutual information (Kirsch et al., 2019).

    Parameters
    ----------
    num_add : int
        Batch size (number of points to acquire).
    n_candidates : int or None
        Optional BALD shortlist size before the joint BatchBALD step.
        ``None`` uses ``max(30, 3 * num_add)`` capped by the candidate pool.

    Returns
    -------
    selected : list[int]
        BatchBALD-selected indices (acquisition order).
    bald_scores : ndarray
        Per-point BALD scores (for plotting / diagnostics).
    aleatoric : ndarray
        Per-point aleatoric uncertainty E_t[H[p_t]].
    log_marginal : ndarray
        Per-point E_t[log p(x|θ_t)] under the dual-class mixture.
    p_var : ndarray
        Per-point Var_t[p(y=pass | x, θ_t)] (MCMC disagreement).
    """
    if scoring_model_fail is None or target_vectors_fail is None:
        raise ValueError(
            "extend_reference_set_batchbald requires dual GMM "
            "(scoring_model_fail / target_vectors_fail)."
        )
    if prior_pass is None:
        prior_pass = 0.5
    if prior_fail is None:
        prior_fail = 0.5

    probs, log_marginal = dual_gmm_predictive_probs(
        scoring_model,
        scoring_model_fail,
        target_vectors,
        target_vectors_fail,
        prior_pass=prior_pass,
        prior_fail=prior_fail,
    )
    # Pointwise BALD for shortlist + visualization (same as extend_reference_set_bald).
    p_t = probs[:, :, 0]
    p_bar = p_t.mean(axis=0)
    aleatoric = _binary_shannon_entropy(p_t).mean(axis=0)
    scores = np.maximum(_binary_shannon_entropy(p_bar) - aleatoric, 0.0)
    p_var = np.var(p_t, axis=0)
    _log_dual_bald_diagnostics(
        scores, aleatoric, log_marginal, refset_indices, prior_pass, prior_fail, tag="batchbald"
    )

    refset = set(int(i) for i in refset_indices)
    candidate_indices = [i for i in range(probs.shape[1]) if i not in refset]
    if len(candidate_indices) < num_add:
        raise ValueError(
            f"Not enough selectable indices to add {num_add} new indices."
        )

    if n_candidates is None:
        n_candidates = max(30, 3 * int(num_add))
    n_candidates = int(min(max(int(num_add), n_candidates), len(candidate_indices)))
    cand = np.asarray(candidate_indices, dtype=int)
    shortlist = cand[np.argsort(scores[cand])[::-1][:n_candidates]]

    selected, mi_scores = batch_bald_greedy(
        probs, batch_size=int(num_add), candidate_idx=shortlist
    )
    assert set(selected).isdisjoint(refset), (
        "Selected indices must not overlap with reference set indices. "
        f"Current reference set indices: {refset_indices}, selected indices: {selected}"
    )
    logging.info(
        f"batchbald (dual) selected {len(selected)} pts "
        f"(shortlist={n_candidates}, incremental MI={mi_scores})"
    )
    return selected, scores, aleatoric, log_marginal, p_var


# Ranking EPIG for Spearman. Labels lie on {0, 1/n, ..., 1} (n = 10, 20, or 50).
# Events are strict pairwise order of scores snapped to that grid; same-bin
# pairs are ties. The score is the expected drop in q(1-q) after a binomial label.
_EPIG_ETA_EPS = 1e-6


def _as_rng(random_state):
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _ess(sample_weight):
    w = np.asarray(sample_weight, dtype=float)
    return float(1.0 / np.sum(np.square(w)))


def _clip_eta(eta):
    return np.clip(np.asarray(eta, dtype=float), _EPIG_ETA_EPS, 1.0 - _EPIG_ETA_EPS)


def _sample_rank_pairs(mean_eta, n_pairs, top_bias, top_k, rng):
    """Sample index pairs. ``top_bias`` is the fraction with one end in the current top-k."""
    mean_eta = np.asarray(mean_eta, dtype=float)
    n = int(mean_eta.shape[0])
    if n < 2 or n_pairs < 1:
        return np.zeros((0, 2), dtype=int)
    n_pairs = int(n_pairs)
    n_biased = int(round(float(np.clip(top_bias, 0.0, 1.0)) * n_pairs))
    n_biased = min(max(n_biased, 0), n_pairs)
    n_uniform = n_pairs - n_biased
    blocks = []
    if n_uniform:
        left = rng.integers(0, n, size=n_uniform)
        right = rng.integers(0, n - 1, size=n_uniform)
        right = right + (right >= left)
        blocks.append(np.stack([left, right], axis=1))
    if n_biased:
        k = min(max(int(top_k), 1), n)
        order = np.argsort(mean_eta)[::-1]
        top = order[:k]
        rest = order[k:] if k < n else order
        left = rng.choice(top, size=n_biased, replace=True)
        right = rng.choice(rest, size=n_biased, replace=True)
        same = left == right
        if np.any(same):
            right = right.copy()
            right[same] = (right[same] + 1) % n
        blocks.append(np.stack([left, right], axis=1))
    return np.concatenate(blocks, axis=0)


def _snap_counts(eta, n_trials):
    """Nearest repeated-trial count in ``0 .. n_trials``. Ties at .5 follow ``rint``."""
    n_trials = int(n_trials)
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    counts = np.rint(np.asarray(eta, dtype=float) * n_trials)
    return np.clip(counts, 0, n_trials).astype(int)


def _log_binomial_coefficients(n_trials):
    n_trials = int(n_trials)
    ks = np.arange(n_trials + 1, dtype=float)
    return gammaln(n_trials + 1) - gammaln(ks + 1) - gammaln(n_trials - ks + 1)


def build_ranking_event_matrix(
    eta,
    n_trials=10,
    n_pairs=2048,
    rank_indices=None,
    random_state=None,
):
    """Spearman pairwise-order events on the label grid.

    ``eta`` is ``[S, N]``. Scores are snapped to ``{0, 1/n, ..., 1}`` with
    ``n = n_trials`` before comparison, so two predictions in the same bin are
    tied. Each column is one sampled pair: ``T = 1`` when i's snapped count is
    strictly above j's. Columns constant across posterior samples are dropped.
    ``rank_indices`` restricts the pool (default: every point).
    """
    eta = np.asarray(eta, dtype=float)
    if eta.ndim != 2:
        raise ValueError(f"eta must have shape (S, N), got {eta.shape}")
    n_trials = int(n_trials)
    _s, n = eta.shape
    if rank_indices is None:
        rank_idx = np.arange(n, dtype=int)
    else:
        rank_idx = np.asarray(list(rank_indices), dtype=int)
        if rank_idx.size == 0:
            raise ValueError("rank_indices is empty.")
        if int(rank_idx.min()) < 0 or int(rank_idx.max()) >= n:
            raise ValueError("rank_indices out of range for eta.")
    counts = _snap_counts(eta[:, rank_idx], n_trials)
    n_rank = int(counts.shape[1])
    if n_rank < 2 or int(n_pairs) < 1:
        raise ValueError("Spearman EPIG needs at least two ranked points and one pair.")
    rng = _as_rng(random_state)
    pairs = _sample_rank_pairs(counts.mean(axis=0), int(n_pairs), 0.0, 1, rng)
    left = counts[:, pairs[:, 0]]
    right = counts[:, pairs[:, 1]]
    events = (left > right).astype(float)
    vary = events.min(axis=0) < events.max(axis=0)
    if not np.any(vary):
        raise ValueError(
            "Every pairwise order is constant across posterior samples, so EPIG is zero."
        )
    events = events[:, vary]
    return events, np.ones(events.shape[1], dtype=float)


def _bernoulli_variance_utility(q, event_weight):
    q = np.clip(np.asarray(q, dtype=float), 0.0, 1.0)
    return (event_weight * q * (1.0 - q)).sum(axis=-1)


# Candidates per CUDA matmul. posterior_q is (K, C, n_pairs) float32.
_EPIG_CUDA_CHUNK = 4096
_EPIG_CUDA_LOGGED = False


def _expected_ranking_utility_cuda(weights_k, probs_k, events, event_weight):
    """Batched ``W_k^T @ events`` and Bernoulli utility on CUDA.

    ``weights_k`` is ``(S, C, K)``, ``probs_k`` is ``(C, K)``, ``events`` is
    ``(S, E)``. Returns expected utility of shape ``(C,)`` in float64.
    """
    device = torch.device("cuda")
    weights = torch.as_tensor(
        np.ascontiguousarray(weights_k), dtype=torch.float32, device=device
    ).permute(2, 1, 0)
    ev = torch.as_tensor(
        np.ascontiguousarray(events), dtype=torch.float32, device=device
    )
    posterior_q = torch.matmul(weights, ev).clamp_(0.0, 1.0)
    ew = torch.as_tensor(
        np.ascontiguousarray(event_weight), dtype=torch.float32, device=device
    )
    utility = (ew * posterior_q * (1.0 - posterior_q)).sum(dim=-1)
    probs = torch.as_tensor(
        np.ascontiguousarray(probs_k), dtype=torch.float32, device=device
    )
    expected = (probs * utility.transpose(0, 1)).sum(dim=-1)
    return expected.detach().cpu().numpy().astype(np.float64, copy=False)


def _expected_ranking_utility_numpy(weights_k, probs_k, events, event_weight):
    expected_u = np.zeros(weights_k.shape[1], dtype=float)
    for k in range(weights_k.shape[2]):
        posterior_q = weights_k[:, :, k].T @ events
        expected_u += probs_k[:, k] * _bernoulli_variance_utility(
            posterior_q, event_weight
        )
    return expected_u


def _normalize_sample_weight(sample_weight, n_samples):
    if sample_weight is None:
        return np.full(n_samples, 1.0 / n_samples, dtype=float)
    sample_weight = np.asarray(sample_weight, dtype=float).ravel()
    if sample_weight.shape[0] != n_samples:
        raise ValueError(
            f"sample_weight length {sample_weight.shape[0]} != n_samples {n_samples}"
        )
    total = float(sample_weight.sum())
    if (not np.isfinite(total)) or total <= 0.0:
        raise ValueError("sample_weight must have positive finite mass.")
    return sample_weight / total


def _discrete_label_posterior(sample_weight, eta_c, n_trials):
    """Binomial label model on ``k = 0 .. n_trials`` successes.

    ``eta_c`` is ``[S, C]``, ``P(one trial succeeds | θ_s, x_c)``.
    Returns ``W`` of shape ``[S, C, K]`` (posterior over samples given ``k``)
    and ``p`` of shape ``[C, K]`` (marginal probability of each count).
    ``n_trials=1`` is the Bernoulli update ``w ∝ η`` or ``w ∝ 1-η``.
    """
    n_trials = int(n_trials)
    ks = np.arange(n_trials + 1, dtype=float)
    log_eta = np.log(eta_c)
    log_one = np.log(1.0 - eta_c)
    log_lik = (
        ks[None, None, :] * log_eta[:, :, None]
        + (n_trials - ks)[None, None, :] * log_one[:, :, None]
    )
    log_w = np.log(np.clip(sample_weight, 1e-300, None))[:, None, None] + log_lik
    max_log = np.max(log_w, axis=0, keepdims=True)
    unnorm = np.exp(log_w - max_log)
    weights = unnorm / np.maximum(unnorm.sum(axis=0, keepdims=True), 1e-300)
    log_mass = max_log[0] + np.log(np.maximum(unnorm.sum(axis=0), 1e-300))
    log_mass = log_mass + _log_binomial_coefficients(n_trials)[None, :]
    log_mass -= np.max(log_mass, axis=1, keepdims=True)
    probs = np.exp(log_mass)
    probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-300)
    return weights, probs


def ranking_epig_scores(
    eta,
    events,
    event_weight=None,
    sample_weight=None,
    candidate_idx=None,
    chunk_size=256,
    n_trials=10,
):
    """Expected drop in ranking-event variance from one grid label.

    ``eta[s, i] = P(one trial succeeds | x_i, θ_s)``. The label is a count
    ``k`` out of ``n_trials`` (so ``y = k/n`` on the evaluation grid), with

        w_s(k) ∝ w_s * η_s^k * (1-η_s)^(n-k)
        score = U(q) - Σ_k P(k) U(q | k)

    ``n_trials=1`` reduces to the binary formula

        score = U(q0) - p1 * U(W1.T @ T) - (1-p1) * U(W0.T @ T).

    Returns scores of shape ``(N,)``; non-candidates are ``nan``.
    """
    eta = _clip_eta(eta)
    events = np.asarray(events, dtype=float)
    if eta.ndim != 2 or events.ndim != 2:
        raise ValueError(f"eta {eta.shape} and events {events.shape} must be 2D.")
    s, n = eta.shape
    if events.shape[0] != s:
        raise ValueError(
            f"events has {events.shape[0]} rows but eta has {s} posterior samples."
        )
    if event_weight is None:
        event_weight = np.ones(events.shape[1], dtype=float)
    else:
        event_weight = np.asarray(event_weight, dtype=float).ravel()
    if event_weight.shape[0] != events.shape[1]:
        raise ValueError(
            f"event_weight length {event_weight.shape[0]} != n_events {events.shape[1]}"
        )
    sample_weight = _normalize_sample_weight(sample_weight, s)
    n_trials = int(n_trials)
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")

    if candidate_idx is None:
        candidate_idx = np.arange(n, dtype=int)
    else:
        candidate_idx = np.asarray(candidate_idx, dtype=int)
    scores = np.full(n, np.nan, dtype=float)
    if candidate_idx.size == 0:
        return scores

    q0 = sample_weight @ events
    u0 = float(_bernoulli_variance_utility(q0, event_weight))
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        global _EPIG_CUDA_LOGGED
        if not _EPIG_CUDA_LOGGED:
            logging.info(
                "EPIG ranking matmul on CUDA float32 "
                f"(chunk={_EPIG_CUDA_CHUNK})"
            )
            _EPIG_CUDA_LOGGED = True
    chunk = _EPIG_CUDA_CHUNK if use_cuda else max(int(chunk_size), 1)
    for start in range(0, candidate_idx.size, chunk):
        idx = candidate_idx[start : start + chunk]
        weights_k, probs_k = _discrete_label_posterior(
            sample_weight, eta[:, idx], n_trials
        )
        if use_cuda:
            expected_u = _expected_ranking_utility_cuda(
                weights_k, probs_k, events, event_weight
            )
        else:
            expected_u = _expected_ranking_utility_numpy(
                weights_k, probs_k, events, event_weight
            )
        # q(1-q) is concave in the Bayes mixture, so EPIG is nonnegative.
        # Clip only float dust that would otherwise rank below a true zero.
        scores[idx] = np.maximum(u0 - expected_u, 0.0)
    return scores


def _expected_label_reweight(sample_weight, eta_col, n_trials=10, eps=_EPIG_ETA_EPS):
    """Reweight samples by a soft observation of the expected success count.

    ``p = E[η]`` and ``k̂ = n_trials * p``, then
    ``w_s ∝ w_s * η_s^{k̂} * (1-η_s)^{n-k̂}``.
    """
    w = np.asarray(sample_weight, dtype=float)
    eta_col = np.clip(np.asarray(eta_col, dtype=float), eps, 1.0 - eps)
    p = float(np.clip(np.sum(w * eta_col), eps, 1.0 - eps))
    n_trials = int(n_trials)
    log_lik = (n_trials * p) * np.log(eta_col) + (n_trials * (1.0 - p)) * np.log(1.0 - eta_col)
    log_lik -= np.max(log_lik)
    updated = w * np.exp(log_lik)
    total = float(updated.sum())
    if (not np.isfinite(total)) or total <= 0.0:
        return np.full(w.shape[0], 1.0 / w.shape[0], dtype=float)
    return updated / total


def ranking_epig_select(
    eta,
    refset_indices,
    num_add=10,
    n_trials=10,
    n_pairs=2048,
    rank_indices=None,
    candidate_chunk=256,
    random_state=None,
):
    """Greedy batch of points maximizing Spearman EPIG.

    Labels are counts out of ``n_trials`` (y = k/n). Events are strict order of
    scores snapped to that grid. The first point maximizes EPIG under the current
    posterior. Each later point is chosen after reweighting samples by that
    point's expected success count.

    Returns
    -------
    selected : list[int]
    scores : ndarray, shape (N,)
        EPIG of every pool point under the *current* posterior (before the
        greedy updates). Already-labeled indices are 0.
    """
    eta = _clip_eta(eta)
    s, n = eta.shape
    if num_add < 1:
        raise ValueError(f"num_add must be >= 1, got {num_add}")
    refset = set(int(i) for i in refset_indices)
    candidate_idx = np.asarray([i for i in range(n) if i not in refset], dtype=int)
    if candidate_idx.size < num_add:
        raise ValueError(
            f"Not enough selectable indices to add {num_add} new indices."
        )
    rng = _as_rng(random_state)
    events, event_weight = build_ranking_event_matrix(
        eta,
        n_trials=n_trials,
        n_pairs=n_pairs,
        rank_indices=rank_indices,
        random_state=rng,
    )
    sample_weight = np.full(s, 1.0 / s, dtype=float)
    base_scores = ranking_epig_scores(
        eta,
        events,
        event_weight=event_weight,
        sample_weight=sample_weight,
        candidate_idx=candidate_idx,
        chunk_size=candidate_chunk,
        n_trials=n_trials,
    )
    scores_out = np.zeros(n, dtype=float)
    finite = np.isfinite(base_scores)
    scores_out[finite] = base_scores[finite]

    selected = []
    selected_set = set()
    incremental = []
    remaining = candidate_idx
    for _step in range(int(num_add)):
        if selected_set:
            remaining = np.asarray(
                [i for i in remaining if int(i) not in selected_set], dtype=int
            )
            step_scores = ranking_epig_scores(
                eta,
                events,
                event_weight=event_weight,
                sample_weight=sample_weight,
                candidate_idx=remaining,
                chunk_size=candidate_chunk,
                n_trials=n_trials,
            )
        else:
            step_scores = base_scores
        elig = step_scores[remaining]
        if not np.any(np.isfinite(elig)):
            raise RuntimeError("EPIG scores are all non-finite.")
        best = np.nanmax(elig)
        tied = remaining[np.flatnonzero(np.abs(elig - best) <= 1e-12)]
        pick = int(tied[0] if tied.size == 1 else rng.choice(tied))
        selected.append(pick)
        selected_set.add(pick)
        incremental.append(float(best))
        sample_weight = _expected_label_reweight(
            sample_weight, eta[:, pick], n_trials=n_trials
        )

    ess = _ess(sample_weight)
    logging.info(
        f"epig: S={s}, n_trials={int(n_trials)}, spearman pairs={events.shape[1]} "
        f"(sampled {int(n_pairs)}), "
        f"base score mean={np.nanmean(base_scores):.6g} max={np.nanmax(base_scores):.6g}, "
        f"greedy ESS={ess:.1f}/{s}, incremental={incremental}"
    )
    if ess < max(10.0, 0.05 * s):
        logging.warning(
            f"epig greedy effective sample size collapsed to {ess:.1f} "
            f"(S={s}). Increase the number of posterior samples."
        )
    assert set(selected).isdisjoint(refset)
    return selected, scores_out


# Spearman EPIG. Ranks are average ranks of clipped η (no k/n snap). The utility
# is the sum of rank variances, which is the squared Euclidean distance of a
# posterior ranking from the mean ranking. The label model is still Binomial(n_trials).
_EPIG_SPEARMAN_CHUNK = 128
_EPIG_SPEARMAN_CUDA_LOGGED = False


def _pool_rank_matrix(eta, rank_indices=None):
    """Average ranks of each draw. Higher score = higher rank. Shape ``(S, M)``."""
    eta = np.asarray(eta, dtype=float)
    if eta.ndim != 2:
        raise ValueError(f"eta must have shape (S, N), got {eta.shape}")
    _s, n = eta.shape
    if rank_indices is None:
        rank_idx = np.arange(n, dtype=int)
    else:
        rank_idx = np.asarray(list(rank_indices), dtype=int)
        if rank_idx.size == 0:
            raise ValueError("rank_indices is empty.")
        if int(rank_idx.min()) < 0 or int(rank_idx.max()) >= n:
            raise ValueError("rank_indices out of range for eta.")
    if rank_idx.size < 2:
        raise ValueError("Spearman EPIG needs at least two ranked points.")
    scored = np.array(eta[:, rank_idx], dtype=float, copy=True)
    scored[~np.isfinite(scored)] = -np.inf
    return rankdata(scored, method="average", axis=1).astype(np.float64, copy=False)


def _center_rank_matrix(ranks):
    """Column-center ranks. Variance is unchanged; the sums stay smaller."""
    ranks = np.asarray(ranks, dtype=np.float64)
    ranks_c = ranks - ranks.mean(axis=0, keepdims=True)
    q_row = np.einsum("sm,sm->s", ranks_c, ranks_c)
    return ranks_c, q_row


def _sum_rank_variance(sample_weight, ranks_c, q_row):
    """Σ_i Var_w(rank_i) for one normalized sample weight."""
    w = np.asarray(sample_weight, dtype=np.float64).ravel()
    total = float(w.sum())
    if (not np.isfinite(total)) or total <= 0.0:
        raise ValueError("sample_weight must have positive finite mass.")
    w = w / total
    mean = ranks_c.T @ w
    return float(w @ q_row - np.dot(mean, mean))


def _expected_rank_variance_numpy(weights_k, probs_k, ranks_c, q_row):
    """E_k[Σ_i Var(rank_i)] for ``weights_k`` of shape ``(S, C, K)``."""
    s, c, k = weights_k.shape
    w = np.ascontiguousarray(weights_k, dtype=np.float64).reshape(s, c * k)
    dot_q = q_row @ w
    proj = ranks_c.T @ w
    sum_var = dot_q - np.einsum("ac,ac->c", proj, proj)
    return (np.asarray(probs_k, dtype=np.float64) * sum_var.reshape(c, k)).sum(axis=1)


def _expected_rank_variance_cuda(weights_k, probs_k, ranks_c, q_row):
    """Float64 version of ``_expected_rank_variance_numpy`` on CUDA."""
    device = torch.device("cuda")
    w = torch.as_tensor(
        np.ascontiguousarray(weights_k), dtype=torch.float64, device=device
    )
    r = torch.as_tensor(
        np.ascontiguousarray(ranks_c), dtype=torch.float64, device=device
    )
    q = torch.as_tensor(
        np.ascontiguousarray(q_row), dtype=torch.float64, device=device
    )
    p = torch.as_tensor(
        np.ascontiguousarray(probs_k), dtype=torch.float64, device=device
    )
    dot_q = torch.einsum("s,sck->ck", q, w)
    proj = torch.einsum("sm,sck->mck", r, w)
    sum_var = dot_q - (proj * proj).sum(dim=0)
    return (p * sum_var).sum(dim=-1).detach().cpu().numpy()


def _spearman_chunk_size(n_rank, n_trials, chunk_size):
    """Keep the ``(M, C, K)`` posterior-rank tensor near 256 MiB."""
    k = int(n_trials) + 1
    per = max(int(n_rank), 1) * k * 8
    cap = max(1, (256 * 1024 * 1024) // per)
    return max(1, min(int(chunk_size), cap))


def ranking_epig_spearman_scores(
    eta,
    ranks_c,
    q_row,
    sample_weight=None,
    candidate_idx=None,
    chunk_size=_EPIG_SPEARMAN_CHUNK,
    n_trials=10,
):
    """Expected drop in Σ_i Var(rank_i) from one binomial label.

    ``ranks_c`` is the column-centered rank matrix of shape ``(S, M)``.
    ``q_row[s] = ||ranks_c[s]||^2``. Non-candidates are ``nan``.
    """
    eta = _clip_eta(eta)
    ranks_c = np.asarray(ranks_c, dtype=np.float64)
    q_row = np.asarray(q_row, dtype=np.float64).ravel()
    if eta.ndim != 2 or ranks_c.ndim != 2:
        raise ValueError(f"eta {eta.shape} and ranks {ranks_c.shape} must be 2D.")
    s, n = eta.shape
    if ranks_c.shape[0] != s or q_row.shape[0] != s:
        raise ValueError(
            f"ranks {ranks_c.shape} / q_row {q_row.shape} do not match S={s}."
        )
    sample_weight = _normalize_sample_weight(sample_weight, s)
    n_trials = int(n_trials)
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if candidate_idx is None:
        candidate_idx = np.arange(n, dtype=int)
    else:
        candidate_idx = np.asarray(candidate_idx, dtype=int)
    scores = np.full(n, np.nan, dtype=float)
    if candidate_idx.size == 0:
        return scores

    u0 = _sum_rank_variance(sample_weight, ranks_c, q_row)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        global _EPIG_SPEARMAN_CUDA_LOGGED
        if not _EPIG_SPEARMAN_CUDA_LOGGED:
            logging.info(
                "EPIG Spearman rank-variance on CUDA float64 "
                f"(chunk<={_EPIG_SPEARMAN_CHUNK})"
            )
            _EPIG_SPEARMAN_CUDA_LOGGED = True
    chunk = _spearman_chunk_size(ranks_c.shape[1], n_trials, chunk_size)
    for start in range(0, candidate_idx.size, chunk):
        idx = candidate_idx[start : start + chunk]
        weights_k, probs_k = _discrete_label_posterior(
            sample_weight, eta[:, idx], n_trials
        )
        if use_cuda:
            expected_u = _expected_rank_variance_cuda(
                weights_k, probs_k, ranks_c, q_row
            )
        else:
            expected_u = _expected_rank_variance_numpy(
                weights_k, probs_k, ranks_c, q_row
            )
        # Law of total variance: E[Var | k] <= Var, up to float dust.
        scores[idx] = np.maximum(u0 - expected_u, 0.0)
    return scores


def ranking_epig_spearman_select(
    eta,
    refset_indices,
    num_add=10,
    n_trials=10,
    rank_indices=None,
    candidate_chunk=_EPIG_SPEARMAN_CHUNK,
    random_state=None,
):
    """Greedy batch maximizing the expected drop in total rank variance.

    Ranks use raw (clipped) η with average ties, not the ``k/n`` label grid.
    The first point maximizes the drop under the current posterior. Each later
    point is chosen after reweighting samples by that point's expected success
    count.

    Returns
    -------
    selected : list[int]
    scores : ndarray, shape (N,)
        Score of every pool point under the current posterior, before the
        greedy updates. Already-labeled indices are 0.
    """
    eta = _clip_eta(eta)
    s, n = eta.shape
    if num_add < 1:
        raise ValueError(f"num_add must be >= 1, got {num_add}")
    refset = set(int(i) for i in refset_indices)
    candidate_idx = np.asarray([i for i in range(n) if i not in refset], dtype=int)
    if candidate_idx.size < num_add:
        raise ValueError(
            f"Not enough selectable indices to add {num_add} new indices."
        )
    ranks_c, q_row = _center_rank_matrix(_pool_rank_matrix(eta, rank_indices))
    rng = _as_rng(random_state)
    sample_weight = np.full(s, 1.0 / s, dtype=float)
    base_scores = ranking_epig_spearman_scores(
        eta,
        ranks_c,
        q_row,
        sample_weight=sample_weight,
        candidate_idx=candidate_idx,
        chunk_size=candidate_chunk,
        n_trials=n_trials,
    )
    scores_out = np.zeros(n, dtype=float)
    finite = np.isfinite(base_scores)
    scores_out[finite] = base_scores[finite]

    selected = []
    selected_set = set()
    incremental = []
    remaining = candidate_idx
    for _step in range(int(num_add)):
        if selected_set:
            remaining = np.asarray(
                [i for i in remaining if int(i) not in selected_set], dtype=int
            )
            step_scores = ranking_epig_spearman_scores(
                eta,
                ranks_c,
                q_row,
                sample_weight=sample_weight,
                candidate_idx=remaining,
                chunk_size=candidate_chunk,
                n_trials=n_trials,
            )
        else:
            step_scores = base_scores
        elig = step_scores[remaining]
        if not np.any(np.isfinite(elig)):
            raise RuntimeError("Spearman EPIG scores are all non-finite.")
        best = np.nanmax(elig)
        tol = 1e-8 * max(1.0, abs(float(best)))
        tied = remaining[np.flatnonzero(np.abs(elig - best) <= tol)]
        pick = int(tied[0] if tied.size == 1 else rng.choice(tied))
        selected.append(pick)
        selected_set.add(pick)
        incremental.append(float(best))
        sample_weight = _expected_label_reweight(
            sample_weight, eta[:, pick], n_trials=n_trials
        )

    ess = _ess(sample_weight)
    logging.info(
        f"epigspearman: S={s}, ranked={ranks_c.shape[1]} (raw η, average ties), "
        f"n_trials={int(n_trials)}, "
        f"base score mean={np.nanmean(base_scores):.6g} max={np.nanmax(base_scores):.6g}, "
        f"greedy ESS={ess:.1f}/{s}, incremental={incremental}"
    )
    if ess < max(10.0, 0.05 * s):
        logging.warning(
            f"epigspearman greedy effective sample size collapsed to {ess:.1f} "
            f"(S={s}). Increase the number of posterior samples."
        )
    assert set(selected).isdisjoint(refset)
    return selected, scores_out


def _model_has_posterior_draws(model):
    if model is None:
        return False
    has_posterior = getattr(model, "has_mcmc_posterior", None)
    if callable(has_posterior) and not has_posterior():
        return False
    return callable(getattr(model, "predictive_log_likelihoods_over_draws", None))


def _eta_from_dual_mcmc(
    scoring_model_pass,
    scoring_model_fail,
    target_vectors_pass,
    target_vectors_fail,
    prior_pass,
    prior_fail,
    n_draws,
):
    kwargs = dict(
        scoring_model_pass=scoring_model_pass,
        scoring_model_fail=scoring_model_fail,
        target_vectors_pass=target_vectors_pass,
        target_vectors_fail=target_vectors_fail,
        prior_pass=prior_pass,
        prior_fail=prior_fail,
        n_draws=n_draws,
    )
    try:
        return dual_gmm_class_posteriors(**kwargs)
    except ValueError as exc:
        msg = str(exc)
        if n_draws is not None and "posterior samples" in msg:
            logging.warning(
                f"EPIG requested n_draws={n_draws} but the stored posterior is smaller "
                f"({msg}). Falling back to the scorer default."
            )
            kwargs["n_draws"] = None
            return dual_gmm_class_posteriors(**kwargs)
        raise


def _gmm_bootstrap_hyperparams(scoring_model):
    cfg = getattr(scoring_model, "config", None)
    n_components = int(scoring_model._n_components())
    covariance_type = str(getattr(cfg, "covariance_type", "diag"))
    reg_covar = float(getattr(cfg, "reg_covar", 1e-6))
    max_iter = int(getattr(cfg, "max_iter", getattr(cfg, "sklearn_max_iter", 200)))
    tol = float(getattr(cfg, "tol", getattr(cfg, "sklearn_tol", 1e-3)))
    return n_components, covariance_type, reg_covar, max_iter, tol


def _resample_rows(rng, n_rows, sample_weight):
    if n_rows < 1:
        raise ValueError("Cannot bootstrap an empty class.")
    if sample_weight is None:
        return rng.integers(0, n_rows, size=n_rows)
    p = np.clip(np.asarray(sample_weight, dtype=float).ravel(), 0.0, None)
    if p.shape[0] != n_rows:
        raise ValueError(
            f"class sample_weight length {p.shape[0]} != n_rows {n_rows}"
        )
    total = float(p.sum())
    if (not np.isfinite(total)) or total <= 0.0:
        return rng.integers(0, n_rows, size=n_rows)
    return rng.choice(n_rows, size=n_rows, replace=True, p=p / total)


def _fit_gmm_loglik(X_train, X_pool, *, n_components, covariance_type, reg_covar, max_iter, tol, random_state):
    n_train = int(len(X_train))
    k = int(min(n_components, n_train))
    if k < 1:
        raise ValueError("GMM bootstrap needs at least one training row.")
    model = GaussianMixture(
        n_components=k,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        max_iter=max_iter,
        tol=tol,
        n_init=1,
        init_params="k-means++",
        random_state=int(random_state),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X_train)
    return model.score_samples(np.asarray(X_pool))


def bootstrap_dual_gmm_eta(
    X_pass_train,
    X_fail_train,
    X_pass_pool,
    X_fail_pool,
    *,
    n_components_pass,
    n_components_fail,
    covariance_type="diag",
    reg_covar=1e-6,
    max_iter=200,
    tol=1e-3,
    n_boot=200,
    prior_pass=0.5,
    prior_fail=0.5,
    sample_weight_pass=None,
    sample_weight_fail=None,
    random_state=None,
):
    """Bootstrap dual-GMM pass posteriors. Returns ``eta`` of shape ``(n_boot, N)``.

    Each replicate resamples the labeled pass set and the labeled fail set, refits
    both GMMs (``n_init=1``), and sets ``η_s(x) = p(y=1 | x, θ_s)``.
    """
    X_pass_train = np.asarray(X_pass_train, dtype=float)
    X_fail_train = np.asarray(X_fail_train, dtype=float)
    X_pass_pool = np.asarray(X_pass_pool, dtype=float)
    X_fail_pool = np.asarray(X_fail_pool, dtype=float)
    if X_pass_pool.shape[0] != X_fail_pool.shape[0]:
        raise ValueError(
            f"Pass/fail pool sizes differ: {X_pass_pool.shape[0]} vs {X_fail_pool.shape[0]}"
        )
    if len(X_pass_train) < 2 or len(X_fail_train) < 2:
        raise ValueError(
            "EPIG bootstrap needs at least 2 pass and 2 fail reference points "
            f"(got n_pass={len(X_pass_train)}, n_fail={len(X_fail_train)})."
        )
    n_boot = int(n_boot)
    if n_boot < 2:
        raise ValueError(f"n_boot must be >= 2, got {n_boot}")
    rng = _as_rng(random_state)
    n_pool = int(X_pass_pool.shape[0])
    eta = np.empty((n_boot, n_pool), dtype=float)
    prior_pass = float(prior_pass)
    prior_fail = float(prior_fail)
    log_prior_pass = np.log(prior_pass)
    log_prior_fail = np.log(prior_fail)
    logging.info(
        f"EPIG bootstrap: {n_boot} dual GMM pairs "
        f"(n_pass={len(X_pass_train)}, n_fail={len(X_fail_train)}, "
        f"k_pass={n_components_pass}, k_fail={n_components_fail})"
    )
    for s in range(n_boot):
        pass_rows = _resample_rows(rng, len(X_pass_train), sample_weight_pass)
        fail_rows = _resample_rows(rng, len(X_fail_train), sample_weight_fail)
        seed = int(rng.integers(0, 2**31 - 1))
        logp_pass = _fit_gmm_loglik(
            X_pass_train[pass_rows],
            X_pass_pool,
            n_components=n_components_pass,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            max_iter=max_iter,
            tol=tol,
            random_state=seed,
        )
        logp_fail = _fit_gmm_loglik(
            X_fail_train[fail_rows],
            X_fail_pool,
            n_components=n_components_fail,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            max_iter=max_iter,
            tol=tol,
            random_state=seed + 1,
        )
        log_joint_pass = logp_pass + log_prior_pass
        log_joint_fail = logp_fail + log_prior_fail
        eta[s] = np.exp(log_joint_pass - np.logaddexp(log_joint_pass, log_joint_fail))
    return eta


def extend_reference_set_epig(
    scoring_model,
    refset_indices,
    target_vectors,
    num_add=10,
    scoring_model_fail=None,
    target_vectors_fail=None,
    prior_pass=None,
    prior_fail=None,
    labels=None,
    sample_weight=None,
    rank_indices=None,
    n_trials=10,
    n_pairs=2048,
    n_draws=200,
    n_boot=200,
    candidate_chunk=256,
    random_state=None,
    objective="pairwise",
):
    """Acquire a batch by ranking-targeted EPIG.

    Posterior samples ``η_s(x) = p(y=1 | x, θ_s)`` come from paired dual-GMM
    MCMC draws when both scorers have a posterior. Otherwise the labeled
    reference set is bootstrapped and a pass/fail GMM pair is refit ``n_boot``
    times (sklearn GMM path).

    ``objective="pairwise"`` (default) scores strict pairwise order of η snapped
    to ``{0, 1/n, ..., 1}``. Same-bin pairs are ties, and every pair has the
    same weight. ``objective="spearman"`` ranks clipped η with average ties
    (no snap) and scores the expected drop in Σ_i Var(rank_i).

    A hypothetical label is a binomial count ``k`` out of ``n_trials``, and the
    score marginalizes ``k``. ``rank_indices`` restricts the ranking to the
    evaluated target pool. Already-labeled indices are not selected. A batch is
    built greedily: after each pick, sample weights are updated with that
    point's expected success count.
    """
    if objective not in ("pairwise", "spearman"):
        raise ValueError(
            f"objective must be 'pairwise' or 'spearman', got {objective!r}"
        )
    if scoring_model_fail is None or target_vectors_fail is None:
        raise ValueError(
            "extend_reference_set_epig requires dual GMM "
            "(scoring_model_fail / target_vectors_fail)."
        )
    if prior_pass is None:
        prior_pass = 0.5
    if prior_fail is None:
        prior_fail = 0.5
    target_vectors = np.asarray(target_vectors)
    target_vectors_fail = np.asarray(target_vectors_fail)
    if target_vectors.shape[0] != target_vectors_fail.shape[0]:
        raise ValueError(
            f"Pass/fail pool sizes differ: {target_vectors.shape[0]} vs {target_vectors_fail.shape[0]}"
        )

    if _model_has_posterior_draws(scoring_model) and _model_has_posterior_draws(scoring_model_fail):
        eta, _log_marginal = _eta_from_dual_mcmc(
            scoring_model,
            scoring_model_fail,
            target_vectors,
            target_vectors_fail,
            prior_pass,
            prior_fail,
            n_draws,
        )
        logging.info(f"EPIG posterior source: MCMC draws S={eta.shape[0]}")
    else:
        if labels is None:
            raise ValueError(
                "EPIG bootstrap requires labels aligned with the pool "
                "(only reference-set rows are used)."
            )
        ref = np.asarray(list(refset_indices), dtype=int)
        y_bin = np.asarray(labels, dtype=float)[ref] > 0.5
        if not np.any(y_bin) or not np.any(~y_bin):
            raise ValueError(
                "EPIG bootstrap needs both pass and fail references "
                f"(n_pass={int(np.sum(y_bin))}, n_fail={int(np.sum(~y_bin))})."
            )
        sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if sw is not None and sw.shape[0] != ref.shape[0]:
            raise ValueError(
                f"sample_weight length {sw.shape[0]} != n_ref {ref.shape[0]}"
            )
        k_pass, cov_pass, reg_pass, max_iter_pass, tol_pass = _gmm_bootstrap_hyperparams(
            scoring_model
        )
        k_fail, _cov_fail, reg_fail, max_iter_fail, tol_fail = _gmm_bootstrap_hyperparams(
            scoring_model_fail
        )
        eta = bootstrap_dual_gmm_eta(
            target_vectors[ref[y_bin]],
            target_vectors_fail[ref[~y_bin]],
            target_vectors,
            target_vectors_fail,
            n_components_pass=k_pass,
            n_components_fail=k_fail,
            covariance_type=cov_pass,
            reg_covar=max(reg_pass, reg_fail),
            max_iter=max(max_iter_pass, max_iter_fail),
            tol=min(tol_pass, tol_fail),
            n_boot=n_boot,
            prior_pass=prior_pass,
            prior_fail=prior_fail,
            sample_weight_pass=None if sw is None else sw[y_bin],
            sample_weight_fail=None if sw is None else sw[~y_bin],
            random_state=random_state,
        )
        logging.info(f"EPIG posterior source: GMM bootstrap S={eta.shape[0]}")

    if objective == "spearman":
        selected, scores = ranking_epig_spearman_select(
            eta,
            refset_indices,
            num_add=num_add,
            n_trials=n_trials,
            rank_indices=rank_indices,
            candidate_chunk=candidate_chunk,
            random_state=random_state,
        )
        logging.info(f"epigspearman selected {len(selected)} pts")
    elif objective == "pairwise":
        selected, scores = ranking_epig_select(
            eta,
            refset_indices,
            num_add=num_add,
            n_trials=n_trials,
            n_pairs=n_pairs,
            rank_indices=rank_indices,
            candidate_chunk=candidate_chunk,
            random_state=random_state,
        )
        logging.info(f"epig selected {len(selected)} pts")
    return selected, scores


def extend_reference_set_random(refset_indices, target_vectors, num_add=10, random_state=None):
    candidate_indices = [i for i in range(len(target_vectors)) if i not in refset_indices]
    if len(candidate_indices) < num_add:
        raise ValueError("Not enough selectable indices to add {} new indices.".format(num_add))

    if random_state is None:
        selected_indices = np.random.choice(candidate_indices, size=num_add, replace=False).tolist()
    else:
        selected_indices = np.random.default_rng(random_state).choice(
            candidate_indices, size=num_add, replace=False
        ).tolist()
    
    assert set(selected_indices).isdisjoint(set(refset_indices)), "Selected indices must not overlap with reference set indices. Current reference set indices: {}, selected indices: {}".format(refset_indices, selected_indices)
    
    return selected_indices

def extend_reference_set_diversity(refset_indices, target_vectors, num_add=10, metric='euclidean'):
    candidate_indices = [i for i in range(len(target_vectors)) if i not in refset_indices]
    if len(candidate_indices) < num_add:
        raise ValueError("Not enough selectable indices to add {} new indices.".format(num_add))
    
    selected_candidates = candidate_indices
    
    base = cdist(target_vectors[selected_candidates], target_vectors[refset_indices], metric=metric)  # (Nc, |R|)
    min_d = base.min(axis=1)                                                # (Nc,)
    
    picked_mask = np.zeros(len(selected_candidates), dtype=bool)
    chosen_local = []
    
    for _ in range(min(num_add, len(selected_candidates))):
        scores = np.where(picked_mask, -np.inf, min_d)
        j = int(np.argmax(scores))
        if not np.isfinite(scores[j]):
            break
        picked_mask[j] = True
        chosen_local.append(j)

        d_new = cdist(target_vectors[selected_candidates], target_vectors[selected_candidates][j:j+1], metric=metric).ravel()  # (Nc,)
        d_new = np.where(np.isnan(d_new), 1.0 if metric == 'cosine' else np.nanmax(d_new[np.isfinite(d_new)]), d_new)
        min_d = np.minimum(min_d, d_new)

    top_n_diverse_indices = [selected_candidates[i] for i in chosen_local]
    
    assert set(top_n_diverse_indices).isdisjoint(set(refset_indices)), "Selected indices must not overlap with reference set indices. Current reference set indices: {}, selected indices: {}".format(refset_indices, top_n_diverse_indices)
    
    return top_n_diverse_indices


# Inverse-propensity weights for mixed random → uncertainty acquisition.
# Deterministic top-k has π ∈ {0,1}; we approximate π with a z-scored softmax
# and Poissonized inclusion π_i = 1 − (1 − p_i)^k.
IPW_MIN_PI = 1e-6


def srs_inclusion_probability(n_candidates, n_add, min_pi=IPW_MIN_PI):
    """First-order inclusion probability for SRS without replacement: k / N."""
    if n_candidates <= 0 or n_add <= 0:
        return float(min_pi)
    return float(np.clip(n_add / float(n_candidates), min_pi, 1.0))


def softmax_topk_inclusion_probabilities(
    scores,
    candidate_indices,
    selected_indices,
    n_add=None,
    min_pi=IPW_MIN_PI,
):
    """Approximate top-k inclusion probabilities from acquisition scores.

    ``p ∝ softmax(z-score(u))`` on the candidate pool, then
    ``π_i = clip(1 − (1 − p_i)^k, min_pi, 1)``.
    Returns an array aligned with ``selected_indices``.
    """
    candidate_indices = np.asarray(candidate_indices, dtype=int)
    selected_indices = np.asarray(selected_indices, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if n_add is None:
        n_add = int(selected_indices.size)
    n_cand = int(candidate_indices.size)
    if n_cand == 0 or n_add <= 0 or selected_indices.size == 0:
        return np.full(selected_indices.size, min_pi, dtype=float)

    cand_scores = scores[candidate_indices]
    cand_scores = np.where(np.isfinite(cand_scores), cand_scores, -np.inf)
    finite = np.isfinite(cand_scores) & (cand_scores > -np.inf)
    if (not np.any(finite)) or float(np.std(cand_scores[finite])) < 1e-12:
        p = np.full(n_cand, 1.0 / n_cand, dtype=float)
    else:
        s = cand_scores[finite]
        mu = float(np.mean(s))
        sd = float(np.std(s))
        if sd < 1e-12:
            sd = 1.0
        z = np.full(n_cand, -1e8, dtype=float)
        z[finite] = (cand_scores[finite] - mu) / sd
        z = z - np.max(z)
        exp_z = np.exp(z)
        p = exp_z / np.maximum(exp_z.sum(), 1e-300)

    pi_cand = np.clip(1.0 - np.power(1.0 - p, float(n_add)), min_pi, 1.0)
    index_to_pi = {int(idx): float(pi) for idx, pi in zip(candidate_indices, pi_cand)}
    return np.array(
        [index_to_pi.get(int(idx), min_pi) for idx in selected_indices],
        dtype=float,
    )


def ipw_from_inclusion_probabilities(pi, min_pi=IPW_MIN_PI):
    pi = np.clip(np.asarray(pi, dtype=float), min_pi, 1.0)
    return 1.0 / pi


def min_distance_diversity_scores(target_vectors, refset_indices, metric='euclidean'):
    """Higher = farther from the current reference set (greedy diversity proxy)."""
    target_vectors = np.asarray(target_vectors)
    n = len(target_vectors)
    ref_idx = np.asarray(list(refset_indices), dtype=int)
    if n == 0:
        return np.zeros(0, dtype=float)
    if ref_idx.size == 0:
        return np.ones(n, dtype=float)
    d = cdist(target_vectors, target_vectors[ref_idx], metric=metric)
    d = np.where(np.isfinite(d), d, np.nan)
    with np.errstate(all='ignore'):
        scores = np.nanmin(d, axis=1)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores[ref_idx] = 0.0
    return scores


def al_ipw_weights_for_new_points(
    n_pool,
    current_ref_indices,
    new_indices,
    new_unc_indices=None,
    new_div_indices=None,
    acquisition_scores=None,
    pool_vectors=None,
    force_random=False,
):
    """Horvitz–Thompson weights ``w=1/π`` for a newly acquired AL batch.

    Random / SRS uses ``π = k / N``. Uncertainty / BALD uses a softmax-top-k
    approximation of the (deterministic) acquisition rule. Diversity uses the
    same softmax on min-distance-to-ref scores. Balanced splits the two.

    Returns
    -------
    weights : ndarray, shape (len(new_indices),)
    scheme : str
        ``'random'``, ``'uncertainty'``, ``'diversity'``, or ``'balanced'``.
    """
    new_indices = [int(i) for i in new_indices]
    if not new_indices:
        return np.zeros(0, dtype=float), 'random'

    current_ref = set(int(i) for i in current_ref_indices)
    candidates = [i for i in range(int(n_pool)) if i not in current_ref]
    new_unc = [int(i) for i in (new_unc_indices or [])]
    new_div = [int(i) for i in (new_div_indices or [])]
    k = len(new_indices)

    if force_random or (
        acquisition_scores is None
        and not new_div
        and not new_unc
    ):
        pi = srs_inclusion_probability(len(candidates), k)
        return np.full(k, 1.0 / pi, dtype=float), 'random'

    def _map_selected(selected, pi_sel, out):
        pos = {idx: t for t, idx in enumerate(new_indices)}
        w_sel = ipw_from_inclusion_probabilities(pi_sel)
        for j, idx in enumerate(selected):
            if idx in pos:
                out[pos[idx]] = float(w_sel[j])
        return out

    if new_unc and new_div:
        weights = np.ones(k, dtype=float)
        if acquisition_scores is None:
            pi_unc = np.full(
                len(new_unc),
                srs_inclusion_probability(len(candidates), len(new_unc)),
            )
        else:
            pi_unc = softmax_topk_inclusion_probabilities(
                acquisition_scores, candidates, new_unc, n_add=len(new_unc)
            )
        _map_selected(new_unc, pi_unc, weights)
        ref_after_unc = list(current_ref_indices) + new_unc
        cand_div = [i for i in range(int(n_pool)) if i not in set(int(x) for x in ref_after_unc)]
        if pool_vectors is None:
            pi_div = np.full(
                len(new_div),
                srs_inclusion_probability(len(cand_div), len(new_div)),
            )
        else:
            div_scores = min_distance_diversity_scores(pool_vectors, ref_after_unc)
            pi_div = softmax_topk_inclusion_probabilities(
                div_scores, cand_div, new_div, n_add=len(new_div)
            )
        _map_selected(new_div, pi_div, weights)
        return weights, 'balanced'

    if new_div and not new_unc:
        if pool_vectors is None:
            pi = srs_inclusion_probability(len(candidates), k)
            return np.full(k, 1.0 / pi, dtype=float), 'random'
        div_scores = min_distance_diversity_scores(pool_vectors, current_ref_indices)
        pi = softmax_topk_inclusion_probabilities(
            div_scores, candidates, new_indices, n_add=k
        )
        return ipw_from_inclusion_probabilities(pi), 'diversity'

    selected = new_unc if new_unc else new_indices
    if acquisition_scores is None:
        pi = srs_inclusion_probability(len(candidates), len(selected))
        return np.full(k, 1.0 / pi, dtype=float), 'random'
    pi = softmax_topk_inclusion_probabilities(
        acquisition_scores, candidates, selected, n_add=len(selected)
    )
    if selected == new_indices:
        return ipw_from_inclusion_probabilities(pi), 'uncertainty'
    weights = np.ones(k, dtype=float)
    _map_selected(selected, pi, weights)
    return weights, 'uncertainty'


# Two-stage (random then BALD) λ* weights from PCA-5 conservative n0 / R*.
N0_PCA_DIM = 5
N0_TOTAL_N_REF = 500
_FALLBACK_N0 = 30  # PCA-5 pos_detection upper, ceiled
_FALLBACK_N0_DOUBLE = 54  # same with u → u/2


def qinv_rstar(eps):
    eps = float(np.clip(eps, 1e-15, 1.0 - 1e-15))
    return float(norm.ppf(1.0 - eps))


def n0_from_rstar(rstar, p=N0_PCA_DIM, difficulty_scale=1.0):
    """Conservative integer n0 = ceil(p (1 + 1/u_eff)).

    u = [Q^{-1}(R*)]^2. ``difficulty_scale=2`` uses u_eff = u/2 (twice as hard).
    """
    q = qinv_rstar(rstar)
    u = float(q * q)
    scale = float(difficulty_scale) if difficulty_scale else 1.0
    if scale <= 0:
        scale = 1.0
    u_eff = u / scale
    if u_eff <= 0:
        fallback = _FALLBACK_N0_DOUBLE if scale > 1.5 else _FALLBACK_N0
        return fallback, u_eff
    n0 = int(math.ceil(float(p) * (1.0 + 1.0 / u_eff) - 1e-12))
    return max(n0, 1), u_eff


def lambda_star_two_stage(alpha, u):
    """Optimal second-stage weight λ* (Eq 14–15).

    γ = (√(α u) − u − 1)_+
    α₀* from Eq 15, then η, then λ*.
    ``alpha`` is 500 − n0 (remaining budget after the random start).
    """
    alpha = float(alpha)
    u = float(u)
    if alpha <= 0 or u <= 0:
        return 1.0, {'gamma': 0.0, 'alpha0_star': alpha, 'eta': 0.0}
    gamma = max(math.sqrt(alpha * u) - u - 1.0, 0.0)
    denom_a0 = u * gamma + u * u + u
    alpha0_star = (
        alpha
        - ((gamma + 1.0) ** 2 - 1.0) / u
        + ((u + 1.0) * gamma ** 2 + u * gamma) / denom_a0
    )
    num = (alpha - alpha0_star) * alpha * u
    den = (
        (alpha - alpha0_star) * (alpha * alpha0_star - 1.0) * (u + 1.0) * u
        + alpha * (alpha0_star - 1.0) * (alpha0_star * u + 1.0)
    )
    eta = 0.0 if abs(den) < 1e-18 else num / den
    frac_den = u * u + u * (gamma + 1.0)
    frac = 0.0 if abs(frac_den) < 1e-18 else (
        (u + 1.0) * (gamma + 1.0) * (1.0 - u * gamma) / frac_den
    )
    lam = 1.0 + eta * frac
    if (not math.isfinite(lam)) or lam <= 0:
        lam = 1.0
    return float(lam), {
        'gamma': float(gamma),
        'alpha0_star': float(alpha0_star),
        'eta': float(eta),
    }


def load_conservative_n0_lambda(
    task,
    *,
    pca_dim=N0_PCA_DIM,
    n_total=N0_TOTAL_N_REF,
    rstar_json=None,
    difficulty_scale=1.0,
):
    """PCA-5 upper R* → per-class n0 and BALD EM weight λ*.

    ``difficulty_scale=2`` uses u_eff = u/2 in both n0 and λ*.
    Returns (n0, lambda_star, info).
    """
    scale = float(difficulty_scale) if difficulty_scale else 1.0
    if rstar_json is None:
        here = Path(__file__).resolve()
        candidates = [
            here.parents[2] / "experiments" / "ber_estimation" / "results"
            / f"{task}_l22_pca{pca_dim}" / "recommended_rstar.json",
            Path.cwd() / "ber_estimation" / "results"
            / f"{task}_l22_pca{pca_dim}" / "recommended_rstar.json",
        ]
        rstar_json = next((p for p in candidates if p.is_file()), None)
    else:
        rstar_json = Path(rstar_json)

    if rstar_json is None or not Path(rstar_json).is_file():
        n0 = _FALLBACK_N0_DOUBLE if scale > 1.5 else _FALLBACK_N0
        u = None
        rstar = None
    else:
        rec = json.loads(Path(rstar_json).read_text())
        rstar = float(rec["Rstar_upper"])
        n0, u = n0_from_rstar(rstar, p=pca_dim, difficulty_scale=scale)

    alpha = float(n_total - n0)
    if u is None or u <= 0:
        lam, extra = 1.0, {'gamma': 0.0, 'alpha0_star': alpha, 'eta': 0.0}
        u = float("nan")
    else:
        lam, extra = lambda_star_two_stage(alpha, u)
    info = {
        'task': task,
        'pca_dim': int(pca_dim),
        'Rstar_upper': rstar,
        'u': u,
        'n0': int(n0),
        'alpha': float(alpha),
        'lambda_star': float(lam),
        'difficulty_scale': float(scale),
        'rstar_json': str(rstar_json) if rstar_json is not None else None,
        **extra,
    }
    return int(n0), float(lam), info
