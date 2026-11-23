"""Feature-dim and reg_covar CV helpers for Clotho iteration runs."""
from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import KFold

from clotho.metrics.reference_based.density_estimation import (
    GMMScorer,
    GMMScorerConfig,
    MCMCGMMScorer,
    MCMCGMMScorerConfig,
)
from clotho_run_helpers import (
    BAYES_GMM_MODELS,
    CV_MIN_FOLD_FLOOR,
    FEATURE_STEP,
    INIT_REG_COVAR_CV,
    MAX_REG_COVAR_CV,
    MIN_REG_COVAR_CV,
    PASS_THRESH,
    REG_COVAR_CV_FACTOR,
    _cheap_probe_n_init,
    _class_fit_labels_and_weight,
    _ensure_feature_rep,
    bayes_pass_posterior,
)

def get_next_n_features(LIH, LIH_rep, current_reference_set_indices, labels, current_n_features, max_n_features, dim_reduction_method, scorer_config, scoring_model='GMM', feature_step=FEATURE_STEP, target_class=1, d_schedule=False, sample_weight=None, success_rate_weight=False):
    next_n_features_higher = min(current_n_features + feature_step, max_n_features)
    next_n_features_lower = max(MIN_N_FEATURES, current_n_features - feature_step)
    if d_schedule:
        # Monotone up: never propose a lower dim.
        next_n_features_lower = current_n_features
    
    if next_n_features_higher not in LIH_rep:
        _ensure_feature_rep(LIH, LIH_rep, next_n_features_higher, dim_reduction_method)

    if next_n_features_lower not in LIH_rep:
        _ensure_feature_rep(LIH, LIH_rep, next_n_features_lower, dim_reduction_method)

    if next_n_features_lower != current_n_features and next_n_features_lower in LIH_rep:
        rep_LD = LIH_rep[next_n_features_lower]
    else:
        rep_LD = None

    if next_n_features_higher != current_n_features and next_n_features_higher in LIH_rep:
        rep_HD = LIH_rep[next_n_features_higher]
    else:
        rep_HD = None
    
    if rep_LD is not None or rep_HD is not None:
        return select_feature_dimension_by_CV(
            labels[current_reference_set_indices],
            np.asarray(LIH_rep[current_n_features][current_reference_set_indices]),
            np.asarray(rep_LD[current_reference_set_indices]) if rep_LD is not None else None,
            np.asarray(rep_HD[current_reference_set_indices]) if rep_HD is not None else None,
            scorer_config,
            scoring_model=scoring_model,
            target_class=target_class,
            d_schedule=d_schedule,
            sample_weight=sample_weight,
            success_rate_weight=success_rate_weight,
        )
        
    else:
        return current_n_features
    

def select_feature_dimension_by_CV(reference_scores, rep_original, rep_LD, rep_HD, scorer_config, scoring_model='GMM', target_class=1, d_schedule=False, sample_weight=None, success_rate_weight=False):
    assert not (rep_LD is None and rep_HD is None), "At least one of rep_LD or rep_HD must be provided."
    
    kf = KFold(n_splits=3, shuffle=True, random_state=scorer_config.random_state)
    score_correlations = {
        'LD': [],
        'original': [],
        'HD': [],
    }

    train_labels_binary = (np.asarray(reference_scores) > PASS_THRESH).astype(int)
    fit_labels, class_w = _class_fit_labels_and_weight(
        train_labels_binary,
        reference_scores,
        target_class,
        sample_weight=sample_weight,
        success_rate_weight=success_rate_weight,
    )
    # Fail-GMM density should track failure (low pass rate) → correlate with 1 - score.
    if target_class == 1:
        corr_targets = np.asarray(reference_scores, dtype=float)
    else:
        corr_targets = 1.0 - np.asarray(reference_scores, dtype=float)
    n_components_for_cv = _cv_n_components(scorer_config)

    scorer_original = construct_cv_scorer(scorer_config, scoring_model=scoring_model)
    scorer_LD = construct_cv_scorer(scorer_config, scoring_model=scoring_model)
    scorer_HD = construct_cv_scorer(scorer_config, scoring_model=scoring_model)

    for train_idx, test_idx in kf.split(rep_original):
        if success_rate_weight and class_w is not None:
            num_effective_references = int(np.sum(np.asarray(class_w, dtype=float)[train_idx] > 1e-8))
        else:
            num_effective_references = int(np.sum(fit_labels[train_idx] == 1))
        if num_effective_references < n_components_for_cv:
            logging.warning(
                f"Not enough effective references to perform cross-validation "
                f"(target_class={target_class}, num_references={len(reference_scores[train_idx])}, "
                f"num_effective_references={num_effective_references}, n_components={n_components_for_cv})"
            )
            return rep_original.shape[1]

        sw_train = None if class_w is None else np.asarray(class_w, dtype=float)[train_idx]
        scorer_original.fit(rep_original[train_idx], fit_labels[train_idx], sample_weight=sw_train)
        predicted_scores_original = convert_to_normalized_rank_score(scorer_original.score(rep_original[test_idx]))
        correlation_original = calculate_score_correlations_safe(predicted_scores_original, corr_targets[test_idx])
        score_correlations['original'].append(correlation_original)

        if rep_LD is not None:
            scorer_LD.fit(rep_LD[train_idx], fit_labels[train_idx], sample_weight=sw_train)
            predicted_scores_LD = convert_to_normalized_rank_score(scorer_LD.score(rep_LD[test_idx]))
            correlation_LD = calculate_score_correlations_safe(predicted_scores_LD, corr_targets[test_idx])
            score_correlations['LD'].append(correlation_LD)

        if rep_HD is not None: 
            scorer_HD.fit(rep_HD[train_idx], fit_labels[train_idx], sample_weight=sw_train)
            predicted_scores_HD = convert_to_normalized_rank_score(scorer_HD.score(rep_HD[test_idx]))
            correlation_HD = calculate_score_correlations_safe(predicted_scores_HD, corr_targets[test_idx])
            score_correlations['HD'].append(correlation_HD)
        
    best_CV = _pick_cv_best_key(score_correlations, d_schedule=d_schedule)
    if best_CV == 'HD':
        return rep_HD.shape[1]
    if best_CV == 'LD':
        return rep_LD.shape[1]
    return rep_original.shape[1]


def _cv_step_ok(candidate_folds, original_folds, min_fold_floor=CV_MIN_FOLD_FLOOR):
    """True if candidate mean is higher and min fold is stable vs original."""
    if len(candidate_folds) == 0 or len(original_folds) == 0:
        return False
    return (
        np.mean(candidate_folds) > np.mean(original_folds)
        and np.min(candidate_folds) > max(np.min(original_folds), min_fold_floor)
    )


def _pick_cv_best_key(score_correlations, d_schedule=False):
    """PCA-dim CV: HD if higher mean; LD if higher mean and stable min fold.

    --d_schedule: never pick LD; HD uses the same min-fold / 0.1 gate as LD.
    """
    max_mean_correlation = np.mean(score_correlations['original'])
    max_min_correlation = np.min(score_correlations['original'])
    best_CV = 'original'
    if len(score_correlations['HD']) > 0:
        mean_correlation_HD = np.mean(score_correlations['HD'])
        if d_schedule:
            if _cv_step_ok(score_correlations['HD'], score_correlations['original']):
                best_CV = 'HD'
        elif mean_correlation_HD > max_mean_correlation:
            max_mean_correlation = mean_correlation_HD
            best_CV = 'HD'
    if (not d_schedule) and len(score_correlations['LD']) > 0:
        mean_correlation_LD = np.mean(score_correlations['LD'])
        min_correlation_LD = np.min(score_correlations['LD'])
        if (
            mean_correlation_LD > max_mean_correlation
            and min_correlation_LD > max(max_min_correlation, CV_MIN_FOLD_FLOOR)
        ):
            best_CV = 'LD'
    return best_CV


def _pick_reg_decay_key(score_correlations):
    """--reg_decay: step to λ/10 only if mean is higher and min fold is stable.

    Same gate as PCA-dim LD / --d_schedule HD:
    mean(λ/10) > mean(λ) and min(λ/10) > max(min(λ), CV_MIN_FOLD_FLOOR).
    HD is unused: decay never increases λ.
    """
    if _cv_step_ok(score_correlations['LD'], score_correlations['original']):
        return 'LD'
    return 'original'


def _reg_covar_decade_neighbors(
    current,
    factor=REG_COVAR_CV_FACTOR,
    lo=MIN_REG_COVAR_CV,
    hi=MAX_REG_COVAR_CV,
):
    """Return (lower, higher) decade neighbors, or None if clamped onto ``current``."""
    current = float(current)
    lower = max(current / float(factor), float(lo))
    higher = min(current * float(factor), float(hi))

    def _distinct(a, b):
        return abs(np.log10(max(a, 1e-300)) - np.log10(max(b, 1e-300))) > 0.05

    return (lower if _distinct(lower, current) else None,
            higher if _distinct(higher, current) else None)


def get_next_reg_covar_decay(
    current_reg_covar,
    X_ref,
    reference_scores,
    scorer_config,
    scoring_model='GMM',
    target_class=1,
    dual_gmm=False,
    scorer_config_fail=None,
    sample_weight=None,
    success_rate_weight=False,
):
    """Monotone decade decay: λ vs λ/10 with mean + min-fold gate; never increase."""
    current = float(current_reg_covar)
    lower, _ = _reg_covar_decade_neighbors(current)
    if lower is None:
        return current
    if dual_gmm:
        return select_reg_covar_by_dual_bayes_CV(
            reference_scores,
            np.asarray(X_ref),
            current,
            lower,
            scorer_config,
            scorer_config_fail,
            scoring_model=scoring_model,
            sample_weight=sample_weight,
            success_rate_weight=success_rate_weight,
        )
    return select_reg_covar_by_CV(
        reference_scores,
        np.asarray(X_ref),
        current,
        lower,
        None,
        scorer_config,
        scoring_model=scoring_model,
        target_class=target_class,
        pick_key=_pick_reg_decay_key,
        sample_weight=sample_weight,
        success_rate_weight=success_rate_weight,
    )


def _try_dual_bayes_scores(
    X_train, y_bin, X_test, cfg_pass, cfg_fail, scoring_model,
    sample_weight=None, success_rate_weight=False, y_score=None,
):
    """Fit pass/fail GMMs; return Bayes p(y=pass|x) on X_test, or None if EM collapses."""
    scorer_pass = construct_cv_scorer(cfg_pass, scoring_model=scoring_model)
    scorer_fail = construct_cv_scorer(cfg_fail, scoring_model=scoring_model)
    try:
        if success_rate_weight:
            scores = np.clip(np.asarray(y_score, dtype=float).ravel(), 0.0, 1.0)
            ones = np.ones(scores.shape[0], dtype=int)
            scorer_pass.fit(X_train, ones, sample_weight=scores)
            scorer_fail.fit(X_train, ones, sample_weight=1.0 - scores)
        else:
            scorer_pass.fit(X_train, y_bin, sample_weight=sample_weight)
            scorer_fail.fit(X_train, 1 - y_bin, sample_weight=sample_weight)
        return bayes_pass_posterior(
            scorer_pass.score(X_test),
            scorer_fail.score(X_test),
        )
    except ValueError as exc:
        logging.warning(f"dual Bayes GMM fit failed during reg_covar CV ({exc})")
        return None


def select_reg_covar_by_dual_bayes_CV(
    reference_scores,
    X_ref,
    reg_original,
    reg_LD,
    scorer_config,
    scorer_config_fail,
    scoring_model='GMM',
    sample_weight=None,
    success_rate_weight=False,
):
    """Pick λ by 3-fold Pearson of dual Bayes p(y=pass|x) vs labels (eval functional)."""
    if scorer_config_fail is None:
        raise ValueError("select_reg_covar_by_dual_bayes_CV requires scorer_config_fail")
    kf = KFold(n_splits=3, shuffle=True, random_state=scorer_config.random_state)
    score_correlations = {'LD': [], 'original': [], 'HD': []}
    y = np.asarray(reference_scores, dtype=float)
    y_bin = (y > PASS_THRESH).astype(int)
    X_ref = np.asarray(X_ref)
    k_pass = _cv_n_components(scorer_config)
    k_fail = _cv_n_components(scorer_config_fail)
    cfg_orig_p = replace(scorer_config, reg_covar=float(reg_original))
    cfg_orig_f = replace(scorer_config_fail, reg_covar=float(reg_original))
    cfg_ld_p = replace(scorer_config, reg_covar=float(reg_LD))
    cfg_ld_f = replace(scorer_config_fail, reg_covar=float(reg_LD))

    for train_idx, test_idx in kf.split(X_ref):
        if success_rate_weight:
            n_pass = int(np.sum(y[train_idx] > 1e-8))
            n_fail = int(np.sum((1.0 - y[train_idx]) > 1e-8))
        else:
            n_pass = int(np.sum(y_bin[train_idx] == 1))
            n_fail = int(np.sum(y_bin[train_idx] == 0))
        if n_pass < k_pass or n_fail < k_fail:
            logging.warning(
                f"Not enough class refs for dual Bayes reg_covar CV "
                f"(n_pass={n_pass}, n_fail={n_fail}, k_pass={k_pass}, k_fail={k_fail}); "
                f"keeping {reg_original}"
            )
            return float(reg_original)

        sw_train = None if sample_weight is None else np.asarray(sample_weight, dtype=float)[train_idx]
        p_orig = _try_dual_bayes_scores(
            X_ref[train_idx], y_bin[train_idx], X_ref[test_idx],
            cfg_orig_p, cfg_orig_f, scoring_model,
            sample_weight=sw_train,
            success_rate_weight=success_rate_weight,
            y_score=y[train_idx],
        )
        if p_orig is None:
            logging.warning(
                f"current λ={reg_original:g} collapsed in dual Bayes CV; keeping it"
            )
            return float(reg_original)
        score_correlations['original'].append(
            calculate_score_correlations_safe(p_orig, y[test_idx])
        )

        p_ld = _try_dual_bayes_scores(
            X_ref[train_idx], y_bin[train_idx], X_ref[test_idx],
            cfg_ld_p, cfg_ld_f, scoring_model,
            sample_weight=sw_train,
            success_rate_weight=success_rate_weight,
            y_score=y[train_idx],
        )
        if p_ld is None:
            # Collapse at λ/10: do not decay.
            score_correlations['LD'].append(-1.0)
        else:
            score_correlations['LD'].append(
                calculate_score_correlations_safe(p_ld, y[test_idx])
            )

    best = _pick_reg_decay_key(score_correlations)
    mean_orig = float(np.mean(score_correlations['original']))
    mean_ld = float(np.mean(score_correlations['LD']))
    min_orig = float(np.min(score_correlations['original']))
    min_ld = float(np.min(score_correlations['LD']))
    logging.info(
        f"[reg_covar CV dual-Bayes] λ={reg_original:g} mean={mean_orig:.4f} "
        f"min={min_orig:.4f} λ/10={reg_LD:g} mean={mean_ld:.4f} "
        f"min={min_ld:.4f} -> {best}"
    )
    if best == 'LD':
        return float(reg_LD)
    return float(reg_original)


def select_reg_covar_by_CV(
    reference_scores,
    X_ref,
    reg_original,
    reg_LD,
    reg_HD,
    scorer_config,
    scoring_model='GMM',
    target_class=1,
    pick_key=_pick_cv_best_key,
    sample_weight=None,
    success_rate_weight=False,
):
    """Pick reg_covar by 3-fold Pearson of GMM rank-scores vs labels (sklearn probe)."""
    kf = KFold(n_splits=3, shuffle=True, random_state=scorer_config.random_state)
    score_correlations = {'LD': [], 'original': [], 'HD': []}
    train_labels_binary = (np.asarray(reference_scores) > PASS_THRESH).astype(int)
    fit_labels, class_w = _class_fit_labels_and_weight(
        train_labels_binary,
        reference_scores,
        target_class,
        sample_weight=sample_weight,
        success_rate_weight=success_rate_weight,
    )
    if target_class == 1:
        corr_targets = np.asarray(reference_scores, dtype=float)
    else:
        corr_targets = 1.0 - np.asarray(reference_scores, dtype=float)
    n_components_for_cv = _cv_n_components(scorer_config)
    X_ref = np.asarray(X_ref)

    scorer_original = construct_cv_scorer(
        replace(scorer_config, reg_covar=float(reg_original)), scoring_model=scoring_model
    )
    scorer_LD = (
        construct_cv_scorer(replace(scorer_config, reg_covar=float(reg_LD)), scoring_model=scoring_model)
        if reg_LD is not None else None
    )
    scorer_HD = (
        construct_cv_scorer(replace(scorer_config, reg_covar=float(reg_HD)), scoring_model=scoring_model)
        if reg_HD is not None else None
    )

    for train_idx, test_idx in kf.split(X_ref):
        if success_rate_weight and class_w is not None:
            num_effective_references = int(np.sum(np.asarray(class_w, dtype=float)[train_idx] > 1e-8))
        else:
            num_effective_references = int(np.sum(fit_labels[train_idx] == 1))
        if num_effective_references < n_components_for_cv:
            logging.warning(
                f"Not enough effective references for reg_covar CV "
                f"(target_class={target_class}, n_eff={num_effective_references}, "
                f"n_components={n_components_for_cv}); keeping {reg_original}"
            )
            return float(reg_original)

        sw_train = None if class_w is None else np.asarray(class_w, dtype=float)[train_idx]
        scorer_original.fit(X_ref[train_idx], fit_labels[train_idx], sample_weight=sw_train)
        predicted = convert_to_normalized_rank_score(scorer_original.score(X_ref[test_idx]))
        score_correlations['original'].append(
            calculate_score_correlations_safe(predicted, corr_targets[test_idx])
        )
        if scorer_LD is not None:
            scorer_LD.fit(X_ref[train_idx], fit_labels[train_idx], sample_weight=sw_train)
            predicted_ld = convert_to_normalized_rank_score(scorer_LD.score(X_ref[test_idx]))
            score_correlations['LD'].append(
                calculate_score_correlations_safe(predicted_ld, corr_targets[test_idx])
            )
        if scorer_HD is not None:
            scorer_HD.fit(X_ref[train_idx], fit_labels[train_idx], sample_weight=sw_train)
            predicted_hd = convert_to_normalized_rank_score(scorer_HD.score(X_ref[test_idx]))
            score_correlations['HD'].append(
                calculate_score_correlations_safe(predicted_hd, corr_targets[test_idx])
            )

    best = pick_key(score_correlations)
    mean_orig = float(np.mean(score_correlations['original'])) if score_correlations['original'] else float('nan')
    mean_ld = float(np.mean(score_correlations['LD'])) if score_correlations['LD'] else float('nan')
    min_orig = float(np.min(score_correlations['original'])) if score_correlations['original'] else float('nan')
    min_ld = float(np.min(score_correlations['LD'])) if score_correlations['LD'] else float('nan')
    logging.info(
        f"[reg_covar CV] λ={reg_original:g} mean={mean_orig:.4f} min={min_orig:.4f} "
        f"λ/10={reg_LD} mean={mean_ld:.4f} min={min_ld:.4f} -> {best}"
    )
    if best == 'HD':
        return float(reg_HD)
    if best == 'LD':
        return float(reg_LD)
    return float(reg_original)


def calculate_score_correlations_safe(predicted_scores, true_scores):
    predicted_scores = np.asarray(predicted_scores, dtype=float)
    true_scores = np.asarray(true_scores, dtype=float)

    if np.allclose(np.var(true_scores), 0.0):
        const_val = true_scores[0]
        mae = np.mean(np.abs(predicted_scores - const_val))  # predicted_scores range: [0, 1]
        return 1 - mae
    if np.allclose(np.var(predicted_scores), 0.0):
        return 0.0
    corr = pearsonr(predicted_scores, true_scores)[0]
    if corr is None or not np.isfinite(corr):
        return 0.0
    return float(corr)


def _cv_n_components(scorer_config):
    return scorer_config.n_components


def construct_cv_scorer(scorer_config, scoring_model='GMM'):
    """Cheap scorer used only for PCA dimension CV.

    MCMCGMM uses an sklearn diag-cov GMM surrogate so CV does not
    launch MCMC. ``n_components`` for MCMCGMM is chosen from MCMC posterior
    weights, not this probe.
    """
    probe_n_init = _cheap_probe_n_init(scorer_config)
    if scoring_model in BAYES_GMM_MODELS:
        cov = 'diag'
        return GMMScorer(
            GMMScorerConfig(
                n_components=_cv_n_components(scorer_config),
                random_state=scorer_config.random_state,
                covariance_type=cov,
                reg_covar=getattr(scorer_config, 'reg_covar', 1e-6),
                n_init=probe_n_init,
                max_iter=getattr(scorer_config, 'sklearn_max_iter', 500),
                tol=getattr(scorer_config, 'sklearn_tol', 1e-3),
            )
        )
    return GMMScorer(replace(scorer_config, n_init=probe_n_init))


def construct_scorer(scorer_config, scoring_model='GMM'):
    if scoring_model in BAYES_GMM_MODELS:
        if not isinstance(scorer_config, MCMCGMMScorerConfig):
            raise TypeError(f"{scoring_model} requires MCMCGMMScorerConfig")
        return MCMCGMMScorer(scorer_config)
    if isinstance(scorer_config, GMMScorerConfig):
        return GMMScorer(scorer_config)
    raise TypeError(f"Unsupported scorer config type for scoring_model={scoring_model!r}: {type(scorer_config)}")


def ensure_nonzero_entropy(reference_data, reference_labels, all_data, scorer_config, entropy_var_lb=1e-3, sample_weight=None):
    while scorer_config.reg_covar < 1.0 - 1e-10:
        scorer = GMMScorer(
            replace(scorer_config, n_init=_cheap_probe_n_init(scorer_config))
        )
        scorer.fit(reference_data, reference_labels, sample_weight=sample_weight)
        entropies = scorer.get_uncertainties(all_data)

        var_entropy = np.var(entropies)
        if var_entropy > entropy_var_lb:
            return True

        scorer_config.reg_covar *= 10
    
    scorer_config.reg_covar = 1e-1
    return False


