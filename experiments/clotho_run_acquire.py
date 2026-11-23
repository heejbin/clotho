"""Reference-set acquisition strategies for Clotho iteration runs."""
from __future__ import annotations

import logging
import time

import numpy as np

from clotho.al.sampling import (
    extend_reference_set_bald,
    extend_reference_set_batchbald,
    extend_reference_set_diversity,
    extend_reference_set_epig,
    extend_reference_set_likelihood_variance,
    extend_reference_set_random,
    extend_reference_set_rankvar,
    extend_reference_set_uncertainty,
)
from clotho_run_helpers import (
    FAST_WARMUP_ADD,
    RANDOM_START_N_REFS,
    _ensure_feature_rep,
)


def acquire_next_reference_indices(
    *,
    refset_extension_method,
    current_reference_set_indices,
    LIH_rep,
    current_n_features,
    current_n_features_fail,
    scorer,
    scorer_fail,
    dual_gmm,
    use_mcmc_gmm,
    prior_pass,
    prior_fail,
    num_add,
    iter_num_add,
    n_trials,
    n_target,
    gmm_collapse_fallback,
    use_random_startpoint,
    use_random_100_start,
    use_random_n0_start,
    n0_gate_sum,
    n0_per_class,
    n_pass_now,
    n_fail_now,
    n0_flag_name,
    iteration_i,
    skip,
    fast,
    min_class_refs,
    num_effective_references,
    num_fail_references,
    bald_thin_draws,
    LIH,
    labels,
    dim_reduction_method,
    random_seed,
    _ipw_sample_weight,
    run_prior_pass,
    run_prior_fail,
    fit_gmm_class,
):
    """Select the next batch of reference indices.

    Returns a dict with keys:
      new_reference_set_indices, new_unc_indices, new_div_indices,
      uncertainties, aleatoric_uncertainties, posterior_disagreement,
      bald_log_marginal, duration_reference_set_extension
    """
    start_time_reference_set_extension = time.time()
    uncertainties = None
    aleatoric_uncertainties = None  # E_t[H[p_t]] when BALD is used
    posterior_disagreement = None  # Var_t[p_t] when BALD is used
    bald_log_marginal = None  # E_t[log p(x)] dual-class mixture (OOD diag)
    new_unc_indices = []
    new_div_indices = []
    # --fast: while fitting is skipped for lack of class refs, add 20 at a time.
    iter_num_add = FAST_WARMUP_ADD if (fast and skip and not gmm_collapse_fallback) else num_add
    if fast and skip and not gmm_collapse_fallback and iter_num_add != num_add:
        logging.info(
            f"[Iteration {iteration_i+1}] --fast warmup: adding {iter_num_add} refs "
            f"(need {min_class_refs}/class; "
            f"pass={num_effective_references}, fail={num_fail_references})"
        )
    if use_random_startpoint:
        if use_random_100_start:
            remain = RANDOM_START_N_REFS - len(current_reference_set_indices)
            if remain < iter_num_add:
                iter_num_add = int(remain)
    effective_sampling_method = (
        'random_start' if use_random_startpoint else refset_extension_method
    )

    if gmm_collapse_fallback:
        logging.warning(
            f"[Iteration {iteration_i+1}] GMM collapse fallback: random +{iter_num_add} refs "
            f"(n_ref={len(current_reference_set_indices)}, "
            f"n_feat={current_n_features})"
        )
        new_reference_set_indices = extend_reference_set_random(
            current_reference_set_indices,
            LIH_rep[current_n_features],
            num_add=iter_num_add,
        )
    elif use_random_startpoint:
        if use_random_n0_start:
            if n0_gate_sum:
                switch_hint = (
                    f"n_pass+n_fail={n_pass_now + n_fail_now}/{n0_per_class}; "
                    f"switch when sum > n0"
                )
            else:
                switch_hint = (
                    f"n_pass={n_pass_now}/{n0_per_class}, "
                    f"n_fail={n_fail_now}/{n0_per_class}; "
                    f"switch when both > n0"
                )
            logging.info(
                f"[Iteration {iteration_i+1}] --{n0_flag_name}: random +{iter_num_add} refs "
                f"(n_ref={len(current_reference_set_indices)} → "
                f"{len(current_reference_set_indices) + iter_num_add}, "
                f"{switch_hint})"
            )
        else:
            logging.info(
                f"[Iteration {iteration_i+1}] --random_100_start: random +{iter_num_add} refs "
                f"(n_ref={len(current_reference_set_indices)} → "
                f"{len(current_reference_set_indices) + iter_num_add}, "
                f"switch at {RANDOM_START_N_REFS})"
            )
        new_reference_set_indices = extend_reference_set_random(
            current_reference_set_indices,
            LIH_rep[current_n_features],
            num_add=iter_num_add,
        )
    elif refset_extension_method == 'random':
        new_reference_set_indices = extend_reference_set_random(current_reference_set_indices, LIH_rep[current_n_features], num_add=iter_num_add)

    elif refset_extension_method in ('diversity_euclidean', 'diversity'):
        new_reference_set_indices = extend_reference_set_diversity(current_reference_set_indices, LIH_rep[current_n_features], num_add=iter_num_add, metric='euclidean')
        new_div_indices = list(new_reference_set_indices)

    elif refset_extension_method == 'uncertainty':
        if scorer is None:
            logging.warning(f'Scorer is not fitted due to the insufficient effective reference points. Using random sampling instead.')
            new_reference_set_indices = extend_reference_set_random(current_reference_set_indices, LIH_rep[current_n_features], num_add=iter_num_add)
        else:
            if use_mcmc_gmm and not getattr(scorer, 'has_mcmc_posterior', lambda: False)():
                logging.info(
                    f"[Iteration {iteration_i+1}] uncertainty sampling needs MCMC posterior; running MCMC now"
                )
                _ensure_feature_rep(LIH, LIH_rep, current_n_features, dim_reduction_method)
                X_ref = np.asarray(LIH_rep[current_n_features][current_reference_set_indices])
                fit_gmm_class(
                    scorer,
                    X_ref,
                    1,
                    run_mcmc=True,
                )
            new_reference_set_indices, uncertainties = extend_reference_set_uncertainty(scorer, current_reference_set_indices, LIH_rep[current_n_features], num_add=iter_num_add)
            new_unc_indices = list(new_reference_set_indices)

    elif refset_extension_method in ('balanced', 'llvar'):
        # balanced MCMCGMM: half BALD + half diversity.
        # llvar (MCMCGMM): half Var_t[p(x|θ)] + half diversity
        #   (dual: mixture likelihood; single: pass GMM).
        # balanced / llvar GMM: half predictive uncertainty + half diversity.
        use_bald_half = use_mcmc_gmm and refset_extension_method == 'balanced'
        if scorer is None or (use_mcmc_gmm and dual_gmm and scorer_fail is None):
            logging.warning(
                f'Scorer is not fitted due to the insufficient effective reference points. '
                f'Using diversity sampling instead.'
            )
            new_reference_set_indices = extend_reference_set_diversity(
                current_reference_set_indices, LIH_rep[current_n_features],
                num_add=iter_num_add, metric='euclidean',
            )
            new_div_indices = list(new_reference_set_indices)
        else:
            n_unc = iter_num_add // 2
            n_div = iter_num_add - n_unc
            if use_mcmc_gmm and not getattr(scorer, 'has_mcmc_posterior', lambda: False)():
                logging.info(
                    f"[Iteration {iteration_i+1}] {refset_extension_method} sampling needs MCMC posterior"
                    f"{' [pass]' if dual_gmm else ''}; running MCMC now"
                )
                _ensure_feature_rep(LIH, LIH_rep, current_n_features, dim_reduction_method)
                X_ref = np.asarray(LIH_rep[current_n_features][current_reference_set_indices])
                fit_gmm_class(
                    scorer,
                    X_ref,
                    1,
                    run_mcmc=True,
                )
            if (
                use_mcmc_gmm
                and dual_gmm
                and not getattr(scorer_fail, 'has_mcmc_posterior', lambda: False)()
            ):
                logging.info(
                    f"[Iteration {iteration_i+1}] {refset_extension_method} sampling needs MCMC posterior [fail]; "
                    f"running MCMC now"
                )
                _ensure_feature_rep(
                    LIH, LIH_rep, current_n_features_fail, dim_reduction_method,
                                        )
                X_ref_fail = np.asarray(
                    LIH_rep[current_n_features_fail][current_reference_set_indices]
                )
                fit_gmm_class(
                    scorer_fail,
                    X_ref_fail,
                    0,
                    run_mcmc=True,
                )
            if use_bald_half:
                if not dual_gmm:
                    raise ValueError(
                        "MCMCGMM balanced BALD requires dual_gmm "
                        "(pass/fail class-conditional models)."
                    )
                (
                    new_reference_set_indices_by_uncertainty,
                    uncertainties,
                    aleatoric_uncertainties,
                    bald_log_marginal,
                    posterior_disagreement,
                ) = extend_reference_set_bald(
                    scoring_model=scorer,
                    refset_indices=current_reference_set_indices,
                    target_vectors=LIH_rep[current_n_features],
                    num_add=n_unc,
                    scoring_model_fail=scorer_fail,
                    target_vectors_fail=LIH_rep[current_n_features_fail],
                    prior_pass=run_prior_pass,
                    prior_fail=run_prior_fail,
                )
            elif use_mcmc_gmm:
                llvar_kwargs = dict(
                    scoring_model=scorer,
                    refset_indices=current_reference_set_indices,
                    target_vectors=LIH_rep[current_n_features],
                    num_add=n_unc,
                )
                if dual_gmm:
                    llvar_kwargs.update(
                        scoring_model_fail=scorer_fail,
                        target_vectors_fail=LIH_rep[current_n_features_fail],
                        prior_pass=run_prior_pass,
                        prior_fail=run_prior_fail,
                    )
                (
                    new_reference_set_indices_by_uncertainty,
                    uncertainties,
                ) = extend_reference_set_likelihood_variance(**llvar_kwargs)
            else:
                new_reference_set_indices_by_uncertainty, uncertainties = (
                    extend_reference_set_uncertainty(
                        scorer,
                        current_reference_set_indices,
                        LIH_rep[current_n_features],
                        num_add=n_unc,
                    )
                )
            new_reference_set_indices_by_diversity = extend_reference_set_diversity(
                current_reference_set_indices + new_reference_set_indices_by_uncertainty,
                LIH_rep[current_n_features],
                num_add=n_div,
                metric='euclidean',
            )
            assert set(new_reference_set_indices_by_uncertainty).isdisjoint(
                set(new_reference_set_indices_by_diversity)
            ), "The two sets of new reference indices should be disjoint."
            new_reference_set_indices = (
                new_reference_set_indices_by_uncertainty
                + new_reference_set_indices_by_diversity
            )
            new_unc_indices = list(new_reference_set_indices_by_uncertainty)
            new_div_indices = list(new_reference_set_indices_by_diversity)

    elif refset_extension_method == 'bald':
        # Dual-GMM binary BALD: H[p̄] − E_t[H[p_t]] from Bayes pass posteriors.
        if not dual_gmm:
            raise ValueError(
                "bald sampling requires dual_gmm "
                "(pass/fail class-conditional models)."
            )
        if scorer is None or scorer_fail is None:
            logging.warning(
                f'Scorer is not fitted due to the insufficient effective reference points. '
                f'Using random sampling instead.'
            )
            new_reference_set_indices = extend_reference_set_random(
                current_reference_set_indices, LIH_rep[current_n_features], num_add=iter_num_add
            )
        else:
            if use_mcmc_gmm and not getattr(scorer, 'has_mcmc_posterior', lambda: False)():
                logging.info(
                    f"[Iteration {iteration_i+1}] bald sampling needs MCMC posterior [pass]; "
                    f"running MCMC now"
                )
                _ensure_feature_rep(LIH, LIH_rep, current_n_features, dim_reduction_method)
                X_ref = np.asarray(LIH_rep[current_n_features][current_reference_set_indices])
                fit_gmm_class(
                    scorer,
                    X_ref,
                    1,
                    run_mcmc=True,
                )
            if use_mcmc_gmm and not getattr(scorer_fail, 'has_mcmc_posterior', lambda: False)():
                logging.info(
                    f"[Iteration {iteration_i+1}] bald sampling needs MCMC posterior [fail]; "
                    f"running MCMC now"
                )
                _ensure_feature_rep(
                    LIH, LIH_rep, current_n_features_fail, dim_reduction_method,
                                        )
                X_ref_fail = np.asarray(
                    LIH_rep[current_n_features_fail][current_reference_set_indices]
                )
                fit_gmm_class(
                    scorer_fail,
                    X_ref_fail,
                    0,
                    run_mcmc=True,
                )
            prior_pass, prior_fail = run_prior_pass, run_prior_fail
            new_reference_set_indices, uncertainties, aleatoric_uncertainties, bald_log_marginal, posterior_disagreement = (
                extend_reference_set_bald(
                    scoring_model=scorer,
                    refset_indices=current_reference_set_indices,
                    target_vectors=LIH_rep[current_n_features],
                    num_add=iter_num_add,
                    scoring_model_fail=scorer_fail,
                    target_vectors_fail=LIH_rep[current_n_features_fail],
                    prior_pass=prior_pass,
                    prior_fail=prior_fail,
                )
            )
            new_unc_indices = list(new_reference_set_indices)

    elif refset_extension_method == 'batchbald':
        # Dual-GMM BatchBALD: greedy joint MI on MCMC predictive probs (S, N, 2).
        if not dual_gmm:
            raise ValueError(
                "batchbald sampling requires dual_gmm "
                "(pass/fail class-conditional models)."
            )
        if scorer is None or scorer_fail is None:
            logging.warning(
                f'Scorer is not fitted due to the insufficient effective reference points. '
                f'Using random sampling instead.'
            )
            new_reference_set_indices = extend_reference_set_random(
                current_reference_set_indices, LIH_rep[current_n_features], num_add=iter_num_add
            )
        else:
            if use_mcmc_gmm and not getattr(scorer, 'has_mcmc_posterior', lambda: False)():
                logging.info(
                    f"[Iteration {iteration_i+1}] batchbald sampling needs MCMC posterior [pass]; "
                    f"running MCMC now"
                )
                _ensure_feature_rep(LIH, LIH_rep, current_n_features, dim_reduction_method)
                X_ref = np.asarray(LIH_rep[current_n_features][current_reference_set_indices])
                fit_gmm_class(
                    scorer,
                    X_ref,
                    1,
                    run_mcmc=True,
                )
            if use_mcmc_gmm and not getattr(scorer_fail, 'has_mcmc_posterior', lambda: False)():
                logging.info(
                    f"[Iteration {iteration_i+1}] batchbald sampling needs MCMC posterior [fail]; "
                    f"running MCMC now"
                )
                _ensure_feature_rep(
                    LIH, LIH_rep, current_n_features_fail, dim_reduction_method,
                                        )
                X_ref_fail = np.asarray(
                    LIH_rep[current_n_features_fail][current_reference_set_indices]
                )
                fit_gmm_class(
                    scorer_fail,
                    X_ref_fail,
                    0,
                    run_mcmc=True,
                )
            prior_pass, prior_fail = run_prior_pass, run_prior_fail
            new_reference_set_indices, uncertainties, aleatoric_uncertainties, bald_log_marginal, posterior_disagreement = (
                extend_reference_set_batchbald(
                    scoring_model=scorer,
                    refset_indices=current_reference_set_indices,
                    target_vectors=LIH_rep[current_n_features],
                    num_add=iter_num_add,
                    scoring_model_fail=scorer_fail,
                    target_vectors_fail=LIH_rep[current_n_features_fail],
                    prior_pass=prior_pass,
                    prior_fail=prior_fail,
                )
            )
            new_unc_indices = list(new_reference_set_indices)

    elif refset_extension_method == 'rankvar':
        # Var_t of the average rank of η snapped to y's grid, on the
        # target pool that Spearman/Pearson actually rank.
        if not dual_gmm:
            raise ValueError(
                "rankvar sampling requires dual_gmm "
                "(pass/fail class-conditional models)."
            )
        if scorer is None or scorer_fail is None:
            logging.warning(
                f'Scorer is not fitted due to the insufficient effective reference points. '
                f'Using random sampling instead.'
            )
            new_reference_set_indices = extend_reference_set_random(
                current_reference_set_indices, LIH_rep[current_n_features], num_add=iter_num_add
            )
        else:
            if use_mcmc_gmm and not getattr(scorer, 'has_mcmc_posterior', lambda: False)():
                logging.info(
                    f"[Iteration {iteration_i+1}] rankvar sampling needs MCMC posterior [pass]; "
                    f"running MCMC now"
                )
                _ensure_feature_rep(LIH, LIH_rep, current_n_features, dim_reduction_method)
                X_ref = np.asarray(LIH_rep[current_n_features][current_reference_set_indices])
                fit_gmm_class(
                    scorer,
                    X_ref,
                    1,
                    run_mcmc=True,
                )
            if use_mcmc_gmm and not getattr(scorer_fail, 'has_mcmc_posterior', lambda: False)():
                logging.info(
                    f"[Iteration {iteration_i+1}] rankvar sampling needs MCMC posterior [fail]; "
                    f"running MCMC now"
                )
                _ensure_feature_rep(
                    LIH, LIH_rep, current_n_features_fail, dim_reduction_method,
                                        )
                X_ref_fail = np.asarray(
                    LIH_rep[current_n_features_fail][current_reference_set_indices]
                )
                fit_gmm_class(
                    scorer_fail,
                    X_ref_fail,
                    0,
                    run_mcmc=True,
                )
            prior_pass, prior_fail = run_prior_pass, run_prior_fail
            logging.info(
                f"[Iteration {iteration_i+1}] rankvar ranks raw η "
                f"on the target pool (n={n_target}); average ties"
            )
            new_reference_set_indices, uncertainties, bald_log_marginal, posterior_disagreement = (
                extend_reference_set_rankvar(
                    scoring_model=scorer,
                    refset_indices=current_reference_set_indices,
                    target_vectors=LIH_rep[current_n_features],
                    num_add=iter_num_add,
                    scoring_model_fail=scorer_fail,
                    target_vectors_fail=LIH_rep[current_n_features_fail],
                    prior_pass=prior_pass,
                    prior_fail=prior_fail,
                    n_trials=n_trials,
                    rank_indices=np.arange(n_target),
                )
            )
            new_unc_indices = list(new_reference_set_indices)

    elif refset_extension_method == 'epig':
        # Ranking EPIG: Spearman order of η snapped to the y grid
        # {0, 1/n, ..., 1}. n_trials is 10, 20, or 50 (label resolution).
        if not dual_gmm:
            raise ValueError(
                "epig sampling requires dual_gmm "
                "(pass/fail class-conditional models)."
            )
        if scorer is None or scorer_fail is None:
            logging.warning(
                f'Scorer is not fitted due to the insufficient effective reference points. '
                f'Using random sampling instead.'
            )
            new_reference_set_indices = extend_reference_set_random(
                current_reference_set_indices, LIH_rep[current_n_features], num_add=iter_num_add
            )
        else:
            if use_mcmc_gmm and not getattr(scorer, 'has_mcmc_posterior', lambda: False)():
                logging.info(
                    f"[Iteration {iteration_i+1}] epig sampling needs MCMC posterior [pass]; "
                    f"running MCMC now"
                )
                _ensure_feature_rep(LIH, LIH_rep, current_n_features, dim_reduction_method)
                X_ref = np.asarray(LIH_rep[current_n_features][current_reference_set_indices])
                fit_gmm_class(
                    scorer,
                    X_ref,
                    1,
                    run_mcmc=True,
                )
            if use_mcmc_gmm and not getattr(scorer_fail, 'has_mcmc_posterior', lambda: False)():
                logging.info(
                    f"[Iteration {iteration_i+1}] epig sampling needs MCMC posterior [fail]; "
                    f"running MCMC now"
                )
                _ensure_feature_rep(
                    LIH, LIH_rep, current_n_features_fail, dim_reduction_method,
                                        )
                X_ref_fail = np.asarray(
                    LIH_rep[current_n_features_fail][current_reference_set_indices]
                )
                fit_gmm_class(
                    scorer_fail,
                    X_ref_fail,
                    0,
                    run_mcmc=True,
                )
            prior_pass, prior_fail = run_prior_pass, run_prior_fail
            new_reference_set_indices, uncertainties = extend_reference_set_epig(
                scoring_model=scorer,
                refset_indices=current_reference_set_indices,
                target_vectors=LIH_rep[current_n_features],
                num_add=iter_num_add,
                scoring_model_fail=scorer_fail,
                target_vectors_fail=LIH_rep[current_n_features_fail],
                prior_pass=prior_pass,
                prior_fail=prior_fail,
                labels=labels,
                sample_weight=_ipw_sample_weight(),
                rank_indices=np.arange(n_target),
                n_trials=int(n_trials),
                random_state=int(random_seed) * 100003 + int(iteration_i + 1),
            )
            new_unc_indices = list(new_reference_set_indices)

    elif refset_extension_method in (
        'epigbalanced', 'rankvarbalanced', 'epigspearmanbalanced',
    ):
        # Half the batch from epig, epigspearman, or rankvar, half from
        # Euclidean diversity. Score points are chosen first; diversity
        # then avoids that half. With num_add=10 this is 5 + 5.
        if refset_extension_method == 'epigbalanced':
            score_name = 'epig'
        elif refset_extension_method == 'epigspearmanbalanced':
            score_name = 'epigspearman'
        else:
            score_name = 'rankvar'
        if not dual_gmm:
            raise ValueError(
                f"{refset_extension_method} sampling requires dual_gmm "
                "(pass/fail class-conditional models)."
            )
        if scorer is None or scorer_fail is None:
            logging.warning(
                f'Scorer is not fitted due to the insufficient effective reference points. '
                f'Using diversity sampling instead.'
            )
            new_reference_set_indices = extend_reference_set_diversity(
                current_reference_set_indices, LIH_rep[current_n_features],
                num_add=iter_num_add, metric='euclidean',
            )
            new_div_indices = list(new_reference_set_indices)
        else:
            n_score = iter_num_add // 2
            n_div = iter_num_add - n_score
            if use_mcmc_gmm and not getattr(scorer, 'has_mcmc_posterior', lambda: False)():
                logging.info(
                    f"[Iteration {iteration_i+1}] {refset_extension_method} sampling needs MCMC posterior [pass]; "
                    f"running MCMC now"
                )
                _ensure_feature_rep(LIH, LIH_rep, current_n_features, dim_reduction_method)
                X_ref = np.asarray(LIH_rep[current_n_features][current_reference_set_indices])
                fit_gmm_class(
                    scorer,
                    X_ref,
                    1,
                    run_mcmc=True,
                )
            if use_mcmc_gmm and not getattr(scorer_fail, 'has_mcmc_posterior', lambda: False)():
                logging.info(
                    f"[Iteration {iteration_i+1}] {refset_extension_method} sampling needs MCMC posterior [fail]; "
                    f"running MCMC now"
                )
                _ensure_feature_rep(
                    LIH, LIH_rep, current_n_features_fail, dim_reduction_method,
                                        )
                X_ref_fail = np.asarray(
                    LIH_rep[current_n_features_fail][current_reference_set_indices]
                )
                fit_gmm_class(
                    scorer_fail,
                    X_ref_fail,
                    0,
                    run_mcmc=True,
                )
            prior_pass, prior_fail = run_prior_pass, run_prior_fail
            logging.info(
                f"[Iteration {iteration_i+1}] {refset_extension_method}: "
                f"{n_score} {score_name} + {n_div} diversity"
            )
            if n_score > 0 and score_name in ('epig', 'epigspearman'):
                if score_name == 'epigspearman':
                    logging.info(
                        f"[Iteration {iteration_i+1}] epigspearman ranks raw η "
                        f"on the target pool (n={n_target}); "
                        f"binomial label 1/{int(n_trials)}"
                    )
                new_reference_set_indices_by_score, uncertainties = extend_reference_set_epig(
                    scoring_model=scorer,
                    refset_indices=current_reference_set_indices,
                    target_vectors=LIH_rep[current_n_features],
                    num_add=n_score,
                    scoring_model_fail=scorer_fail,
                    target_vectors_fail=LIH_rep[current_n_features_fail],
                    prior_pass=prior_pass,
                    prior_fail=prior_fail,
                    labels=labels,
                    sample_weight=_ipw_sample_weight(),
                    rank_indices=np.arange(n_target),
                    n_trials=int(n_trials),
                    random_state=int(random_seed) * 100003 + int(iteration_i + 1),
                    objective=(
                        'spearman' if score_name == 'epigspearman' else 'pairwise'
                    ),
                )
            elif n_score > 0:
                logging.info(
                    f"[Iteration {iteration_i+1}] rankvar ranks raw η "
                    f"on the target pool (n={n_target}); average ties"
                )
                (
                    new_reference_set_indices_by_score,
                    uncertainties,
                    bald_log_marginal,
                    posterior_disagreement,
                ) = extend_reference_set_rankvar(
                    scoring_model=scorer,
                    refset_indices=current_reference_set_indices,
                    target_vectors=LIH_rep[current_n_features],
                    num_add=n_score,
                    scoring_model_fail=scorer_fail,
                    target_vectors_fail=LIH_rep[current_n_features_fail],
                    prior_pass=prior_pass,
                    prior_fail=prior_fail,
                    n_trials=n_trials,
                    rank_indices=np.arange(n_target),
                )
            else:
                new_reference_set_indices_by_score = []
            ref_for_div = current_reference_set_indices + new_reference_set_indices_by_score
            if n_div > 0:
                new_reference_set_indices_by_diversity = extend_reference_set_diversity(
                    ref_for_div,
                    LIH_rep[current_n_features],
                    num_add=n_div,
                    metric='euclidean',
                )
            else:
                new_reference_set_indices_by_diversity = []
            assert set(new_reference_set_indices_by_score).isdisjoint(
                set(new_reference_set_indices_by_diversity)
            ), "The two sets of new reference indices should be disjoint."
            new_reference_set_indices = (
                new_reference_set_indices_by_score
                + new_reference_set_indices_by_diversity
            )
            new_unc_indices = list(new_reference_set_indices_by_score)
            new_div_indices = list(new_reference_set_indices_by_diversity)

    else:
        raise NotImplementedError(f"Reference set extension method '{refset_extension_method}' is not implemented.")

    return {
        'new_reference_set_indices': new_reference_set_indices,
        'new_unc_indices': new_unc_indices,
        'new_div_indices': new_div_indices,
        'uncertainties': uncertainties,
        'aleatoric_uncertainties': aleatoric_uncertainties,
        'posterior_disagreement': posterior_disagreement,
        'bald_log_marginal': bald_log_marginal,
        'duration_reference_set_extension': time.time() - start_time_reference_set_extension,
        'iter_num_add': int(iter_num_add),
        'effective_sampling_method': effective_sampling_method,
    }
