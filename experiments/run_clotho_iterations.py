"""Clotho active-learning iteration runner (CLI + orchestration)."""
import gc
import json
import os
import random
import time
import torch
import argparse
import logging
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from dataclasses import replace
from utils import (
    get_test_results,
    target_testsuites,
    prompt_templates,
    load_input_hidden_states,
    get_vis_components,
    get_gmm_components_perplexity,
    convert_to_normalized_rank_score,
)

# Canonical Clotho task list. `--target_task all` / `--task all` expands to this.
CLOTHO_TASKS = tuple(prompt_templates.keys())

import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr, rankdata
from sklearn.metrics import roc_auc_score

from clotho.preprocessing.feature_reduction import PCAFeatureReducer
from clotho.metrics.reference_based.density_estimation import (
    GMMScorer,
    GMMScorerConfig,
    MCMCGMMScorer,
    MCMCGMMScorerConfig,
)
from clotho.metrics.reference_based.pymc_gmm import (
    diag_gmm_aic_bic,
)
from clotho.al.sampling import (
    al_ipw_weights_for_new_points,
    flag_dual_density_ood,
    load_conservative_n0_lambda,
)

from plot_utils import (
    compute_bald_pool_diagnostics,
    plot_bald_decomposition_diagnostics,
    plot_logprob_ranks,
    plot_probability_density,
    plot_reference_points_simple,
    plot_score_distribution,
    plot_uncertainties,
    plot_uncertainty_marginal_dist,
    plot_uncertainty_prediction_compare,
    save_figure,
)

from clotho_run_helpers import (
    ALL_POSS_COMB,
    ALEATORIC_UNC_VMAX,
    BALD_UNC_VMAX,
    BAYES_GMM_MODELS,
    CV_MIN_FOLD_FLOOR,
    D_SCHEDULE_FEATURE_STEP,
    D_SCHEDULE_INIT_N_FEATURES,
    DEFAULT_EM,
    DEFAULT_N_TRIALS,
    DEFAULT_PLOT_EVERY_POINTS,
    FAST_POSTERIOR_EVERY,
    FAST_THIN_STRIDE,
    FAST_WARMUP_ADD,
    FEATURE_STEP,
    GENEROUS_EM,
    INIT_N_FEATURES,
    INIT_REG_COVAR_CV,
    MAX_N_FEATURES_GMM,
    MAX_N_FEATURES_MCMC,
    MIN_CLASS_REFS_TO_FIT,
    MIN_CLUSTER_SIZE,
    MIN_CLUSTER_SIZE_MCMC,
    MIN_N_FEATURES,
    NUM_ADD,
    PASS_THRESH,
    RANDOM_START_N_REFS,
    _apply_em_settings,
    _cheap_probe_n_init,
    _success_rate_weights,
    _class_fit_labels_and_weight,
    _em_settings,
    _ensure_feature_rep,
    _format_reg_covar_tag,
    _gmm_warm_start_init_params,
    _is_gmm_collapse_error,
    _is_recoverable_mcmc_fit_error,
    _maybe_update_n_components_from_perplexity,
    _results_clotho_root,
    _serialize_fit_profile,
    _uncertainty_color_limits_for_method,
    bayes_pass_posterior,
    set_global_seed,
)
from clotho_run_cv import (
    construct_cv_scorer,
    construct_scorer,
    ensure_nonzero_entropy,
    get_next_n_features,
    get_next_reg_covar_decay,
)
from clotho_run_viz import _save_gmm_fitting_figures
from clotho_run_acquire import acquire_next_reference_indices

def main(target_task, target_layer, refset_extension_methods=['random', 'diversity'], scoring_model='GMM', num_add=NUM_ADD, random_seeds=[42], target_llm='llama', verbose=False, covariance_type='diag', dual_gmm=True, oracle_prior=False, generous=False, reg_covar=None, n_features=None, fast=False, verbose_figure=False, map_score=False, reg_decay=True, d_schedule=True, d_fail=True, random_100_start=False, random_100_start_weighted=False, random_start_weighted=False, random_start_double_hard_weighted=False, random_start_sum_weighted=False, random_start_sum_double_hard_weighted=False, success_rate_weight=False, epistemic_multistart=False, epistemic_em_inits=4, max_n_ref=None, n_trials=DEFAULT_N_TRIALS, jax_platform="cpu"):
    em = _em_settings(generous, reg_covar=reg_covar)
    logging.info(
        f"[{datetime.now()}] Starting experiments for target_task={target_task}, target_llm={target_llm}, "
        f"scoring_model={scoring_model}, covariance_type={covariance_type}, dual_gmm={dual_gmm}, "
        f"oracle_prior={oracle_prior}, generous={generous}, reg_covar={em['reg_covar']}, "
        f"n_features={'CV' if n_features is None else n_features}, "
        f"d_schedule={d_schedule}, d_fail={d_fail}, "
        f"reg_decay={reg_decay}, fast={fast}, random_100_start={random_100_start}, "
        f"random_100_start_weighted={random_100_start_weighted}, "
        f"random_start_weighted={random_start_weighted}, "
        f"random_start_double_hard_weighted={random_start_double_hard_weighted}, "
        f"random_start_sum_weighted={random_start_sum_weighted}, "
        f"random_start_sum_double_hard_weighted={random_start_sum_double_hard_weighted}, "
        f"success_rate_weight={success_rate_weight}, "
        f"epistemic_multistart={epistemic_multistart}, epistemic_em_inits={epistemic_em_inits}, "
        f"max_n_ref={max_n_ref}, n_trials={int(n_trials)}, jax_platform={jax_platform}, "
        f"verbose_figure={verbose_figure}, map_score={map_score}, "
        f"refset_extension_methods={refset_extension_methods}, "
        f"random_seeds={random_seeds}, verbose={verbose}"
    )

    target_llm_suffix = f'_{target_llm}'
    result_suffix = f'_{target_llm}_{scoring_model}_{covariance_type}'
    if dual_gmm:
        result_suffix = f'{result_suffix}_dual'
    if oracle_prior:
        result_suffix = f'{result_suffix}_oracle_prior'
    if generous:
        result_suffix = f'{result_suffix}_generous'
    if reg_decay:
        result_suffix = f'{result_suffix}_regdecay'
    elif reg_covar is not None:
        # Only tag when --reg_covar is explicit. Untagged = original Clotho
        # (start at EM preset, raise λ as entropy-variance fallback).
        result_suffix = f'{result_suffix}_{_format_reg_covar_tag(float(reg_covar))}'
    if n_features is not None:
        result_suffix = f'{result_suffix}_d{int(n_features)}'
    if d_schedule:
        result_suffix = f'{result_suffix}_dsched'
    if d_fail:
        result_suffix = f'{result_suffix}_dfail'
    if map_score:
        result_suffix = f'{result_suffix}_mapscore'
    if fast:
        result_suffix = f'{result_suffix}_fast'
    if random_start_sum_double_hard_weighted:
        result_suffix = f'{result_suffix}_rstart_lam_2x_sum'
    elif random_start_sum_weighted:
        result_suffix = f'{result_suffix}_rstart_lam_sum'
    elif random_start_double_hard_weighted:
        result_suffix = f'{result_suffix}_rstart_lam_2x'
    elif random_start_weighted:
        result_suffix = f'{result_suffix}_rstart_lam'
    elif random_100_start_weighted:
        result_suffix = f'{result_suffix}_r100start_ipw'
    elif random_100_start:
        result_suffix = f'{result_suffix}_r100start'
    if success_rate_weight:
        result_suffix = f'{result_suffix}_srw'
    if epistemic_multistart:
        n_epi = int(epistemic_em_inits)
        result_suffix = f'{result_suffix}_epimulti' + ('' if n_epi == 4 else str(n_epi))
    if max_n_ref is not None:
        result_suffix = f'{result_suffix}_n{int(max_n_ref)}'
    n_trials = int(n_trials)
    if n_trials < 1:
        raise ValueError(f"--n_trials must be >= 1, got {n_trials}")
    if n_trials != DEFAULT_N_TRIALS:
        result_suffix = f'{result_suffix}_ntrials{n_trials}'
    
    logging.info(f"> Loading initial inference results and hidden vectors for task: {target_task}, layer: {target_layer}, target_llm: {target_llm}")
    with open(f'./initial_tests{target_llm_suffix}/{target_task}/inference_results.json', 'r') as f:
        inference_results = json.load(f)
    LIH_init = torch.load(f'./initial_tests{target_llm_suffix}/{target_task}/hidden_vectors_layer_{target_layer}.pt', weights_only=False)
    LIH_target = load_input_hidden_states(target_task, target_layer, model=target_llm)
    _, _, test_scores = get_test_results(target_llm, target_task)
    
    LIH_vis = get_vis_components(np.concatenate((LIH_target, LIH_init), axis=0))
    
    logging.info(f"> LIHS_initial: {LIH_init.shape}, LIHS_target: {LIH_target.shape}")

    n_iterations = 50
    if max_n_ref is not None:
        n_init = int(LIH_init.shape[0])
        n_target_ref = int(max_n_ref)
        if n_target_ref <= n_init:
            raise ValueError(
                f"--max_n_ref={n_target_ref} must exceed n_init={n_init}"
            )
        leftover = (n_target_ref - n_init) % int(num_add)
        if leftover != 0:
            raise ValueError(
                f"--max_n_ref={n_target_ref} is not reachable from n_init={n_init} "
                f"with num_add={num_add}"
            )
        n_iterations = (n_target_ref - n_init) // int(num_add)
        logging.info(
            f"> --max_n_ref={n_target_ref}: n_init={n_init}, num_add={num_add} "
            f"=> n_iterations={n_iterations}"
        )

    for refset_extension_method in refset_extension_methods:
        logging.info(f"task={target_task}, refset_extension_method={refset_extension_method}, random_seeds-{random_seeds}")
        
        for random_seed in tqdm(random_seeds, position=0, desc="{}, {} (seeds: {})".format(target_task, refset_extension_method, len(random_seeds)), leave=True):
            result_df, iteration_records = simulate_iterations(
                target_task=target_task,
                target_layer=target_layer,
                LIH_init=LIH_init,
                LIH_target=LIH_target,
                labels_init=inference_results['test_scores'],
                labels_target=test_scores,
                scoring_model=scoring_model,
                dim_reduction_method='pca',
                refset_extension_method=refset_extension_method,
                random_seed=random_seed,
                n_iterations=n_iterations,
                max_n_components=50,
                max_n_features=MAX_N_FEATURES_MCMC if scoring_model in BAYES_GMM_MODELS else MAX_N_FEATURES_GMM,
                num_add=num_add,
                LIH_vis=LIH_vis,
                result_suffix=result_suffix,
                target_llm=target_llm,
                plot_intermediate_results=True,
                verbose=verbose,
                covariance_type=covariance_type,
                dual_gmm=dual_gmm,
                oracle_prior=oracle_prior,
                generous=generous,
                reg_covar=reg_covar,
                n_features=n_features,
                fast=fast,
                verbose_figure=verbose_figure,
                map_score=map_score,
                reg_decay=reg_decay,
                                d_schedule=d_schedule,
                d_fail=d_fail,
                random_100_start=random_100_start,
                random_100_start_weighted=random_100_start_weighted,
                random_start_weighted=random_start_weighted,
                random_start_double_hard_weighted=random_start_double_hard_weighted,
                random_start_sum_weighted=random_start_sum_weighted,
                random_start_sum_double_hard_weighted=random_start_sum_double_hard_weighted,
                success_rate_weight=success_rate_weight,
                epistemic_multistart=epistemic_multistart,
                epistemic_em_inits=epistemic_em_inits,
                n_trials=n_trials,
                jax_platform=jax_platform,
            )

            result_dir = (
                f'{_results_clotho_root(scoring_model)}/'
                f'LIHS_refset_iter_{num_add}{result_suffix}/'
                f'{target_task}/layer_{target_layer}_seed_{random_seed}'
            )
            os.makedirs(result_dir, exist_ok=True)
            result_df.to_pickle(f'{result_dir}/{refset_extension_method}_pca.pkl')

            with open(f'{result_dir}/{refset_extension_method}_pca_iteration_records.json', 'w') as f:
                json.dump(iteration_records, f, indent=2)



def simulate_iterations(target_task, target_layer, LIH_init, LIH_target, labels_init, labels_target, scoring_model='GMM', dim_reduction_method='pca', refset_extension_method='balanced', random_seed=42, n_iterations=50, max_n_components=50, max_n_features=50, num_add=NUM_ADD, LIH_vis=None, result_suffix='_llama', target_llm=None, plot_intermediate_results=False, verbose=False, covariance_type='diag', dual_gmm=True, oracle_prior=False, generous=False, reg_covar=None, n_features=None, fast=False, verbose_figure=False, map_score=False, reg_decay=True, d_schedule=True, d_fail=True, random_100_start=False, random_100_start_weighted=False, random_start_weighted=False, random_start_double_hard_weighted=False, random_start_sum_weighted=False, random_start_sum_double_hard_weighted=False, success_rate_weight=False, epistemic_multistart=False, epistemic_em_inits=4, n_trials=DEFAULT_N_TRIALS, jax_platform="cpu"):
    set_global_seed(random_seed)
    LIH = np.concatenate((LIH_target, LIH_init), axis=0)
    labels = np.concatenate((labels_target, labels_init), axis=0)
    labels_binary = (labels > PASS_THRESH).astype(int)
    # Target-pool size; initial refs are appended after this (RQ eval excludes only those).
    n_target = len(LIH_target)
    initial_reference_set_indices = list(range(n_target, len(LIH)))
    # Dual Bayes / BALD class priors: equal by default, or GT pass/fail rates.
    if oracle_prior:
        n_pass = int(np.sum(labels_binary == 1))
        n_fail = int(np.sum(labels_binary == 0))
        n_total = n_pass + n_fail
        if n_pass == 0 or n_fail == 0:
            raise ValueError(
                f"oracle_prior requires both pass and fail labels "
                f"(got n_pass={n_pass}, n_fail={n_fail})."
            )
        run_prior_pass = n_pass / n_total
        run_prior_fail = n_fail / n_total
        logging.info(
            f"[simulate_iterations] oracle_prior: "
            f"prior_pass={run_prior_pass:.4f}, prior_fail={run_prior_fail:.4f} "
            f"(n_pass={n_pass}, n_fail={n_fail})"
        )
        if not dual_gmm:
            logging.warning(
                "oracle_prior has no effect without --dual_gmm "
                "(class priors are only used for dual Bayes / BALD)."
            )
    else:
        run_prior_pass, run_prior_fail = 0.5, 0.5
    if LIH_vis is None:
        LIH_vis = get_vis_components(LIH, random_state=random_seed)
    
    current_reference_set_indices = list(initial_reference_set_indices)
    n0_gate_sum = bool(
        random_start_sum_weighted or random_start_sum_double_hard_weighted
    )
    n0_difficulty_scale = 2.0 if (
        random_start_double_hard_weighted or random_start_sum_double_hard_weighted
    ) else (
        1.0 if (random_start_weighted or random_start_sum_weighted) else None
    )
    if random_start_sum_double_hard_weighted:
        n0_flag_name = 'random_start_sum_double_hard_weighted'
    elif random_start_sum_weighted:
        n0_flag_name = 'random_start_sum_weighted'
    elif random_start_double_hard_weighted:
        n0_flag_name = 'random_start_double_hard_weighted'
    else:
        n0_flag_name = 'random_start_weighted'
    use_random_n0_start = bool(
        random_start_weighted
        or random_start_double_hard_weighted
        or random_start_sum_weighted
        or random_start_sum_double_hard_weighted
    )
    use_lambda_weight = bool(use_random_n0_start)
    success_rate_weight = bool(success_rate_weight)
    if success_rate_weight and use_lambda_weight:
        raise ValueError(
            "--success_rate_weight cannot be combined with lambda* "
            "(--random_start_weighted / --random_start_double_hard_weighted / "
            "--random_start_sum_weighted / --random_start_sum_double_hard_weighted)"
        )
    if success_rate_weight and random_100_start_weighted:
        raise ValueError(
            "--success_rate_weight cannot be combined with --random_100_start_weighted (IPW)"
        )
    use_random_100_start = bool(
        (random_100_start or random_100_start_weighted) and not use_random_n0_start
    )
    use_ipw = bool(random_100_start_weighted) and not use_lambda_weight and not success_rate_weight
    n0_per_class = None
    lambda_star = 1.0
    n0_lambda_info = None
    if use_random_n0_start:
        n0_per_class, lambda_star, n0_lambda_info = load_conservative_n0_lambda(
            target_task, difficulty_scale=n0_difficulty_scale
        )
        gate_desc = (
            f"random until n_pass+n_fail>n0={n0_per_class}, then {refset_extension_method}"
            if n0_gate_sum
            else f"random until n_pass>n0 and n_fail>n0, then {refset_extension_method}"
        )
        logging.info(
            f"[simulate_iterations] --{n0_flag_name}: PCA-{n0_lambda_info['pca_dim']} "
            f"conservative n0={n0_per_class}"
            f"{' (sum gate)' if n0_gate_sum else '/class'} "
            f"(R* upper={n0_lambda_info['Rstar_upper']}, "
            f"difficulty_scale={n0_lambda_info['difficulty_scale']:g}, "
            f"u={n0_lambda_info['u']:.4g}, alpha={n0_lambda_info['alpha']:.4g}), "
            f"BALD EM weight lambda*={lambda_star:.4g}; "
            f"{gate_desc} "
            f"(no {MIN_CLASS_REFS_TO_FIT}/class fit gate)"
        )
    if success_rate_weight:
        logging.info(
            "[simulate_iterations] --success_rate_weight: pass GMM sample_weight=k/10, "
            "fail GMM sample_weight=1-k/10; every reference is kept (no hard 0/1 split). "
            "Incompatible with lambda*/IPW."
        )
    # Init refs / random-phase points: weight 1. BALD points get lambda*.
    reference_ipw_weights = (
        [1.0] * len(current_reference_set_indices)
        if (use_ipw or use_lambda_weight)
        else None
    )
    use_mcmc_gmm = scoring_model in BAYES_GMM_MODELS
    # Dual: pass/fail share one PCA dim (Bayes needs a shared feature space).
    # Default CV uses pass GMM; --d_fail retargets that shared-d CV to fail.
    # --d freezes PCA dim for the whole run (no feature CV).
    # --d_schedule: start at 5 and only step up (strict HD gate; no MCMC backoff).
    # --d_fail + --d_schedule: same monotone schedule, HD gate on fail-GMM CV.
    d_fail = bool(d_fail and dual_gmm)
    min_class_refs = MIN_CLASS_REFS_TO_FIT
    if n_features is not None:
        n_features = int(n_features)
        if n_features < 1:
            raise ValueError(f"--d / n_features must be >= 1, got {n_features}")
        current_n_features = n_features
        current_n_features_fail = n_features
        logging.info(
            f"[simulate_iterations] PCA n_features fixed at {n_features} "
            f"(skipping feature CV; MCMC dim backoff disabled)"
        )
    elif d_schedule:
        current_n_features = D_SCHEDULE_INIT_N_FEATURES
        current_n_features_fail = D_SCHEDULE_INIT_N_FEATURES
        logging.info(
            f"[simulate_iterations] --d_schedule: PCA starts at "
            f"{D_SCHEDULE_INIT_N_FEATURES}, only increases by "
            f"{D_SCHEDULE_FEATURE_STEP} "
            f"(HD needs higher mean and stable min fold), "
            f"MCMC dim backoff disabled"
        )
    else:
        current_n_features = INIT_N_FEATURES
        current_n_features_fail = INIT_N_FEATURES
    current_n_components = 5
    current_n_components_fail = 5
    feature_step = D_SCHEDULE_FEATURE_STEP if d_schedule else FEATURE_STEP
    if d_fail:
        logging.info(
            f"[simulate_iterations] --d_fail: shared PCA dim from fail-GMM CV "
            f"(pass follows; d_schedule={bool(d_schedule)})"
        )
    cluster_update_size = 1
    
    scorer = None
    scorer_fail = None
    mcmc_draws = 500 if fast else 1000
    mcmc_tune = 500 if fast else 1000
    posterior_every = FAST_POSTERIOR_EVERY if fast else None
    thin_stride = FAST_THIN_STRIDE if fast else None
    bald_thin_draws = (
        max(1, mcmc_draws // FAST_THIN_STRIDE) if fast else None
    )
    if use_mcmc_gmm:
        _jax_platform = str(jax_platform or 'cpu').lower()
        if _jax_platform in ('gpu', 'cuda'):
            _jax_platform = 'cuda'
        elif _jax_platform != 'cpu':
            raise ValueError(
                f'Unsupported jax_platform={jax_platform!r} (use cpu|cuda|gpu)'
            )
        # Single-GPU: vectorized chains. Multi-CPU host: parallel pmap.
        _nuts_kwargs = (
            {'chain_method': 'vectorized'}
            if _jax_platform == 'cuda'
            else {'chain_method': 'parallel'}
        )
        scorer_config = MCMCGMMScorerConfig(
            n_components=current_n_components,
            random_state=random_seed,
            mcmc_draws=mcmc_draws,
            mcmc_tune=mcmc_tune,
            mcmc_chains=int(epistemic_em_inits) if epistemic_multistart else 1,
            target_accept=0.9,
            n_uncertainty_draws=max(1, mcmc_draws // FAST_THIN_STRIDE) if fast else 100,
            uncertainty_thin_stride=thin_stride,
            inference='advi' if scoring_model == 'ADVIGMM' else 'mcmc',
            em_centered_prior=not bool(epistemic_multistart),
            epistemic_multistart=bool(epistemic_multistart),
            epistemic_em_inits=int(epistemic_em_inits),
            jax_platform=_jax_platform,
            nuts_sampler_kwargs=_nuts_kwargs,
            verbose=verbose,
        )
        logging.info(
            f'[simulate_iterations] JAX/NUTS platform={_jax_platform}, '
            f'chain_method={_nuts_kwargs.get("chain_method")}'
        )
        if fast:
            logging.info(
                f"[simulate_iterations] --fast: NUTS tune/draws={mcmc_tune}/{mcmc_draws}, "
                f"posterior every {FAST_POSTERIOR_EVERY} iters, "
                f"BALD thin stride={FAST_THIN_STRIDE}, "
                f"warmup add={FAST_WARMUP_ADD} until {min_class_refs}/class"
            )
    else:
        scorer_config = GMMScorerConfig(
            n_components=current_n_components,
            random_state=random_seed,
            covariance_type=covariance_type,
            verbose=verbose,
        )
    init_reg_covar = float(reg_covar) if reg_covar is not None else (
        INIT_REG_COVAR_CV if reg_decay else None
    )
    scorer_config = _apply_em_settings(scorer_config, generous, reg_covar=init_reg_covar)
    em = _em_settings(generous, reg_covar=init_reg_covar)
    logging.info(
        f"[simulate_iterations] EM settings ({'generous' if generous else 'rigorous'}"
        f"{', reg_covar override' if init_reg_covar is not None else ''}"
        f"{', reg_decay' if reg_decay else ''}): "
        f"n_init={em['n_init']}, max_iter={em['max_iter']}, tol={em['tol']}, "
        f"reg_covar={em['reg_covar']}"
        + (
            f" (monotonic λ vs λ/10 decay from {em['reg_covar']:g}; "
            f"{'dual Bayes Pearson' if dual_gmm else 'rank Pearson'}; "
            f"mean + min-fold > max(current min, {CV_MIN_FOLD_FLOOR:g}))"
            if reg_decay else ""
        )
    )
    if epistemic_multistart:
        if not use_mcmc_gmm:
            raise ValueError("--epistemic_multistart requires scoring_model='MCMCGMM'")
        logging.info(
            f"[simulate_iterations] --epistemic_multistart: {int(epistemic_em_inits)} "
            f"loose EMs (n_init={scorer_config.epistemic_em_n_init}, distinct seeds) "
            f"as uncentered NUTS inits; scoring EM stays n_init={scorer_config.sklearn_n_init}"
        )
    if map_score:
        logging.info(
            "[simulate_iterations] --map_score: Pearson/Spearman/ROC-AUC and "
            "plotted logprob use the highest-likelihood MCMC draw "
            "(BALD still uses the full posterior)"
        )
    if use_random_100_start:
        logging.info(
            f"[simulate_iterations] --random_100_start"
            f"{' (IPW)' if use_ipw else ''}: random acquisition until "
            f"n_ref={RANDOM_START_N_REFS}, then {refset_extension_method} "
            f"(MCMC/BALD deferred until the switch)"
            + (
                "; GMM/MCMC fits use inverse-propensity weights "
                "(random: π=k/N, uncertainty: softmax top-k π)"
                if use_ipw
                else ""
            )
        )
    logging.info(
        "[simulate_iterations] figures: "
        + (
            "every iteration (--verbose_figure)"
            if verbose_figure
            else f"every {DEFAULT_PLOT_EVERY_POINTS} refs (plus last iter)"
        )
    )
    if use_mcmc_gmm and verbose:
        logging.info(
            f"[simulate_iterations] {scoring_model}: cov=diag, inference="
            f"{getattr(scorer_config, 'inference', 'mcmc')}, "
            f"mu_prior_sigma={scorer_config.mu_prior_sigma}, dual_gmm={dual_gmm}"
        )
    # Independent config for the fail GMM (hyperparameters diverge over iterations).
    scorer_config_fail = replace(scorer_config, n_components=current_n_components_fail) if dual_gmm else None
    LIH_rep = {} # feature_size -> vector representation mappings
    use_entropy_sampling = refset_extension_method in (
        'uncertainty', 'balanced', 'bald', 'batchbald', 'rankvar', 'epig', 'llvar',
        'rankvarbalanced', 'epigbalanced', 'epigspearmanbalanced',
    )
    nonzero_entropy_checked = False
    last_added_indices = []
    last_added_unc_indices = []
    last_added_div_indices = []
    prev_uncertainty_plot = None  # {df_vis, n_ref, iteration, n_feat, n_comp, plot_dir}
    uncertainty_plot_history = []  # snapshots for run-level color rescale of prediction figs
    bald_diag_history = []  # per-iter pool BALD decomposition scalars
    run_unc_vmin, run_unc_vmax = _uncertainty_color_limits_for_method(
        refset_extension_method, scoring_model=scoring_model
    )
    
    df_all_iterations = pd.DataFrame()
    iteration_records_all = {}
    logged_random_start_switch = False

    def _ipw_sample_weight():
        if not (use_ipw or use_lambda_weight):
            return None
        w = np.asarray(reference_ipw_weights, dtype=float)
        if w.shape[0] != len(current_reference_set_indices):
            raise RuntimeError(
                f"sample_weight length {w.shape[0]} != n_ref {len(current_reference_set_indices)}"
            )
        return w

    def _class_fit_yw(target_class):
        return _class_fit_labels_and_weight(
            reference_set_labels,
            labels[current_reference_set_indices],
            int(target_class),
            sample_weight=_ipw_sample_weight(),
            success_rate_weight=success_rate_weight,
        )

    def _fit_gmm_class(scorer_obj, X, target_class, **fit_kwargs):
        y_fit, w_fit = _class_fit_yw(target_class)
        return scorer_obj.fit(X, y_fit, sample_weight=w_fit, **fit_kwargs)

    for i in tqdm(range(n_iterations), desc="Simulating iterations", position=1, leave=False):
        skip = False
        gmm_collapse_fallback = False
        # Defaults for skip / collapse paths (scoring block may overwrite).
        prior_pass, prior_fail = run_prior_pass, run_prior_fail
        n_ref = len(current_reference_set_indices)
        ref_labels_bin = (labels[current_reference_set_indices] > PASS_THRESH).astype(int)
        n_pass_now = int(np.sum(ref_labels_bin == 1))
        n_fail_now = int(np.sum(ref_labels_bin == 0))
        if use_random_n0_start:
            if n0_gate_sum:
                # Keep random until pooled pass+fail strictly exceeds n0.
                use_random_startpoint = bool(
                    (n_pass_now + n_fail_now) <= n0_per_class
                )
            else:
                # Keep random until both classes have strictly more than n0 samples.
                use_random_startpoint = bool(
                    n_pass_now <= n0_per_class or n_fail_now <= n0_per_class
                )
        else:
            use_random_startpoint = bool(
                use_random_100_start and n_ref < RANDOM_START_N_REFS
            )
        if (
            (use_random_100_start or use_random_n0_start)
            and not use_random_startpoint
            and not logged_random_start_switch
        ):
            if use_random_n0_start:
                if n0_gate_sum:
                    logging.info(
                        f"[Iteration {i+1}] --{n0_flag_name}: "
                        f"n_pass={n_pass_now}, n_fail={n_fail_now} "
                        f"sum={n_pass_now + n_fail_now} > n0={n0_per_class}, "
                        f"switching to {refset_extension_method}"
                    )
                else:
                    logging.info(
                        f"[Iteration {i+1}] --{n0_flag_name}: "
                        f"n_pass={n_pass_now}, n_fail={n_fail_now} both > n0={n0_per_class}, "
                        f"switching to {refset_extension_method}"
                    )
            else:
                logging.info(
                    f"[Iteration {i+1}] --random_100_start: n_ref={n_ref} "
                    f">= {RANDOM_START_N_REFS}, switching to {refset_extension_method}"
                )
            logged_random_start_switch = True
        
        iteration_record = {
            'duration_feature_reduction': None,
            'duration_model_fitting': None,
            'duration_reference_set_extension': None,
            'duration_total': None,
            'lower_n_features': None,
            'current_n_features': None,
            'larger_n_features': None,
            'selected_n_features': None,
            'n_components': None,
            'n_reference_set': len(current_reference_set_indices)
        }

        num_effective_references = len([l for l in labels[current_reference_set_indices] if l > PASS_THRESH])
        num_fail_references = len([l for l in labels[current_reference_set_indices] if l <= PASS_THRESH])

        if scoring_model in ('GMM',) or scoring_model in BAYES_GMM_MODELS:
            reference_set_labels = (labels[current_reference_set_indices] > PASS_THRESH).astype(int)
            n_pass_ref = int(np.sum(reference_set_labels == 1))
            n_fail_ref = int(np.sum(reference_set_labels == 0))
            if use_random_n0_start:
                # n0 already gates acquisition; do not also require
                # MIN_CLASS_REFS_TO_FIT. Skip the GMM fit only while random
                # start is still active.
                if use_random_startpoint:
                    skip = True
            elif n_pass_ref < min_class_refs:
                logging.info(
                    f"[simulate_iterations] ({i+1}th iteration): Not enough passing reference points to fit {scoring_model}. "
                    f"(Current: {n_pass_ref}, Required: {min_class_refs}) Skipping this iteration."
                )
                skip = True
            elif dual_gmm and n_fail_ref < min_class_refs:
                logging.info(
                    f"[simulate_iterations] ({i+1}th iteration): Not enough failing reference points to fit dual {scoring_model}. "
                    f"(Current: {n_fail_ref}, Required: {min_class_refs}) Skipping this iteration."
                )
                skip = True
        else:
            raise NotImplementedError(f"Scoring model '{scoring_model}' is not implemented.")

        start_time = time.time()
        # Ensure PCA rep for the (shared) scoring feature dim.
        if current_n_features not in LIH_rep:
            if dim_reduction_method == 'pca':
                dim_reducer = PCAFeatureReducer(
                    n_features=current_n_features
                )
            else:
                raise NotImplementedError(f"Dimensionality reduction method '{dim_reduction_method}' is not implemented.")
            LIH_rep[current_n_features] = np.asarray(dim_reducer.fit_transform(LIH))
        if dual_gmm:
            # Fail GMM uses the same PCA dim as pass (Bayes needs a shared feature space).
            current_n_features_fail = current_n_features

        if not skip:
            try:
                # 1. Update n_features (PCA CV still uses a cheap sklearn surrogate for MCMCGMM).
                # Dual: shared dim. Default = pass CV; --d_fail = fail CV (same space).
                # --d: keep the frozen dim (shared by pass/fail).
                if n_features is None:
                    cv_target_class = 0 if d_fail else 1
                    cv_scorer_config = (
                        scorer_config_fail if d_fail else scorer_config
                    )
                    current_n_features = get_next_n_features(
                        LIH,
                        LIH_rep,
                        current_reference_set_indices,
                        labels,
                        current_n_features,
                        max_n_features,
                        dim_reduction_method,
                        cv_scorer_config,
                        scoring_model=scoring_model,
                        feature_step=feature_step,
                        target_class=cv_target_class,
                                                d_schedule=d_schedule,
                        sample_weight=_ipw_sample_weight(),
                        success_rate_weight=success_rate_weight,
                    )
                else:
                    current_n_features = int(n_features)
                    if verbose:
                        logging.info(
                            f"[Iteration {i+1}] n_features fixed at {current_n_features} "
                            f"(--d); skipping PCA CV"
                        )
                if dual_gmm:
                    current_n_features_fail = current_n_features
                    if verbose:
                        if d_fail:
                            logging.info(
                                f"[Iteration {i+1}] dual_gmm: shared n_feat={current_n_features} "
                                f"from fail PCA CV"
                            )
                        else:
                            logging.info(
                                f"[Iteration {i+1}] dual_gmm: fail n_features follows pass "
                                f"(n_feat={current_n_features})"
                            )

                # 1b. Optional monotone λ vs λ/10 decay (--reg_decay).
                if reg_decay:
                    X_ref_pass = np.asarray(
                        LIH_rep[current_n_features][current_reference_set_indices]
                    )
                    prev_reg = float(scorer_config.reg_covar)
                    next_reg = get_next_reg_covar_decay(
                        prev_reg,
                        X_ref_pass,
                        labels[current_reference_set_indices],
                        scorer_config,
                        scoring_model=scoring_model,
                        target_class=1,
                        dual_gmm=dual_gmm,
                        scorer_config_fail=scorer_config_fail if dual_gmm else None,
                        sample_weight=_ipw_sample_weight(),
                        success_rate_weight=success_rate_weight,
                    )
                    scorer_config = replace(scorer_config, reg_covar=float(next_reg))
                    logging.info(
                        f"[Iteration {i+1}] reg_covar decay [pass]: {prev_reg:g} -> {next_reg:g}"
                    )
                    if dual_gmm:
                        prev_reg_fail = float(scorer_config_fail.reg_covar)
                        scorer_config_fail = replace(
                            scorer_config_fail, reg_covar=float(next_reg)
                        )
                        logging.info(
                            f"[Iteration {i+1}] dual_gmm: fail reg_covar follows pass "
                            f"({prev_reg_fail:g} -> {next_reg:g})"
                        )

                # Pick k from a cheap EM weight-perplexity probe before the main fit
                # (MCMCGMM scoring also uses EM, so the same probe applies).
                probe_weights = None
                perplexity = None
                probe_weights_fail = None
                perplexity_fail = None
                if verbose:
                    logging.info(
                        f"[Iteration {i+1}] stage=n_components_probe "
                        f"({'MCMCGMM EM surrogate' if use_mcmc_gmm else 'sklearn GMM'})"
                        f"{' [pass]' if dual_gmm else ''}"
                    )
                if use_mcmc_gmm:
                    _probe = construct_cv_scorer(scorer_config, scoring_model=scoring_model)
                else:
                    _probe = construct_scorer(
                        replace(scorer_config, n_init=_cheap_probe_n_init(scorer_config)),
                        scoring_model=scoring_model,
                    )
                _fit_gmm_class(
                    _probe,
                    np.asarray(LIH_rep[current_n_features][current_reference_set_indices]),
                    1,
                )
                probe_weights = np.asarray(_probe.model.weights_, dtype=float)
                perplexity = get_gmm_components_perplexity(probe_weights)
                logging.debug(
                    f"[n_components probe (Iteration: {i+1}){' pass' if dual_gmm else ''}] "
                    f"k={current_n_components}, perplexity={perplexity:.4f}, "
                    f"weights={np.array2string(probe_weights, precision=4, suppress_small=True)}"
                )
                if verbose:
                    logging.info(
                        f"[Iteration {i+1}] stage=n_components_probe done"
                        f"{' [pass]' if dual_gmm else ''}: "
                        f"k={current_n_components}, perplexity={perplexity:.4f}"
                    )
                next_n_components = _maybe_update_n_components_from_perplexity(
                    current_n_components,
                    perplexity,
                    (
                        len(current_reference_set_indices)
                        if success_rate_weight
                        else num_effective_references
                    ),
                    max_n_components,
                    iteration=i + 1,
                    cluster_update_size=cluster_update_size,
                    min_n_components=MIN_CLUSTER_SIZE_MCMC if use_mcmc_gmm else MIN_CLUSTER_SIZE,
                )
                if next_n_components != current_n_components:
                    current_n_components = next_n_components
                    scorer_config = replace(scorer_config, n_components=current_n_components)

                if dual_gmm:
                    if verbose:
                        logging.info(
                            f"[Iteration {i+1}] stage=n_components_probe [fail] "
                            f"({'MCMCGMM EM surrogate' if use_mcmc_gmm else 'sklearn GMM'})"
                        )
                    if use_mcmc_gmm:
                        _probe_fail = construct_cv_scorer(scorer_config_fail, scoring_model=scoring_model)
                    else:
                        _probe_fail = construct_scorer(
                            replace(
                                scorer_config_fail,
                                n_init=_cheap_probe_n_init(scorer_config_fail),
                            ),
                            scoring_model=scoring_model,
                        )
                    _fit_gmm_class(
                        _probe_fail,
                        np.asarray(LIH_rep[current_n_features_fail][current_reference_set_indices]),
                        0,
                    )
                    probe_weights_fail = np.asarray(_probe_fail.model.weights_, dtype=float)
                    perplexity_fail = get_gmm_components_perplexity(probe_weights_fail)
                    logging.debug(
                        f"[n_components probe (Iteration: {i+1}) fail] "
                        f"k={current_n_components_fail}, perplexity={perplexity_fail:.4f}, "
                        f"weights={np.array2string(probe_weights_fail, precision=4, suppress_small=True)}"
                    )
                    if verbose:
                        logging.info(
                            f"[Iteration {i+1}] stage=n_components_probe done [fail]: "
                            f"k={current_n_components_fail}, perplexity={perplexity_fail:.4f}"
                        )
                    next_n_components_fail = _maybe_update_n_components_from_perplexity(
                        current_n_components_fail,
                        perplexity_fail,
                        (
                            len(current_reference_set_indices)
                            if success_rate_weight
                            else num_fail_references
                        ),
                        max_n_components,
                        iteration=i + 1,
                        cluster_update_size=cluster_update_size,
                        min_n_components=MIN_CLUSTER_SIZE_MCMC if use_mcmc_gmm else MIN_CLUSTER_SIZE,
                    )
                    if next_n_components_fail != current_n_components_fail:
                        current_n_components_fail = next_n_components_fail
                        scorer_config_fail = replace(
                            scorer_config_fail, n_components=current_n_components_fail
                        )

                if (
                    use_entropy_sampling
                    and not use_mcmc_gmm
                    and not reg_decay
                    and reg_covar is None
                    and not use_random_startpoint
                ):
                    if num_effective_references > 10 and not nonzero_entropy_checked:
                        y_ent, w_ent = _class_fit_yw(1)
                        success = ensure_nonzero_entropy(
                            np.asarray(LIH_rep[current_n_features][current_reference_set_indices]),
                            y_ent,
                            np.asarray(LIH_rep[current_n_features]),
                            scorer_config,
                            sample_weight=w_ent,
                        )
                        if not success:
                            logging.warning(
                                f"Failed to adapt scorer configuration to achieve nonzero variance in entropy values. "
                                f"This may lead to ineffective uncertainty-based sampling. "
                                f"(Iteration {i+1}, refset_extension_method: {refset_extension_method}, "
                                f"seed: {random_seed}, reg_covar: {scorer_config.reg_covar})"
                            )
                        nonzero_entropy_checked = True
                elif use_entropy_sampling and use_mcmc_gmm:
                    nonzero_entropy_checked = True

                start_time_model_fitting = time.time()
                # MCMCGMM: score with EM every iter; MCMC for uncertainty explore when
                # reference-set size n hits multiples of 10 (n=10,20,...) — NOT every 10
                # iterations — and whenever n_feat/k invalidate the posterior.
                # With --d, unchanged k reuses last posterior-mean params (skip EM).
                prev_idata = None
                prev_mcmc_n_feat = None
                prev_mcmc_n_comp = None
                prev_idata_fail = None
                prev_mcmc_n_feat_fail = None
                prev_mcmc_n_comp_fail = None
                prev_warm_params = None
                prev_warm_n_feat = None
                prev_warm_n_comp = None
                prev_warm_params_fail = None
                prev_warm_n_feat_fail = None
                prev_warm_n_comp_fail = None
                if use_mcmc_gmm and scorer is not None and getattr(scorer, 'has_mcmc_posterior', lambda: False)():
                    prev_idata = scorer.model.idata_
                    prev_mcmc_n_feat = scorer.model.n_features_in_
                    prev_mcmc_n_comp = scorer.model.n_components
                if use_mcmc_gmm and scorer is not None and scorer.check_is_fitted():
                    prev_warm_params = _gmm_warm_start_init_params(scorer)
                    prev_warm_n_feat = scorer.model.n_features_in_
                    prev_warm_n_comp = scorer.model.n_components
                if (
                    dual_gmm
                    and use_mcmc_gmm
                    and scorer_fail is not None
                    and getattr(scorer_fail, 'has_mcmc_posterior', lambda: False)()
                ):
                    prev_idata_fail = scorer_fail.model.idata_
                    prev_mcmc_n_feat_fail = scorer_fail.model.n_features_in_
                    prev_mcmc_n_comp_fail = scorer_fail.model.n_components
                if dual_gmm and use_mcmc_gmm and scorer_fail is not None and scorer_fail.check_is_fitted():
                    prev_warm_params_fail = _gmm_warm_start_init_params(scorer_fail)
                    prev_warm_n_feat_fail = scorer_fail.model.n_features_in_
                    prev_warm_n_comp_fail = scorer_fail.model.n_components

                n_ref = len(current_reference_set_indices)
                if posterior_every is not None:
                    mcmc_schedule = (i % int(posterior_every) == 0)
                else:
                    mcmc_schedule = (n_ref % 10 == 0)
                idata_compatible = (
                    prev_idata is not None
                    and prev_mcmc_n_feat == current_n_features
                    and prev_mcmc_n_comp == current_n_components
                )
                idata_compatible_fail = (
                    prev_idata_fail is not None
                    and prev_mcmc_n_feat_fail == current_n_features_fail
                    and prev_mcmc_n_comp_fail == current_n_components_fail
                )
                run_mcmc = bool(
                    use_mcmc_gmm
                    and use_entropy_sampling
                    and not use_random_startpoint
                    and (mcmc_schedule or not idata_compatible)
                )
                run_mcmc_fail = bool(
                    dual_gmm
                    and use_mcmc_gmm
                    and use_entropy_sampling
                    and not use_random_startpoint
                    and (mcmc_schedule or not idata_compatible_fail)
                )
                # --d freezes PCA dim: if k is also unchanged, skip EM and seed
                # NUTS/ADVI from last posterior-mean (or EM) params.
                frozen_d = n_features is not None
                warm_start = bool(
                    frozen_d
                    and prev_warm_params is not None
                    and prev_warm_n_feat == current_n_features
                    and prev_warm_n_comp == current_n_components
                    and int(np.asarray(prev_warm_params["mu"]).shape[0]) == current_n_components
                    and int(np.asarray(prev_warm_params["mu"]).shape[1]) == current_n_features
                )
                warm_start_fail = bool(
                    dual_gmm
                    and frozen_d
                    and prev_warm_params_fail is not None
                    and prev_warm_n_feat_fail == current_n_features_fail
                    and prev_warm_n_comp_fail == current_n_components_fail
                    and int(np.asarray(prev_warm_params_fail["mu"]).shape[0]) == current_n_components_fail
                    and int(np.asarray(prev_warm_params_fail["mu"]).shape[1]) == current_n_features_fail
                )

                scorer = construct_scorer(scorer_config, scoring_model=scoring_model)
                if dual_gmm:
                    scorer_fail = construct_scorer(scorer_config_fail, scoring_model=scoring_model)

                def _fit_mcmc_gmm_with_feature_backoff(
                    *,
                    run_mcmc_flag: bool,
                    target_class: int,
                    warm_start_flag: bool = False,
                    init_params=None,
                ):
                    """Fit MCMCGMM (EM, or warm-start if flag; MCMC optional).

                    On recoverable MCMC errors, lower PCA dim (disabled when --d
                    or --d_schedule). In dual_gmm mode pass/fail share one PCA dim:
                    backoff updates both.
                    """
                    nonlocal current_n_features, current_n_features_fail, scorer, scorer_fail
                    nonlocal run_mcmc, run_mcmc_fail, scorer_config, scorer_config_fail
                    is_fail = target_class == 0
                    # Shared feature dim in dual mode (fail follows pass).
                    _n_feat = current_n_features
                    _scorer_cfg = scorer_config_fail if is_fail else scorer_config
                    _n_comp = current_n_components_fail if is_fail else current_n_components
                    _warm = bool(warm_start_flag and init_params is not None)
                    _scorer = construct_scorer(_scorer_cfg, scoring_model=scoring_model)
                    if is_fail:
                        scorer_fail = _scorer
                    else:
                        scorer = _scorer
                    while True:
                        if verbose:
                            logging.info(
                                f"[Iteration {i+1}] stage=model_fitting "
                                f"(MCMCGMM{' fail' if is_fail else (' pass' if dual_gmm else '')}, "
                                f"{'warm_start' if _warm else 'em_score'}, run_mcmc={run_mcmc_flag}, "
                                f"draws={_scorer_cfg.mcmc_draws}, "
                                f"n_feat={_n_feat}, n_comp={_n_comp}, "
                                f"n_ref={len(current_reference_set_indices)})"
                            )
                        try:
                            _ensure_feature_rep(LIH, LIH_rep, _n_feat, dim_reduction_method)
                            X_ref = np.asarray(LIH_rep[_n_feat][current_reference_set_indices])
                            _fit_gmm_class(
                                _scorer,
                                X_ref,
                                target_class,
                                init_params=init_params if _warm else None,
                                warm_start=_warm,
                                run_mcmc=run_mcmc_flag,
                            )
                            current_n_features = _n_feat
                            if dual_gmm:
                                current_n_features_fail = _n_feat
                            if is_fail:
                                scorer_fail = _scorer
                            else:
                                scorer = _scorer
                            return
                        except Exception as fit_exc:
                            if n_features is not None or d_schedule:
                                _freeze_tag = (
                                    '--d_schedule forbids lowering PCA dim'
                                    if d_schedule and n_features is None
                                    else '--d forbids lowering PCA dim'
                                )
                                logging.error(
                                    f"[Iteration {i+1}] MCMCGMM fit failed at fixed n_feat={_n_feat} "
                                    f"({'fail' if is_fail else 'pass'}; {_freeze_tag}): "
                                    f"{fit_exc}"
                                )
                                raise
                            if not run_mcmc_flag or not _is_recoverable_mcmc_fit_error(fit_exc):
                                raise
                            next_n_features = _n_feat - 1
                            if next_n_features < MIN_N_FEATURES:
                                logging.error(
                                    f"[Iteration {i+1}] MCMCGMM fit failed at n_feat={_n_feat} "
                                    f"({'fail' if is_fail else 'pass'}) "
                                    f"and cannot lower further (min={MIN_N_FEATURES}): {fit_exc}"
                                )
                                raise
                            logging.warning(
                                f"[Iteration {i+1}] MCMCGMM fit failed at n_feat={_n_feat} "
                                f"({'fail' if is_fail else 'pass'}, {type(fit_exc).__name__}: {fit_exc}). "
                                f"Retrying with n_feat={next_n_features}"
                                f"{' (shared pass/fail dim)' if dual_gmm else ''}."
                            )
                            _n_feat = next_n_features
                            current_n_features = _n_feat
                            if dual_gmm:
                                current_n_features_fail = _n_feat
                            if is_fail:
                                run_mcmc_fail = True
                            else:
                                run_mcmc = True
                            run_mcmc_flag = True
                            _warm = False
                            _scorer = construct_scorer(_scorer_cfg, scoring_model=scoring_model)
                            if is_fail:
                                scorer_fail = _scorer
                            else:
                                scorer = _scorer

                if use_mcmc_gmm:
                    if warm_start:
                        logging.info(
                            f"[Iteration {i+1}] {scoring_model} warm-start"
                            f"{' [pass]' if dual_gmm else ''} "
                            f"(k={current_n_components}, d={current_n_features} unchanged; --d)"
                        )
                    _fit_mcmc_gmm_with_feature_backoff(
                        run_mcmc_flag=run_mcmc,
                        target_class=1,
                        warm_start_flag=warm_start,
                        init_params=prev_warm_params,
                    )
                    if not run_mcmc and idata_compatible:
                        scorer.model.idata_ = prev_idata
                        if verbose:
                            logging.info(
                                f"[Iteration {i+1}] reusing MCMC posterior from prior refresh"
                                f"{' [pass]' if dual_gmm else ''} "
                                f"(n_feat={prev_mcmc_n_feat}, n_comp={prev_mcmc_n_comp})"
                            )
                    # Record EM mixture weights used for scoring.
                    probe_weights = np.asarray(scorer.model.weights_, dtype=float)
                    perplexity = get_gmm_components_perplexity(probe_weights)
                    iteration_record['mcmc_run'] = bool(run_mcmc)
                    iteration_record['mcmc_posterior_available'] = bool(
                        getattr(scorer, 'has_mcmc_posterior', lambda: False)()
                    )
                    if dual_gmm:
                        n_feat_after_pass = current_n_features
                        current_n_features_fail = current_n_features
                        # idata reuse for fail must match the (possibly backoff-updated) shared dim.
                        idata_compatible_fail = (
                            prev_idata_fail is not None
                            and prev_mcmc_n_feat_fail == current_n_features_fail
                            and prev_mcmc_n_comp_fail == current_n_components_fail
                        )
                        run_mcmc_fail = bool(
                            use_mcmc_gmm
                            and use_entropy_sampling
                            and (mcmc_schedule or not idata_compatible_fail)
                        )
                        if warm_start_fail:
                            logging.info(
                                f"[Iteration {i+1}] {scoring_model} warm-start [fail] "
                                f"(k={current_n_components_fail}, "
                                f"d={current_n_features_fail} unchanged; --d)"
                            )
                        _fit_mcmc_gmm_with_feature_backoff(
                            run_mcmc_flag=run_mcmc_fail,
                            target_class=0,
                            warm_start_flag=warm_start_fail,
                            init_params=prev_warm_params_fail,
                        )
                        if current_n_features != n_feat_after_pass:
                            # Fail backoff lowered the shared dim; re-fit pass in the same space.
                            logging.warning(
                                f"[Iteration {i+1}] dual_gmm: fail backoff lowered shared n_feat "
                                f"{n_feat_after_pass} -> {current_n_features}; re-fitting pass"
                            )
                            run_mcmc = True
                            _fit_mcmc_gmm_with_feature_backoff(run_mcmc_flag=True, target_class=1)
                            probe_weights = np.asarray(scorer.model.weights_, dtype=float)
                            perplexity = get_gmm_components_perplexity(probe_weights)
                            iteration_record['mcmc_run'] = True
                            iteration_record['mcmc_posterior_available'] = bool(
                                getattr(scorer, 'has_mcmc_posterior', lambda: False)()
                            )
                        elif not run_mcmc_fail and idata_compatible_fail:
                            scorer_fail.model.idata_ = prev_idata_fail
                            if verbose:
                                logging.info(
                                    f"[Iteration {i+1}] reusing MCMC posterior from prior refresh [fail] "
                                    f"(n_feat={prev_mcmc_n_feat_fail}, n_comp={prev_mcmc_n_comp_fail})"
                                )
                        probe_weights_fail = np.asarray(scorer_fail.model.weights_, dtype=float)
                        perplexity_fail = get_gmm_components_perplexity(probe_weights_fail)
                        iteration_record['mcmc_run_fail'] = bool(run_mcmc_fail)
                        iteration_record['mcmc_posterior_available_fail'] = bool(
                            getattr(scorer_fail, 'has_mcmc_posterior', lambda: False)()
                        )
                else:
                    if verbose:
                        logging.info(
                            f"[Iteration {i+1}] stage=model_fitting (GMM"
                            f"{' pass' if dual_gmm else ''})"
                        )
                    _fit_gmm_class(
                        scorer,
                        np.asarray(LIH_rep[current_n_features][current_reference_set_indices]),
                        1,
                    )
                    if dual_gmm:
                        current_n_features_fail = current_n_features
                        if verbose:
                            logging.info(f"[Iteration {i+1}] stage=model_fitting (GMM fail)")
                        _fit_gmm_class(
                            scorer_fail,
                            np.asarray(LIH_rep[current_n_features_fail][current_reference_set_indices]),
                            0,
                        )
                iteration_record['duration_model_fitting'] = time.time() - start_time_model_fitting
                if use_mcmc_gmm:
                    iteration_record['mcmc_warm_start'] = bool(warm_start)
                    fit_profile = getattr(scorer, 'get_fit_profile', lambda: None)()
                    if fit_profile:
                        iteration_record['mcmc_fit_profile'] = _serialize_fit_profile(fit_profile)
                    if dual_gmm:
                        iteration_record['mcmc_warm_start_fail'] = bool(warm_start_fail)
                        fit_profile_fail = getattr(scorer_fail, 'get_fit_profile', lambda: None)()
                        if fit_profile_fail:
                            iteration_record['mcmc_fit_profile_fail'] = _serialize_fit_profile(
                                fit_profile_fail
                            )
                if verbose:
                    logging.info(
                        f"[Iteration {i+1}] stage=model_fitting done "
                        f"({iteration_record['duration_model_fitting']:.1f}s)"
                    )
                    if use_mcmc_gmm and iteration_record.get('mcmc_fit_profile'):
                        logging.info(
                            f"[Iteration {i+1}] MCMCGMM profile: {iteration_record['mcmc_fit_profile']}"
                        )

                if map_score:
                    if not use_mcmc_gmm:
                        logging.warning(
                            f"[Iteration {i+1}] --map_score ignored "
                            f"(scoring_model={scoring_model} has no MCMC posterior)"
                        )
                    else:
                        X_map = np.asarray(
                            LIH_rep[current_n_features][current_reference_set_indices]
                        )
                        fit_pass, _ = _class_fit_yw(1)
                        X_map_pass = X_map[fit_pass == 1]
                        if (
                            getattr(scorer, 'has_mcmc_posterior', lambda: False)()
                            and len(X_map_pass) > 0
                        ):
                            map_info = scorer.set_map_scoring_params(X_map_pass)
                            iteration_record['map_score'] = map_info
                            logging.info(
                                f"[Iteration {i+1}] map_score [pass]: draw "
                                f"{map_info['map_draw_index']}/{map_info['n_draws']} "
                                f"ll={map_info['map_train_ll']:.3f}"
                            )
                        if dual_gmm:
                            X_map_fail_all = np.asarray(
                                LIH_rep[current_n_features_fail][current_reference_set_indices]
                            )
                            fit_fail, _ = _class_fit_yw(0)
                            X_map_fail = X_map_fail_all[fit_fail == 1]
                            if (
                                getattr(scorer_fail, 'has_mcmc_posterior', lambda: False)()
                                and len(X_map_fail) > 0
                            ):
                                map_info_fail = scorer_fail.set_map_scoring_params(X_map_fail)
                                iteration_record['map_score_fail'] = map_info_fail
                                logging.info(
                                    f"[Iteration {i+1}] map_score [fail]: draw "
                                    f"{map_info_fail['map_draw_index']}/{map_info_fail['n_draws']} "
                                    f"ll={map_info_fail['map_train_ll']:.3f}"
                                )

                current_n_components = scorer._n_components()
                scorer_config = replace(scorer_config, n_components=current_n_components)
                if dual_gmm:
                    current_n_components_fail = scorer_fail._n_components()
                    scorer_config_fail = replace(
                        scorer_config_fail, n_components=current_n_components_fail
                    )

                logging.debug(f'[Model Fitting (Iteration: {i+1})]')
                logging.debug(
                    f"Fitted {scoring_model}"
                    f"{' pass' if dual_gmm else ''} with n_features={current_n_features}, "
                    f"n_components={current_n_components}, data_size={len(current_reference_set_indices)}"
                    + (f", run_mcmc={run_mcmc}" if use_mcmc_gmm else "")
                    + (f", warm_start={warm_start}" if use_mcmc_gmm else "")
                )
                if dual_gmm:
                    logging.debug(
                        f"Fitted {scoring_model} fail with n_features={current_n_features_fail}, "
                        f"n_components={current_n_components_fail}, data_size={len(current_reference_set_indices)}"
                        + (f", run_mcmc={run_mcmc_fail}" if use_mcmc_gmm else "")
                        + (f", warm_start={warm_start_fail}" if use_mcmc_gmm else "")
                    )

                assert scorer.check_is_fitted()
                assert scorer._n_features() == current_n_features
                assert scorer._n_components() == current_n_components
                assert probe_weights is not None and perplexity is not None
                if dual_gmm:
                    current_n_features_fail = current_n_features
                    assert scorer_fail.check_is_fitted()
                    assert scorer_fail._n_features() == current_n_features_fail
                    assert current_n_features_fail == current_n_features
                    assert scorer_fail._n_components() == current_n_components_fail
                    assert probe_weights_fail is not None and perplexity_fail is not None

                if verbose:
                    logging.info(f"[Iteration {i+1}] stage=predict_proba_and_uncertainty")
                unc_t0 = time.time()
                X_all_cur = np.asarray(LIH_rep[current_n_features])
                probs = scorer.predict_proba(X_all_cur)
                if use_mcmc_gmm:
                    if getattr(scorer, 'has_mcmc_posterior', lambda: False)():
                        entropies = scorer.get_uncertainties(X_all_cur)
                    else:
                        entropies = np.full(len(X_all_cur), np.nan)
                else:
                    entropies = scorer.get_uncertainties(X_all_cur)
                if verbose:
                    logging.info(
                        f"[Iteration {i+1}] stage=predict_proba_and_uncertainty done "
                        f"({time.time() - unc_t0:.1f}s)"
                    )

                iteration_record['n_features'] = current_n_features
                iteration_record['n_components'] = current_n_components
                iteration_record['reg_covar'] = float(scorer_config.reg_covar)
                iteration_record['d_fail'] = bool(d_fail)
                iteration_record['n_features_cv_class'] = (
                    'fail' if d_fail else ('pass' if dual_gmm else 'single')
                )
                if dual_gmm:
                    iteration_record['reg_covar_fail'] = float(scorer_config_fail.reg_covar)
                iteration_record['probe_weights'] = probe_weights.tolist()
                iteration_record['probe_perplexity'] = float(perplexity)
                if dual_gmm:
                    iteration_record['n_features_fail'] = current_n_features_fail
                    iteration_record['n_components_fail'] = current_n_components_fail
                    iteration_record['probe_weights_fail'] = probe_weights_fail.tolist()
                    iteration_record['probe_perplexity_fail'] = float(perplexity_fail)
                    prior_pass, prior_fail = run_prior_pass, run_prior_fail
                    iteration_record['prior_pass'] = float(prior_pass)
                    iteration_record['prior_fail'] = float(prior_fail)
                if use_mcmc_gmm and getattr(scorer.model, 'bic_', None) is not None:
                    iteration_record['bic'] = float(scorer.model.bic_)
                if use_mcmc_gmm and getattr(scorer.model, 'em_aic_', None) is not None:
                    iteration_record['em_aic'] = float(scorer.model.em_aic_)
                if use_mcmc_gmm and getattr(scorer.model, 'mcmc_aic_', None) is not None:
                    iteration_record['mcmc_aic'] = float(scorer.model.mcmc_aic_)
                if use_mcmc_gmm and getattr(scorer.model, 'mcmc_bic_', None) is not None:
                    iteration_record['mcmc_bic'] = float(scorer.model.mcmc_bic_)
                if dual_gmm and use_mcmc_gmm and getattr(scorer_fail.model, 'bic_', None) is not None:
                    iteration_record['bic_fail'] = float(scorer_fail.model.bic_)
                if dual_gmm and use_mcmc_gmm and getattr(scorer_fail.model, 'em_aic_', None) is not None:
                    iteration_record['em_aic_fail'] = float(scorer_fail.model.em_aic_)
                if dual_gmm and use_mcmc_gmm and getattr(scorer_fail.model, 'mcmc_aic_', None) is not None:
                    iteration_record['mcmc_aic_fail'] = float(scorer_fail.model.mcmc_aic_)
                if dual_gmm and use_mcmc_gmm and getattr(scorer_fail.model, 'mcmc_bic_', None) is not None:
                    iteration_record['mcmc_bic_fail'] = float(scorer_fail.model.mcmc_bic_)

                # RQ-style metrics: score the full target pool; exclude only the initial
                # hand-crafted refs (appended after the target block). AL-selected refs
                # from the target pool remain in the evaluation set.
                eval_mask = np.zeros(len(LIH), dtype=bool)
                eval_mask[:n_target] = True

                start_time_model_scoring = time.time()
                if verbose:
                    logging.info(f"[Iteration {i+1}] stage=scoring_and_uncertainty")

                if dual_gmm:
                    prior_pass, prior_fail = run_prior_pass, run_prior_fail
                    logp_pass_all = scorer.score(np.asarray(LIH_rep[current_n_features]))
                    logp_fail_all = scorer_fail.score(np.asarray(LIH_rep[current_n_features_fail]))
                    pred_scores_all = bayes_pass_posterior(
                        logp_pass_all, logp_fail_all, prior_pass, prior_fail
                    )
                    pred_scores = pred_scores_all[eval_mask]
                else:
                    pred_scores = scorer.score(np.asarray(LIH_rep[current_n_features]))[eval_mask]
                rank_normalized = convert_to_normalized_rank_score(pred_scores)

                iteration_record['duration_model_scoring'] = time.time() - start_time_model_scoring
                if verbose:
                    logging.info(
                        f"[Iteration {i+1}] stage=scoring_and_uncertainty done "
                        f"({iteration_record['duration_model_scoring']:.1f}s)"
                    )

                eval_scores = {
                    'pearson': pearsonr(pred_scores, labels[eval_mask])[0],
                    'spearman': spearmanr(pred_scores, labels[eval_mask])[0],
                    'roc_auc': roc_auc_score(labels_binary[eval_mask], pred_scores)
                }
            except Exception as _gmm_exc:
                if not _is_gmm_collapse_error(_gmm_exc):
                    raise
                gmm_collapse_fallback = True
                skip = True
                scorer = None
                scorer_fail = None
                iteration_record['gmm_collapse_fallback'] = True
                logging.warning(
                    f"[Iteration {i+1}] GMM fit collapsed ({_gmm_exc}). "
                    f"Adding {num_add} random refs and continuing."
                )

        # Reference-set acquisition (random / diversity / BALD / BatchBALD / …).
        acq = acquire_next_reference_indices(
            refset_extension_method=refset_extension_method,
            current_reference_set_indices=current_reference_set_indices,
            LIH_rep=LIH_rep,
            current_n_features=current_n_features,
            current_n_features_fail=current_n_features_fail,
            scorer=scorer,
            scorer_fail=scorer_fail,
            dual_gmm=dual_gmm,
            use_mcmc_gmm=use_mcmc_gmm,
            prior_pass=prior_pass,
            prior_fail=prior_fail,
            num_add=num_add,
            iter_num_add=num_add,  # acquire may raise for --fast warmup
            n_trials=n_trials,
            n_target=n_target,
            gmm_collapse_fallback=gmm_collapse_fallback,
            use_random_startpoint=use_random_startpoint,
            use_random_100_start=use_random_100_start,
            use_random_n0_start=use_random_n0_start,
            n0_gate_sum=n0_gate_sum,
            n0_per_class=n0_per_class,
            n_pass_now=n_pass_now,
            n_fail_now=n_fail_now,
            n0_flag_name=n0_flag_name,
            iteration_i=i,
            skip=skip,
            fast=fast,
            min_class_refs=min_class_refs,
            num_effective_references=num_effective_references,
            num_fail_references=num_fail_references,
            bald_thin_draws=bald_thin_draws,
            LIH=LIH,
            labels=labels,
            dim_reduction_method=dim_reduction_method,
            random_seed=random_seed,
            _ipw_sample_weight=_ipw_sample_weight,
            run_prior_pass=run_prior_pass,
            run_prior_fail=run_prior_fail,
            fit_gmm_class=_fit_gmm_class,
        )
        new_reference_set_indices = acq['new_reference_set_indices']
        new_unc_indices = acq['new_unc_indices']
        new_div_indices = acq['new_div_indices']
        uncertainties = acq['uncertainties']
        aleatoric_uncertainties = acq['aleatoric_uncertainties']
        posterior_disagreement = acq['posterior_disagreement']
        bald_log_marginal = acq['bald_log_marginal']
        iter_num_add = int(acq['iter_num_add'])
        effective_sampling_method = acq['effective_sampling_method']
        iteration_record['num_add'] = iter_num_add
        iteration_record['effective_sampling_method'] = effective_sampling_method
        iteration_record['duration_reference_set_extension'] = acq['duration_reference_set_extension']
        iteration_record['duration_total'] = time.time() - start_time

        bald_decomp_metrics = None
        bald_decomp_extras = None
        if uncertainties is not None and aleatoric_uncertainties is not None:
            cand = np.ones(len(LIH), dtype=bool)
            if len(current_reference_set_indices):
                cand[np.asarray(current_reference_set_indices, dtype=int)] = False
            y_bin = np.asarray(labels, dtype=float) > PASS_THRESH
            bald_decomp_metrics, bald_decomp_extras = compute_bald_pool_diagnostics(
                uncertainties,
                aleatoric_uncertainties,
                candidate_mask=cand,
                y_binary=y_bin,
                selected_idx=new_unc_indices,
                random_state=int(random_seed) * 100003 + int(i + 1),
            )
            hist_row = dict(bald_decomp_metrics)
            hist_row['n_ref'] = int(len(current_reference_set_indices))
            hist_row['iteration'] = int(i + 1)
            bald_diag_history.append(hist_row)
            iteration_record['bald_decomp'] = bald_decomp_metrics
            logging.info(
                f"[Iteration {i+1}] BALD decomp pool: "
                f"CV(MI)={bald_decomp_metrics.get('mi_cv')} "
                f"dyn_range={bald_decomp_metrics.get('mi_dynamic_range')} "
                f"MI/E[H]={bald_decomp_metrics.get('mi_over_aleatoric')} "
                f"pearson(MI,E[H])={bald_decomp_metrics.get('pearson_r')} "
                f"sel_E[H]={bald_decomp_metrics.get('selected_aleatoric_mean')} "
                f"rand_E[H]={bald_decomp_metrics.get('random_aleatoric_mean')}"
            )

        if not skip:
            df_vis = pd.DataFrame(LIH_vis, columns=['Component 1', 'Component 2'])
            df_vis['input_index'] = list(range(len(LIH)))
            df_vis['test_score'] = labels
            if dual_gmm:
                # Store Bayes pass-posterior as the evaluation score; keep class LLs too.
                prior_pass, prior_fail = run_prior_pass, run_prior_fail
                logp_pass_all = scorer.score(np.asarray(LIH_rep[current_n_features]))
                logp_fail_all = scorer_fail.score(np.asarray(LIH_rep[current_n_features_fail]))
                bayes_scores = bayes_pass_posterior(
                    logp_pass_all, logp_fail_all, prior_pass, prior_fail
                )
                df_vis['logprob'] = bayes_scores
                df_vis['logprob_pass'] = logp_pass_all
                df_vis['logprob_fail'] = logp_fail_all
                df_vis['logprob_rank'] = convert_to_normalized_rank_score(bayes_scores)
            else:
                df_vis['logprob'] = scorer.score(LIH_rep[current_n_features])
                df_vis['logprob_rank'] = scorer.rank(LIH_rep[current_n_features], normalize=True)
            newly_added_unc = set(last_added_unc_indices)
            newly_added_div = set(last_added_div_indices)
            newly_added = set(last_added_indices)
            labels_vis = []
            for idx in range(len(LIH)):
                if idx in newly_added_unc and idx in current_reference_set_indices:
                    labels_vis.append('ref_new_unc')
                elif idx in newly_added_div and idx in current_reference_set_indices:
                    labels_vis.append('ref_new_div')
                elif idx in newly_added and idx in current_reference_set_indices:
                    labels_vis.append('ref_new')
                elif idx in current_reference_set_indices:
                    labels_vis.append('ref')
                else:
                    labels_vis.append('test')
            df_vis['label'] = labels_vis
            if uncertainties is not None:
                df_vis['uncertainty'] = uncertainties
            else:
                df_vis['uncertainty'] = np.nan
            if aleatoric_uncertainties is not None:
                df_vis['aleatoric'] = aleatoric_uncertainties
            else:
                df_vis['aleatoric'] = np.nan
            if posterior_disagreement is not None:
                df_vis['p_var'] = posterior_disagreement
            else:
                df_vis['p_var'] = np.nan
            if (
                uncertainties is not None
                and aleatoric_uncertainties is not None
            ):
                df_vis['predictive_entropy'] = (
                    np.asarray(uncertainties, dtype=float)
                    + np.asarray(aleatoric_uncertainties, dtype=float)
                )
            else:
                df_vis['predictive_entropy'] = np.nan
            if bald_log_marginal is not None:
                is_ood, ood_thresh = flag_dual_density_ood(
                    bald_log_marginal, current_reference_set_indices
                )
                df_vis['log_marginal'] = bald_log_marginal
                df_vis['is_ood'] = is_ood
                df_vis['ood_thresh'] = float(ood_thresh)
                n_ood = int(np.sum(is_ood))
                n_ood_cand = int(
                    np.sum(
                        is_ood
                        & ~df_vis['label'].astype(str).str.startswith('ref').to_numpy()
                    )
                )
                logging.info(
                    f"[Iteration {i+1}] dual-density OOD on plot pool: "
                    f"{n_ood} total, {n_ood_cand} non-ref "
                    f"(thresh={ood_thresh:.2f})"
                )
            else:
                df_vis['log_marginal'] = np.nan
                df_vis['is_ood'] = False
                df_vis['ood_thresh'] = np.nan

            n_ref_cur = len(current_reference_set_indices)
            # Default: every 50 refs (+ last iter). --verbose_figure: every iter.
            if verbose_figure:
                should_plot = bool(plot_intermediate_results)
            else:
                should_plot = plot_intermediate_results and (
                    (n_ref_cur % DEFAULT_PLOT_EVERY_POINTS == 0)
                    or (i == n_iterations - 1)
                )

            if should_plot:
                if verbose:
                    is_ref = df_vis['label'].isin(
                        ['ref', 'ref_new', 'ref_new_unc', 'ref_new_div']
                    )
                    n_pass = int(((df_vis['test_score'] > PASS_THRESH) & is_ref).sum())
                    n_fail = int(((df_vis['test_score'] <= PASS_THRESH) & is_ref).sum())
                    n_new_unc = int((df_vis['label'] == 'ref_new_unc').sum())
                    n_new_div = int((df_vis['label'] == 'ref_new_div').sum())
                    n_new_other = int((df_vis['label'] == 'ref_new').sum())
                    logging.info(
                        f"[Iteration {i+1}] stage=plotting "
                        f"(ref_pass={n_pass}, ref_fail={n_fail}, "
                        f"ref_new_unc={n_new_unc}, ref_new_div={n_new_div}, "
                        f"test={int((df_vis['label']=='test').sum())})"
                    )
                plot_dir = (
                    f'{_results_clotho_root(scoring_model)}/'
                    f'LIHS_refset_iter_{num_add}{result_suffix}/'
                    f'{target_task}/layer_{target_layer}_seed_{random_seed}/n_{n_ref_cur}'
                )
                os.makedirs(plot_dir, exist_ok=True)

                fig, axes = plt.subplots(1, 5, figsize=(35, 6), sharex=False, sharey=False)

                plot_probability_density(df_vis, ax=axes[0], cname='test_score')
                plot_logprob_ranks(df_vis, ax=axes[1])
                plot_reference_points_simple(df_vis, ax=axes[2], pass_thresh=PASS_THRESH)
                if uncertainties is not None:
                    plot_uncertainties(
                        df_vis,
                        ax=axes[3],
                        show_ref_new=False,
                        use_rank=False,
                        vmin=run_unc_vmin,
                        vmax=run_unc_vmax,
                    )
                plot_score_distribution(df_vis, ax=axes[4])

                dual_title_extra = ''
                if dual_gmm:
                    dual_title_extra = (
                        f', n_feat_fail={current_n_features_fail}, '
                        f'n_comp_fail={current_n_components_fail}, '
                        f'reg_fail={float(scorer_config_fail.reg_covar):g}, dual'
                    )
                fig.suptitle(
                    f'{target_task}: iteration {i+1} ({refset_extension_method}, {dim_reduction_method}, '
                    f'n_feat={current_n_features}, n_comp={current_n_components}, '
                    f'reg={float(scorer_config.reg_covar):g}{dual_title_extra}) '
                    f'=> pearson={eval_scores["spearman"]:.4f}, roc-auc={eval_scores["roc_auc"]:.4f}',
                    fontsize=12,
                )
                plt.tight_layout()
                save_figure(f'{plot_dir}/{refset_extension_method}_{dim_reduction_method}.jpg')
                plt.close(fig)

                # Companion figure: aleatoric E_t[H[p_t]] when BALD was used.
                if (
                    aleatoric_uncertainties is not None
                    and df_vis['aleatoric'].notna().any()
                ):
                    fig_al, ax_al = plt.subplots(figsize=(10, 6))
                    plot_uncertainties(
                        df_vis,
                        ax=ax_al,
                        title=r'Aleatoric Uncertainty $\mathbb{E}_{\theta}[H[p(y|x,\theta)]]$',
                        show_ref_new=False,
                        use_rank=False,
                        vmin=0.0,
                        vmax=ALEATORIC_UNC_VMAX,
                        value_col='aleatoric',
                        cbar_label='aleatoric',
                    )
                    fig_al.suptitle(
                        f'{target_task}: iteration {i+1} ({refset_extension_method}, '
                        f'{dim_reduction_method}, n_feat={current_n_features}, '
                        f'n_comp={current_n_components}, '
                        f'reg={float(scorer_config.reg_covar):g}{dual_title_extra}) '
                        f'=> pearson={eval_scores["spearman"]:.4f}, '
                        f'roc-auc={eval_scores["roc_auc"]:.4f}',
                        fontsize=12,
                    )
                    plt.tight_layout()
                    save_figure(
                        f'{plot_dir}/{refset_extension_method}_aleatoric_'
                        f'{dim_reduction_method}.jpg'
                    )
                    plt.close(fig_al)

                if bald_decomp_metrics is not None:
                    decomp_path = (
                        f'{plot_dir}/{refset_extension_method}_bald_decomp_'
                        f'{dim_reduction_method}.jpg'
                    )
                    plot_bald_decomposition_diagnostics(
                        df_vis,
                        bald_decomp_metrics,
                        bald_decomp_extras,
                        bald_diag_history,
                        out_path=decomp_path,
                        title=(
                            f'{target_task}: iteration {i+1} ({refset_extension_method}, '
                            f'{dim_reduction_method}, n_feat={current_n_features}, '
                            f'n_comp={current_n_components}, '
                            f'reg={float(scorer_config.reg_covar):g}{dual_title_extra}) '
                            f'=> pearson={eval_scores["spearman"]:.4f}, '
                            f'roc-auc={eval_scores["roc_auc"]:.4f}'
                        ),
                        selected_idx=new_unc_indices,
                    )
                    with open(
                        f'{plot_dir}/{refset_extension_method}_bald_decomp_metrics.json',
                        'w',
                    ) as f_diag:
                        json.dump(
                            {
                                'current': bald_decomp_metrics,
                                'history': bald_diag_history,
                            },
                            f_diag,
                            indent=2,
                        )

                # Prev vs curr Uncertainty of Predictions (every plotted n).
                if use_mcmc_gmm and uncertainties is not None and df_vis['uncertainty'].notna().any():
                    curr_meta = {
                        'task': target_task,
                        'n_ref': n_ref_cur,
                        'iteration': i + 1,
                        'n_feat': current_n_features,
                        'n_comp': current_n_components,
                    }
                    prev_df = None
                    prev_meta = {'n_ref': '—', 'iteration': '—', 'n_feat': '—', 'n_comp': '—'}
                    if prev_uncertainty_plot is not None:
                        prev_df = prev_uncertainty_plot['df_vis']
                        prev_meta = {
                            'n_ref': prev_uncertainty_plot['n_ref'],
                            'iteration': prev_uncertainty_plot['iteration'],
                            'n_feat': prev_uncertainty_plot['n_feat'],
                            'n_comp': prev_uncertainty_plot['n_comp'],
                        }
                    plot_uncertainty_prediction_compare(
                        prev_df,
                        df_vis,
                        prev_meta,
                        curr_meta,
                        out_path=f'{plot_dir}/{refset_extension_method}_prediction.jpg',
                        unc_vmin=run_unc_vmin,
                        unc_vmax=run_unc_vmax,
                    )
                    # Joint class-GMM likelihood + PC1/PC2 marginals in **scoring
                    # feature-PCA**. Overlay weights = predictive likelihood p(x|θ_t)
                    # under MCMC draws; scatter color = mean_t log p(x|θ_t).
                    # Dual mode: one figure per class GMM with _pass / _fail suffixes.
                    if current_n_features >= 2:
                        marg_jobs = [{
                            'suffix': '_pass' if dual_gmm else '',
                            'class_name': 'pass' if dual_gmm else None,
                            'scorer': scorer,
                            'n_feat': current_n_features,
                            'n_comp': current_n_components,
                            'show_ref_pass': True,
                            'show_ref_fail': False,
                        }]
                        if (
                            dual_gmm
                            and scorer_fail is not None
                            and scorer_fail.check_is_fitted()
                        ):
                            marg_jobs.append({
                                'suffix': '_fail',
                                'class_name': 'fail',
                                'scorer': scorer_fail,
                                'n_feat': current_n_features_fail,
                                'n_comp': current_n_components_fail,
                                'show_ref_pass': False,
                                'show_ref_fail': True,
                            })

                        for job in marg_jobs:
                            overlay_w = None
                            scatter_ll = None
                            _sc = job['scorer']
                            _n_feat = job['n_feat']
                            _ensure_feature_rep(LIH, LIH_rep, _n_feat, dim_reduction_method)
                            X_feat = np.asarray(LIH_rep[_n_feat])
                            df_marg = pd.DataFrame(
                                X_feat[:, :2], columns=['Component 1', 'Component 2']
                            )
                            df_marg['uncertainty'] = df_vis['uncertainty'].to_numpy()
                            df_marg['label'] = df_vis['label'].to_numpy()
                            df_marg['test_score'] = df_vis['test_score'].to_numpy()
                            if 'input_index' in df_vis.columns:
                                df_marg['input_index'] = df_vis['input_index'].to_numpy()

                            if use_mcmc_gmm and _sc is not None and _sc.check_is_fitted():
                                try:
                                    logp_draws = np.asarray(
                                        _sc.predictive_log_likelihoods_over_draws(
                                            X_feat, n_draws=100
                                        ),
                                        dtype=float,
                                    )
                                    # Stable per-draw likelihood weights for KDEs.
                                    logp_shift = logp_draws - np.nanmax(
                                        logp_draws, axis=1, keepdims=True
                                    )
                                    overlay_w = np.exp(logp_shift)
                                    scatter_ll = np.nanmean(logp_draws, axis=0)
                                except Exception as ov_exc:
                                    _cls = job['class_name']
                                    _tag = f' [{_cls}]' if _cls else ''
                                    logging.warning(
                                        f"[Iteration {i+1}] marginal likelihood draws skipped"
                                        f"{_tag}: {ov_exc}"
                                    )
                                    overlay_w = None
                                    scatter_ll = None
                                    # Fall back to EM point-estimate log-likelihood.
                                    try:
                                        scatter_ll = np.asarray(
                                            _sc.score(X_feat), dtype=float
                                        )
                                    except Exception:
                                        scatter_ll = None
                            elif _sc is not None and _sc.check_is_fitted():
                                try:
                                    scatter_ll = np.asarray(_sc.score(X_feat), dtype=float)
                                except Exception:
                                    scatter_ll = None

                            class_tag = (
                                f" [{job['class_name']}]" if job['class_name'] else ''
                            )
                            plot_uncertainty_marginal_dist(
                                df_marg,
                                out_path=(
                                    f'{plot_dir}/{refset_extension_method}'
                                    f'_marginal_dist{job["suffix"]}.jpg'
                                ),
                                title=(
                                    f"{target_task}: likelihood marginals{class_tag} "
                                    f"(feature-PCA) "
                                    f"(n={curr_meta['n_ref']}, iter {curr_meta['iteration']}, "
                                    f"n_feat={job['n_feat']}, n_comp={job['n_comp']})"
                                ),
                                n_overlay=100,
                                overlay_weights=overlay_w,
                                scatter_values=scatter_ll,
                                scatter_label=(
                                    'mean log p(x|θ)' if overlay_w is not None
                                    else 'log p(x|θ_EM)'
                                ),
                                show_ref_new=False,
                                show_ref_pass=job['show_ref_pass'],
                                show_ref_fail=job['show_ref_fail'],
                                topk_uncertainty=10,
                                unc_vmin=run_unc_vmin,
                                unc_vmax=run_unc_vmax,
                            )
                    snap = {
                        'df_vis': df_vis.copy(),
                        'n_ref': curr_meta['n_ref'],
                        'iteration': curr_meta['iteration'],
                        'n_feat': curr_meta['n_feat'],
                        'n_comp': curr_meta['n_comp'],
                        'plot_dir': plot_dir,
                        'task': target_task,
                    }
                    uncertainty_plot_history.append(snap)
                    prev_uncertainty_plot = snap

                # Verbose overlay: fitted GMM(s) in feature-PCA scoring space.
                # Dual mode: save separate figures with _pass / _fail suffixes.
                gmm_plot_jobs = []
                if current_n_features >= 2 and scorer is not None and scorer.check_is_fitted():
                    gmm_plot_jobs.append({
                        'suffix': '_pass' if dual_gmm else '',
                        'class_name': 'pass' if dual_gmm else None,
                        'scorer': scorer,
                        'n_features': current_n_features,
                        'n_components': current_n_components,
                        'target_class': 1,
                    })
                if (
                    dual_gmm
                    and current_n_features_fail >= 2
                    and scorer_fail is not None
                    and scorer_fail.check_is_fitted()
                ):
                    gmm_plot_jobs.append({
                        'suffix': '_fail',
                        'class_name': 'fail',
                        'scorer': scorer_fail,
                        'n_features': current_n_features_fail,
                        'n_components': current_n_components_fail,
                        'target_class': 0,
                    })
                for job in gmm_plot_jobs:
                    _save_gmm_fitting_figures(
                        scorer=job['scorer'],
                        LIH_rep=LIH_rep,
                        n_features=job['n_features'],
                        n_components=job['n_components'],
                        current_reference_set_indices=current_reference_set_indices,
                        reference_set_labels=reference_set_labels,
                        target_class=job['target_class'],
                        df_vis=df_vis,
                        plot_dir=plot_dir,
                        refset_extension_method=refset_extension_method,
                        dim_reduction_method=dim_reduction_method,
                        target_task=target_task,
                        iteration=i + 1,
                        random_seed=random_seed,
                        use_mcmc_gmm=use_mcmc_gmm,
                        file_suffix=job['suffix'],
                        class_name=job['class_name'],
                    )

                gc.collect()
                if verbose:
                    logging.info(f"[Iteration {i+1}] stage=plotting done")
            
            df_vis['iteration'] = i + 1
            
            df_all_iterations = pd.concat([df_all_iterations, df_vis], ignore_index=True)
    
        assert len(new_reference_set_indices) == iter_num_add, f"Expected {iter_num_add} new reference set indices, got {len(new_reference_set_indices)}."

        if use_lambda_weight:
            if use_random_startpoint or gmm_collapse_fallback:
                w_new = np.ones(len(new_reference_set_indices), dtype=float)
                lam_scheme = 'random'
            else:
                bald_set = set(int(idx) for idx in new_unc_indices)
                w_new = np.array(
                    [
                        float(lambda_star) if int(idx) in bald_set else 1.0
                        for idx in new_reference_set_indices
                    ],
                    dtype=float,
                )
                lam_scheme = 'bald_lambda' if bald_set else 'ones'
            reference_ipw_weights.extend(float(x) for x in w_new.ravel())
            iteration_record['lambda_star'] = float(lambda_star)
            iteration_record['n0_per_class'] = int(n0_per_class)
            iteration_record['weight_scheme'] = lam_scheme
            iteration_record['new_weights'] = [float(x) for x in w_new.ravel()]
            logging.info(
                f"[Iteration {i+1}] weights ({lam_scheme}): new w "
                f"mean={np.mean(w_new):.3g} min={np.min(w_new):.3g} max={np.max(w_new):.3g} "
                f"(lambda*={lambda_star:.3g}, n0={n0_per_class}, "
                f"n={len(reference_ipw_weights)})"
            )
        elif use_ipw:
            w_new, ipw_scheme = al_ipw_weights_for_new_points(
                n_pool=len(LIH_rep[current_n_features]),
                current_ref_indices=current_reference_set_indices,
                new_indices=new_reference_set_indices,
                new_unc_indices=new_unc_indices,
                new_div_indices=new_div_indices,
                acquisition_scores=uncertainties,
                pool_vectors=LIH_rep[current_n_features],
                force_random=bool(use_random_startpoint or gmm_collapse_fallback),
            )
            reference_ipw_weights.extend(float(x) for x in np.asarray(w_new, dtype=float).ravel())
            iteration_record['ipw_scheme'] = ipw_scheme
            iteration_record['ipw_new_weights'] = [float(x) for x in np.asarray(w_new, dtype=float).ravel()]
            iteration_record['ipw_weight_mean'] = float(np.mean(reference_ipw_weights))
            iteration_record['ipw_weight_min'] = float(np.min(reference_ipw_weights))
            iteration_record['ipw_weight_max'] = float(np.max(reference_ipw_weights))
            logging.info(
                f"[Iteration {i+1}] IPW ({ipw_scheme}): new w "
                f"mean={np.mean(w_new):.3g} min={np.min(w_new):.3g} max={np.max(w_new):.3g} "
                f"(ref mean={np.mean(reference_ipw_weights):.3g}, n={len(reference_ipw_weights)})"
            )
        if success_rate_weight:
            w_pass = _success_rate_weights(labels[current_reference_set_indices], 1)
            iteration_record['success_rate_weight'] = True
            iteration_record['srw_pass_mean'] = float(np.mean(w_pass))
            iteration_record['srw_fail_mean'] = float(np.mean(1.0 - w_pass))
            logging.info(
                f"[Iteration {i+1}] success_rate_weight: pass w "
                f"mean={np.mean(w_pass):.3g} min={np.min(w_pass):.3g} max={np.max(w_pass):.3g}; "
                f"fail w mean={np.mean(1.0 - w_pass):.3g} (n={len(w_pass)})"
            )
        
        iteration_records_all["iteration_" + str(i + 1)] = iteration_record
        current_reference_set_indices.extend(new_reference_set_indices)
        last_added_indices = list(new_reference_set_indices)
        last_added_unc_indices = list(new_unc_indices)
        last_added_div_indices = list(new_div_indices)
        
    df_all_iterations['task'] = target_task
    df_all_iterations['layer'] = target_layer
    df_all_iterations['random_seed'] = random_seed
    df_all_iterations['scoring_model'] = scoring_model
    df_all_iterations['refset_extension_method'] = refset_extension_method
    df_all_iterations['dim_reduction_method'] = dim_reduction_method
    df_all_iterations['dual_gmm'] = bool(dual_gmm)
    df_all_iterations['oracle_prior'] = bool(oracle_prior)
    df_all_iterations['generous'] = bool(generous)
    df_all_iterations['reg_covar'] = float(scorer_config.reg_covar)
    df_all_iterations['reg_decay'] = bool(reg_decay)
    df_all_iterations['map_score'] = bool(map_score)
    df_all_iterations['d_schedule'] = bool(d_schedule)
    df_all_iterations['d_fail'] = bool(d_fail)
    df_all_iterations['fixed_n_features'] = (
        int(n_features) if n_features is not None else np.nan
    )
    df_all_iterations['prior_pass'] = float(run_prior_pass)
    df_all_iterations['prior_fail'] = float(run_prior_fail)
    df_all_iterations['random_start_weighted'] = bool(random_start_weighted)
    df_all_iterations['random_start_double_hard_weighted'] = bool(
        random_start_double_hard_weighted
    )
    df_all_iterations['random_start_sum_weighted'] = bool(random_start_sum_weighted)
    df_all_iterations['random_start_sum_double_hard_weighted'] = bool(
        random_start_sum_double_hard_weighted
    )
    df_all_iterations['success_rate_weight'] = bool(success_rate_weight)
    df_all_iterations['epistemic_multistart'] = bool(epistemic_multistart)
    df_all_iterations['epistemic_em_inits'] = (
        int(epistemic_em_inits) if epistemic_multistart else 0
    )
    if n0_per_class is not None:
        df_all_iterations['n0_per_class'] = int(n0_per_class)
        df_all_iterations['n0_gate'] = 'sum' if n0_gate_sum else 'per_class'
        df_all_iterations['lambda_star'] = float(lambda_star)
        df_all_iterations['n0_difficulty_scale'] = float(n0_difficulty_scale)

    # Re-save prediction compares with a single run-level uncertainty color scale.
    if len(uncertainty_plot_history) >= 2:
        if run_unc_vmin is not None and run_unc_vmax is not None:
            g_vmin, g_vmax = float(run_unc_vmin), float(run_unc_vmax)
        else:
            all_u = np.concatenate([
                np.asarray(h['df_vis']['uncertainty'].to_numpy(dtype=float))
                for h in uncertainty_plot_history
            ])
            all_u = all_u[np.isfinite(all_u)]
            if all_u.size:
                g_vmin = float(np.nanmin(all_u))
                g_vmax = float(np.nanmax(all_u))
                if not np.isfinite(g_vmin) or not np.isfinite(g_vmax) or g_vmax <= g_vmin:
                    g_vmin, g_vmax = 0.0, 1.0
            else:
                g_vmin, g_vmax = 0.0, 1.0

        delta_abs = []
        for k in range(1, len(uncertainty_plot_history)):
            prev_h = uncertainty_plot_history[k - 1]
            curr_h = uncertainty_plot_history[k]
            if 'input_index' not in prev_h['df_vis'].columns or 'input_index' not in curr_h['df_vis'].columns:
                continue
            merged = prev_h['df_vis'][['input_index', 'uncertainty']].merge(
                curr_h['df_vis'][['input_index', 'uncertainty']].rename(
                    columns={'uncertainty': 'uncertainty_curr'}
                ),
                on='input_index',
                how='inner',
            )
            d = (
                merged['uncertainty_curr'].to_numpy(dtype=float)
                - merged['uncertainty'].to_numpy(dtype=float)
            )
            d = d[np.isfinite(d)]
            if d.size:
                delta_abs.append(np.max(np.abs(d)))
        g_delta_vmax = float(max(delta_abs)) if delta_abs else None
        if g_delta_vmax is not None and (not np.isfinite(g_delta_vmax) or g_delta_vmax <= 0):
            g_delta_vmax = None

        if verbose:
            logging.info(
                f"[simulate_iterations] rescaling {len(uncertainty_plot_history)-1} "
                f"prediction figures with unc=[{g_vmin:.4g}, {g_vmax:.4g}], "
                f"delta_vmax={g_delta_vmax}"
            )
        for k in range(1, len(uncertainty_plot_history)):
            prev_h = uncertainty_plot_history[k - 1]
            curr_h = uncertainty_plot_history[k]
            prev_meta = {
                'n_ref': prev_h['n_ref'],
                'iteration': prev_h['iteration'],
                'n_feat': prev_h['n_feat'],
                'n_comp': prev_h['n_comp'],
            }
            curr_meta = {
                'task': curr_h.get('task', target_task),
                'n_ref': curr_h['n_ref'],
                'iteration': curr_h['iteration'],
                'n_feat': curr_h['n_feat'],
                'n_comp': curr_h['n_comp'],
            }
            plot_uncertainty_prediction_compare(
                prev_h['df_vis'],
                curr_h['df_vis'],
                prev_meta,
                curr_meta,
                out_path=(
                    f"{curr_h['plot_dir']}/{refset_extension_method}_prediction.jpg"
                ),
                unc_vmin=g_vmin,
                unc_vmax=g_vmax,
                delta_vmax=g_delta_vmax,
            )

    return df_all_iterations, iteration_records_all



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Online Adequacy Modelling Experiment w/ Probability Density Model")
    parser.add_argument(
        '--target_task',
        '--task',
        dest='target_task',
        type=str,
        nargs='+',
        required=True,
        metavar='TASK',
        help=(
            'One or more target tasks (space-separated). '
            f'`all` expands to every Clotho task: {", ".join(CLOTHO_TASKS)}. '
            'Example: --target_task topic_classification adding_odd_numbers '
            'or --task all'
        ),
    )
    parser.add_argument('--scoring_model', type=str, help='Scoring model for the experiment', default='GMM', choices=['GMM', 'MCMCGMM', 'ADVIGMM'])
    parser.add_argument(
        '--refset_extension_methods',
        type=str,
        nargs='+',
        default=['random', 'diversity'],
        help=(
            'Methods for reference set extension: random, diversity, '
            'uncertainty, balanced (half BALD/uncertainty + half diversity), '
            'bald, batchbald, rankvar (Var_t of the target-pool rank of raw η, '
            'average ties, same ranking as spearmanr), '
            'epig (pairwise-order EPIG on the k/n label grid; n is --n_trials), '
            'llvar (half Var_t[p(x|θ)] + half diversity), '
            'epigbalanced (half epig + half diversity), '
            'epigspearmanbalanced (half Spearman rank-variance EPIG + half '
            'diversity; ranks raw η, binomial label still uses --n_trials), '
            'rankvarbalanced (half rankvar + half diversity).'
        ),
    )
    parser.add_argument(
        '--n_trials',
        type=int,
        default=DEFAULT_N_TRIALS,
        help=(
            'Label-grid size for pairwise epig: η is snapped to '
            '{0, 1/n, ..., 1} before ranking. rankvar ignores this and ranks '
            'raw η. epigspearmanbalanced uses the same n as the binomial label '
            'but also ranks raw η. Default 10. A value other than 10 adds a '
            '_ntrials<n> suffix to the result path.'
        ),
    )
    parser.add_argument(
        '--random_100_start',
        action='store_true',
        help=(
            'Shared random startpoint: acquire the first '
            f'{RANDOM_START_N_REFS} reference points with random sampling, '
            'then switch to --refset_extension_methods for the remaining adds '
            f'(typically 400, n={RANDOM_START_N_REFS}→500). MCMC/BALD is deferred '
            f'until n_ref>={RANDOM_START_N_REFS}. Result paths get a _r100start suffix.'
        ),
    )
    parser.add_argument(
        '--random_100_start_weighted',
        action='store_true',
        help=(
            'Same acquisition as --random_100_start (random until '
            f'{RANDOM_START_N_REFS} refs, then --refset_extension_methods), '
            'but GMM/MCMC fits use inverse-propensity weights: random-phase '
            'points get w=N/k (SRS), later uncertainty/BALD points get w=1/π '
            'with π from a softmax approximation of top-k inclusion. '
            'Result paths get a _r100start_ipw suffix.'
        ),
    )
    parser.add_argument(
        '--random_start_weighted',
        action='store_true',
        help=(
            'Random acquisition until both pass and fail refs exceed the '
            'PCA-5 conservative n0 (from R* upper; n0=p(1+1/u)), then switch '
            'to --refset_extension_methods and fit immediately (the '
            f'{MIN_CLASS_REFS_TO_FIT}/class fit gate is skipped). '
            'BALD/uncertainty points get EM sample_weight=lambda* '
            '(Eq 14-15, alpha=500-n0); random points stay at weight 1. '
            'Result paths get a _rstart_lam suffix.'
        ),
    )
    parser.add_argument(
        '--random_start_double_hard_weighted',
        action='store_true',
        help=(
            'Same as --random_start_weighted but treats the problem as twice '
            'as hard: u_eff=u/2 in both n0 and lambda*. Result paths get a '
            '_rstart_lam_2x suffix.'
        ),
    )
    parser.add_argument(
        '--random_start_sum_weighted',
        action='store_true',
        help=(
            'Same n0/lambda* as --random_start_weighted, but random only until '
            'n_pass+n_fail > n0 (pooled count, not both classes). Result paths '
            'get a _rstart_lam_sum suffix.'
        ),
    )
    parser.add_argument(
        '--random_start_sum_double_hard_weighted',
        action='store_true',
        help=(
            'Pooled n0 gate (n_pass+n_fail > n0) with difficulty_scale=2. '
            'Result paths get a _rstart_lam_2x_sum suffix.'
        ),
    )
    parser.add_argument(
        '--success_rate_weight',
        action='store_true',
        help=(
            'Use repeated-eval success rate k/10 as GMM EM/MCMC sample_weight: '
            'pass density gets w=k/10, fail density gets w=1-k/10, and every '
            'reference is kept (no hard 0/1 split). Incompatible with lambda* '
            '(--random_start_*_weighted) and --random_100_start_weighted. '
            'Result paths get an _srw suffix.'
        ),
    )
    parser.add_argument(
        '--epistemic_multistart',
        action='store_true',
        help=(
            'Estimate BALD/epistemic from N uncentered NUTS chains, each started '
            'from a loose sklearn EM (n_init=3, distinct seeds). Scoring still uses '
            'the run\'s rigorous EM. Default N=4 (see --epistemic_em_inits). '
            'MCMCGMM only. Result paths get an _epimulti suffix.'
        ),
    )
    parser.add_argument(
        '--jax_platform',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'gpu'],
        help=(
            'JAX device for MCMCGMM/ADVIGMM NUTS/ADVI (default cpu). '
            'Use cuda|gpu for BlackJAX on GPU; EM/sklearn stays on CPU. '
            'GPU uses chain_method=vectorized. Does not change result path tags.'
        ),
    )
    parser.add_argument(
        '--epistemic_em_inits',
        type=int,
        default=4,
        metavar='N',
        help=(
            'Number of loose EM starts / NUTS chains with --epistemic_multistart '
            '(default 4). Each start uses n_init=3 vs the scoring EM n_init=30.'
        ),
    )
    parser.add_argument(
        '--max_n_ref',
        type=int,
        default=None,
        metavar='N',
        help=(
            'Stop the AL loop when the reference set size reaches N '
            '(must be n_init + k * num_add). Default is 50 iterations '
            f'(typically n_init=10, num_add={NUM_ADD} => n=510). '
            'Result paths get an _n<N> suffix.'
        ),
    )
    parser.add_argument('--target_llm', type=str, default='llama', help='Target LLM for the experiment')
    parser.add_argument('--target_layer', type=int, default=None, help='Target layer for the experiment (If not specified, use default mapping)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0], help='Random seeds for the experiment')
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Log fine-grained stage progress (EM, model build, MCMC/XLA compile+tune+draws, scoring, plotting) to console and log file',
    )
    parser.add_argument(
        '--verbose_figure',
        action='store_true',
        help=(
            'Save intermediate figures every iteration. Default is every '
            f'{DEFAULT_PLOT_EVERY_POINTS} reference points (plus the last iter).'
        ),
    )
    parser.add_argument(
        '--fast',
        action='store_true',
        help=(
            'Cheaper MCMCGMM/ADVIGMM run: refresh the posterior every 20 iterations, '
            'halve NUTS tune/draws '
            f'(1000→500), thin BALD posterior draws every {FAST_THIN_STRIDE}, and '
            f'add {FAST_WARMUP_ADD} refs per skip-iter until each class has '
            f'{MIN_CLASS_REFS_TO_FIT} points. Result paths get a _fast suffix. '
            'Figure cadence is unchanged (see --verbose_figure).'
        ),
    )
    parser.add_argument(
        '--dual_gmm',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'Fit separate pass and fail GMMs (default: on). PCA n_features is shared '
            '(same feature space for Bayes); by default pass-side CV chooses D and fail '
            'follows. Pass --d_fail to choose that shared D with fail-side CV instead. '
            'n_components stay independent per class. Point scores are Bayes posteriors '
            'p(y=pass|x) from the two class-conditional likelihoods '
            '(Pearson / Spearman / ROC-AUC). '
            'GMM fitting figures are saved with _pass / _fail suffixes. '
            'Use --no-dual_gmm for single pass-only GMM.'
        ),
    )
    parser.add_argument(
        '--oracle_prior',
        action='store_true',
        help=(
            'Use ground-truth pass/fail rates over the full labeled pool as class priors '
            'for dual Bayes / BALD / BatchBALD instead of equal priors [0.5, 0.5]. '
            'Only affects --dual_gmm runs; results get an _oracle_prior suffix.'
        ),
    )
    parser.add_argument(
        '--generous',
        action='store_true',
        help=(
            'Use the older (pre-tightening) MCMCGMM EM hyperparameters for both '
            'GMM and MCMCGMM EM init: '
            f"n_init={GENEROUS_EM['n_init']} (default {DEFAULT_EM['n_init']}), "
            f"max_iter={GENEROUS_EM['max_iter']} (default {DEFAULT_EM['max_iter']}), "
            f"tol={GENEROUS_EM['tol']} (default {DEFAULT_EM['tol']}), "
            f"reg_covar={GENEROUS_EM['reg_covar']} (default {DEFAULT_EM['reg_covar']}). "
            'Results get a _generous suffix.'
        ),
    )
    parser.add_argument(
        '--d',
        dest='n_features',
        type=int,
        default=None,
        metavar='D',
        help=(
            'Freeze PCA dimensionality at D for the whole run and skip the '
            'per-iteration feature CV (±step search). Dual pass/fail share this D. '
            'MCMC no longer lowers D on sampling failure. '
            'Result paths get a _d<D> suffix (e.g. ..._d15). '
            f'Default is adaptive CV starting at {INIT_N_FEATURES} '
            f'(or {D_SCHEDULE_INIT_N_FEATURES} with --d_schedule).'
        ),
    )
    parser.add_argument(
        '--d_schedule',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'Monotone PCA-dim schedule (default: on): start at '
            f'{D_SCHEDULE_INIT_N_FEATURES} and only step up by {D_SCHEDULE_FEATURE_STEP} '
            '(never down). HD is accepted only with the old LD gate '
            '(higher mean Pearson and min fold > max(current min fold, 0.1)). '
            'MCMC no longer lowers D on sampling failure. '
            'Compatible with --d_fail (same up-only schedule, fail-GMM CV). '
            'Incompatible with --d. Result paths get a _dsched suffix. '
            'Use --no-d_schedule to disable.'
        ),
    )
    parser.add_argument(
        '--d_fail',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'Dual only (default: on): choose the shared PCA dim with fail-GMM feature CV '
            'instead of pass-GMM CV (pass still uses that same D). '
            'Combines with --d_schedule (monotone up, HD gate on fail CV). '
            'Incompatible with --d. Requires --dual_gmm '
            '(or dual configs under --all_poss_comb). '
            'Result paths get a _dfail suffix. '
            'Use --no-d_fail to disable.'
        ),
    )
    parser.add_argument(
        '--reg_covar',
        type=float,
        default=None,
        help=(
            'Freeze EM covariance regularization (sklearn GaussianMixture '
            'reg_covar) and tag the result path (e.g. ..._reg1e-6). '
            'If omitted: original Clotho — start at the EM preset '
            f"(rigorous={DEFAULT_EM['reg_covar']}, "
            f"generous={GENEROUS_EM['reg_covar']}) and raise λ ×10 until "
            'predictive entropy has variance; path stays untagged. '
            'With --reg_decay this is only the starting λ (default 1e-1) '
            'and the path uses _regdecay instead of _reg<value>.'
        ),
    )
    parser.add_argument(
        '--reg_decay',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'Monotone reg_covar schedule (default: on): start at 1e-1 (or --reg_covar) and each '
            'iteration compare {λ, λ/10}. Dual GMM uses Bayes p(y=pass|x) vs labels '
            '(same functional as eval, not pass-density ranks). Single GMM still '
            'uses rank-scores. Decay only if mean Pearson is higher and min fold '
            f'> max(current min fold, {CV_MIN_FOLD_FLOOR}) — same gate as PCA-dim '
            'LD / --d_schedule HD. Never increases. Dual fail copies the pass '
            'value. Result paths get a _regdecay suffix. '
            'Use --no-reg_decay to disable.'
        ),
    )
    parser.add_argument(
        '--map_score',
        action='store_true',
        help=(
            'Score Pearson / Spearman / ROC-AUC and plotted logprob from the '
            'single MCMC draw with highest training likelihood (all posterior '
            'samples, typically 1000). BALD / BatchBALD still use the full '
            'posterior. No-op for --scoring_model GMM. Result paths get a '
            '_mapscore suffix.'
        ),
    )
    cov_group = parser.add_mutually_exclusive_group()
    cov_group.add_argument(
        '--diag',
        action='store_true',
        help='Use diagonal covariance GMM (default). Result path uses _diag. MCMCGMM/ADVIGMM are always diag.',
    )
    cov_group.add_argument(
        '--full',
        action='store_true',
        help='Use full covariance GMM. Result path uses _full. Only applies to --scoring_model GMM.',
    )
    parser.add_argument(
        '--all_poss_comb',
        action='store_true',
        help=(
            'Sweep the 8 valid EM/GMM ablation configs from the RQ1-1 inventory '
            '(diag×{¬dual,dual}×{generous,rigorous} with oracle only on dual; '
            'plus full×{¬dual,dual} under rigorous¬oracle). '
            'Forces --scoring_model GMM and ignores --dual_gmm / --oracle_prior / '
            '--generous / --diag / --full for the sweep.'
        ),
    )

    args = parser.parse_args()
    if args.jax_platform in ('gpu', 'cuda'):
        os.environ['JAX_PLATFORMS'] = 'cuda'
        os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    elif args.jax_platform == 'cpu':
        os.environ['JAX_PLATFORMS'] = 'cpu'

    n0_start_flags = [
        args.random_start_weighted,
        args.random_start_double_hard_weighted,
        args.random_start_sum_weighted,
        args.random_start_sum_double_hard_weighted,
    ]
    if sum(bool(x) for x in n0_start_flags) > 1:
        raise SystemExit(
            'Use only one of --random_start_weighted, '
            '--random_start_double_hard_weighted, '
            '--random_start_sum_weighted, '
            '--random_start_sum_double_hard_weighted.'
        )
    if args.success_rate_weight and any(bool(x) for x in n0_start_flags):
        raise SystemExit(
            '--success_rate_weight cannot be combined with lambda* flags '
            '(--random_start_weighted / --random_start_double_hard_weighted / '
            '--random_start_sum_weighted / --random_start_sum_double_hard_weighted).'
        )
    if args.success_rate_weight and args.random_100_start_weighted:
        raise SystemExit(
            '--success_rate_weight cannot be combined with --random_100_start_weighted.'
        )
    if args.n_features is not None and int(args.n_features) < 1:
        raise SystemExit(f'--d must be >= 1, got {args.n_features}')
    if args.d_schedule and args.n_features is not None:
        raise SystemExit('--d_schedule cannot be combined with --d')
    if args.d_fail and args.n_features is not None:
        raise SystemExit('--d_fail cannot be combined with --d (no feature CV to retarget)')
    if args.d_fail and not args.dual_gmm and not args.all_poss_comb:
        raise SystemExit('--d_fail requires --dual_gmm')
    if int(args.epistemic_em_inits) < 1:
        raise SystemExit(f'--epistemic_em_inits must be >= 1, got {args.epistemic_em_inits}')
    if args.max_n_ref is not None and int(args.max_n_ref) < 1:
        raise SystemExit(f'--max_n_ref must be >= 1, got {args.max_n_ref}')
    if args.epistemic_multistart and args.scoring_model != 'MCMCGMM':
        raise SystemExit('--epistemic_multistart requires --scoring_model MCMCGMM')
    if args.all_poss_comb:
        if args.scoring_model != 'GMM':
            raise SystemExit(
                f'--all_poss_comb only supports EM/GMM (got --scoring_model {args.scoring_model}).'
            )
        run_configs = [
            {**cfg, 'scoring_model': 'GMM', 'reg_covar': args.reg_covar}
            for cfg in ALL_POSS_COMB
        ]
    else:
        covariance_type = 'full' if args.full else 'diag'
        if args.scoring_model in BAYES_GMM_MODELS and args.full:
            raise SystemExit(
                f'--full is only supported for --scoring_model GMM '
                f'({args.scoring_model} is always diag).'
            )
        run_configs = [
            dict(
                scoring_model=args.scoring_model,
                covariance_type=covariance_type,
                dual_gmm=args.dual_gmm,
                oracle_prior=args.oracle_prior,
                generous=args.generous,
                reg_covar=args.reg_covar,
            )
        ]

    target_tasks = []
    seen_tasks = set()
    unknown_tasks = []
    for raw in args.target_task:
        name = str(raw).strip()
        if not name:
            continue
        if name.lower() == 'all':
            for t in CLOTHO_TASKS:
                if t not in seen_tasks:
                    seen_tasks.add(t)
                    target_tasks.append(t)
            continue
        if name not in prompt_templates:
            unknown_tasks.append(name)
            continue
        if name not in seen_tasks:
            seen_tasks.add(name)
            target_tasks.append(name)
    if unknown_tasks:
        raise SystemExit(
            f"Unknown task(s): {unknown_tasks}. "
            f"Use one of {list(CLOTHO_TASKS)} or `all`."
        )
    if not target_tasks:
        raise SystemExit(
            f"--target_task / --task is empty after expansion. "
            f"Use one of {list(CLOTHO_TASKS)} or `all`."
        )
    
    target_layer_map = {
        'llama': 21,
        'gemma': 28,
        'mistral': 22
    }

    result_suffix = f"_{args.target_llm}"
    target_layer = args.target_layer if args.target_layer is not None else target_layer_map[args.target_llm]

    refset_extension_methods_str = '_'.join(args.refset_extension_methods)
    tasks_log_str = '+'.join(target_tasks)
    comb_log_tag = 'all_poss_comb' if args.all_poss_comb else 'single'
    os.makedirs(
        f"logs_clotho/{target_layer}{result_suffix}/{refset_extension_methods_str}/{tasks_log_str}",
        exist_ok=True,
    )
    log_filename = (
        f"logs_clotho/{target_layer}{result_suffix}/{refset_extension_methods_str}/"
        f"{tasks_log_str}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{comb_log_tag}.log"
    )

    logging.basicConfig(
        filename=log_filename,
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    if args.verbose:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(console)
        logging.info(f"Verbose stage logging enabled; also writing to {log_filename}")
    # Root logger is DEBUG for our own logs; keep third-party samplers quiet.
    for noisy_logger in ('jax', 'jaxlib', 'numba', 'matplotlib', 'pytensor', 'filelock'):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    logging.info(
        f"Running {len(target_tasks)} task(s): {target_tasks} "
        f"× {len(run_configs)} config(s) "
        f"(methods={args.refset_extension_methods}, num_add={NUM_ADD}, seeds={args.seeds}, "
        f"n_features={'CV' if args.n_features is None else args.n_features}, "
        f"d_schedule={args.d_schedule}, d_fail={args.d_fail}, "
        f"reg_decay={args.reg_decay}, fast={args.fast}, "
        f"random_100_start={args.random_100_start}, "
        f"random_100_start_weighted={args.random_100_start_weighted}, "
        f"random_start_weighted={args.random_start_weighted}, "
        f"random_start_double_hard_weighted={args.random_start_double_hard_weighted}, "
        f"random_start_sum_weighted={args.random_start_sum_weighted}, "
        f"random_start_sum_double_hard_weighted={args.random_start_sum_double_hard_weighted}, "
        f"success_rate_weight={args.success_rate_weight}, "
        f"epistemic_multistart={args.epistemic_multistart}, "
        f"epistemic_em_inits={args.epistemic_em_inits}, "
        f"max_n_ref={args.max_n_ref}, jax_platform={args.jax_platform}, "
        f"verbose_figure={args.verbose_figure}, map_score={args.map_score}, "
        f"all_poss_comb={args.all_poss_comb})"
    )
    # Outer task, inner config: finish all ablations for one task before the next.
    for task_i, target_task in enumerate(target_tasks, start=1):
        logging.info(f"=== task {task_i}/{len(target_tasks)}: {target_task} ===")
        for cfg_i, cfg in enumerate(run_configs, start=1):
            em_param = 'generous' if cfg['generous'] else 'rigorous'
            logging.info(
                f"=== config {cfg_i}/{len(run_configs)}: scoring_model={cfg['scoring_model']}, "
                f"covariance={cfg['covariance_type']}, dual={cfg['dual_gmm']}, "
                f"em_param={em_param}, oracle_prior={cfg['oracle_prior']}, "
                f"reg_covar={cfg.get('reg_covar')}, reg_decay={args.reg_decay} ==="
            )
            main(
                target_task=target_task,
                target_layer=target_layer,
                scoring_model=cfg['scoring_model'],
                refset_extension_methods=args.refset_extension_methods,
                random_seeds=args.seeds,
                num_add=NUM_ADD,
                target_llm=args.target_llm,
                verbose=args.verbose,
                covariance_type=cfg['covariance_type'],
                dual_gmm=cfg['dual_gmm'],
                oracle_prior=cfg['oracle_prior'],
                generous=cfg['generous'],
                reg_covar=cfg.get('reg_covar'),
                n_features=args.n_features,
                fast=args.fast,
                verbose_figure=args.verbose_figure,
                map_score=args.map_score,
                reg_decay=args.reg_decay,
                                d_schedule=args.d_schedule,
                d_fail=args.d_fail,
                random_100_start=args.random_100_start,
                random_100_start_weighted=args.random_100_start_weighted,
                random_start_weighted=args.random_start_weighted,
                random_start_double_hard_weighted=args.random_start_double_hard_weighted,
                random_start_sum_weighted=args.random_start_sum_weighted,
                random_start_sum_double_hard_weighted=args.random_start_sum_double_hard_weighted,
                success_rate_weight=args.success_rate_weight,
                epistemic_multistart=args.epistemic_multistart,
                epistemic_em_inits=args.epistemic_em_inits,
                max_n_ref=args.max_n_ref,
                n_trials=args.n_trials,
                jax_platform=args.jax_platform,
            )