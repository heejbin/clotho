"""GMM / MCMC fitting figure helpers for Clotho iteration runs."""
from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from plot_utils import (
    plot_gmm_components_on_feature_pca,
    plot_gmm_posterior_draws_on_feature_pca,
    plot_mcmc_gmm_parameter_distributions,
    plot_probability_density,
    save_figure,
)
from clotho_run_helpers import _diag_vars_to_covmats

def _save_gmm_fitting_figures(
    *,
    scorer,
    LIH_rep,
    n_features,
    n_components,
    current_reference_set_indices,
    reference_set_labels,
    target_class,
    df_vis,
    plot_dir,
    refset_extension_method,
    dim_reduction_method,
    target_task,
    iteration,
    random_seed,
    use_mcmc_gmm,
    file_suffix='',
    class_name=None,
):
    """Save verbose / EM-init / MCMC-parameter GMM overlay figures for one class GMM.

    ``file_suffix`` is appended before ``.jpg`` (e.g. ``_pass`` / ``_fail`` in dual mode).
    Single-GMM mode keeps ``file_suffix=''`` for backward-compatible filenames.
    """
    X_all = np.asarray(LIH_rep[n_features])
    class_mask = (np.asarray(reference_set_labels) == target_class)
    X_ref_eff = X_all[current_reference_set_indices][class_mask]
    if len(X_ref_eff) == 0:
        logging.warning(
            f"[Iteration {iteration}] skip GMM fitting figures"
            f"{f' ({class_name})' if class_name else ''}: no effective references"
        )
        return

    means_all = covs_all = weights_all = None
    if use_mcmc_gmm and getattr(scorer.model, 'idata_', None) is not None:
        means_all, covs_all, weights_all = scorer.model._flattened_posterior_params()
        means_model = np.asarray(scorer.model.means_, dtype=float)
        covs_model = np.asarray(scorer.model.covariances_, dtype=float)
        weights_viz = np.asarray(scorer.model.weights_, dtype=float)
        weights_viz = weights_viz / weights_viz.sum()
        gmm_overlay_label = 'EM'
        means_viz, covs_viz = scorer.inverse_transform_gmm_params(means_model, covs_model)
        if covs_viz.ndim == 2:
            covs_viz = _diag_vars_to_covmats(covs_viz)
        if means_all is not None:
            means_all, covs_all = scorer.inverse_transform_gmm_params(means_all, covs_all)
    elif use_mcmc_gmm:
        means_model = np.asarray(scorer.model.means_, dtype=float)
        covs_model = np.asarray(scorer.model.covariances_, dtype=float)
        weights_viz = np.asarray(scorer.model.weights_, dtype=float)
        weights_viz = weights_viz / weights_viz.sum()
        gmm_overlay_label = 'EM'
        means_viz, covs_viz = scorer.inverse_transform_gmm_params(means_model, covs_model)
        if covs_viz.ndim == 2:
            covs_viz = _diag_vars_to_covmats(covs_viz)
    else:
        means_viz, covs_viz, weights_viz = get_highest_likelihood_gmm_params(scorer, X_ref_eff)
        means_model, covs_model = means_viz, covs_viz
        gmm_overlay_label = 'MAP/MLE'

    df_feat2d = pd.DataFrame(X_all[:, :2], columns=['Component 1', 'Component 2'])
    df_feat2d['label'] = df_vis['label'].to_numpy()
    df_feat2d['test_score'] = df_vis['test_score'].to_numpy()

    class_tag = f' [{class_name}]' if class_name else ''
    mcmc_ic_suffix = ''
    if use_mcmc_gmm:
        mcmc_aic = getattr(scorer.model, 'mcmc_aic_', None)
        mcmc_bic = getattr(scorer.model, 'mcmc_bic_', None)
        if mcmc_aic is None or mcmc_bic is None:
            mcmc_aic, mcmc_bic = diag_gmm_aic_bic(
                X_ref_eff, means_model, covs_model, weights_viz
            )
        mcmc_ic_suffix = f' | MCMC AIC={mcmc_aic:.1f}, BIC={mcmc_bic:.1f}'

    fig_v, axes_v = plt.subplots(1, 2, figsize=(16, 6), sharex=False, sharey=False)
    plot_gmm_components_on_feature_pca(
        df_feat2d,
        means_viz,
        covs_viz,
        weights=weights_viz,
        ax=axes_v[0],
        n_std=1.0,
        title=f'Ref/Test + GMM {gmm_overlay_label}{class_tag} (centers + 1σ)',
    )
    if use_mcmc_gmm and means_all is not None:
        plot_probability_density(df_feat2d, ax=axes_v[1], cname='test_score')
        means_draw, covs_draw, weights_draw = scorer.model.sample_posterior_params(n_draws=100)
        means_draw, covs_draw = scorer.inverse_transform_gmm_params(means_draw, covs_draw)
        if covs_draw.ndim == 3:
            covs_draw = _diag_vars_to_covmats(covs_draw)
        try:
            means_draw, covs_draw, weights_draw = scorer.model.relabel_params_to_reference(
                means_draw,
                covs_draw,
                weights_draw,
                reference=means_viz,
            )
        except Exception as relabel_exc:
            logging.warning(
                f"[Iteration {iteration}] MCMC draw relabel skipped{class_tag}: {relabel_exc}"
            )
        plot_gmm_posterior_draws_on_feature_pca(
            df_feat2d,
            means_draw,
            covs_draw,
            weights_draws=weights_draw,
            ax=axes_v[1],
            n_std=1.0,
            max_draws=100,
            random_state=random_seed,
            scatter_points=False,
            title=f'Test score + MCMC draws{class_tag} (n=100, 1σ)',
        )
    else:
        plot_probability_density(df_feat2d, ax=axes_v[1], cname='test_score')
        plot_gmm_components_on_feature_pca(
            df_feat2d,
            means_viz,
            covs_viz,
            weights=weights_viz,
            ax=axes_v[1],
            n_std=1.0,
            title=f'Test score + GMM {gmm_overlay_label}{class_tag} (centers + 1σ)',
        )

    fig_v.suptitle(
        f'{target_task}: iteration {iteration} verbose{class_tag} ({refset_extension_method}, '
        f'n_feat={n_features}, n_comp={n_components})'
        f'{mcmc_ic_suffix}',
        fontsize=12,
    )
    plt.tight_layout()
    save_figure(
        f'{plot_dir}/{refset_extension_method}_{dim_reduction_method}_verbose{file_suffix}.jpg'
    )
    plt.close(fig_v)

    em_init = getattr(scorer.model, 'em_init_', None) if use_mcmc_gmm else None
    if em_init is not None:
        means_em = np.asarray(em_init.means_, dtype=float)
        covs_em = np.asarray(em_init.covariances_, dtype=float)
        weights_em = np.asarray(em_init.weights_, dtype=float)
        weights_em = weights_em / weights_em.sum()
        means_em, covs_em = scorer.inverse_transform_gmm_params(means_em, covs_em)
        if covs_em.ndim == 2:
            covs_em = _diag_vars_to_covmats(covs_em)
        em_bic = float(
            getattr(scorer.model, 'bic_', None)
            if getattr(scorer.model, 'bic_', None) is not None
            else em_init.bic(scorer.transform_features(X_ref_eff))
        )
        em_aic = float(
            getattr(scorer.model, 'em_aic_', None)
            if getattr(scorer.model, 'em_aic_', None) is not None
            else em_init.aic(scorer.transform_features(X_ref_eff))
        )

        fig_em, axes_em = plt.subplots(1, 2, figsize=(16, 6), sharex=False, sharey=False)
        plot_gmm_components_on_feature_pca(
            df_feat2d,
            means_em,
            covs_em,
            weights=weights_em,
            ax=axes_em[0],
            n_std=1.0,
            title=f'Ref/Test + GMM EM init{class_tag} (centers + 1σ)',
        )
        plot_probability_density(df_feat2d, ax=axes_em[1], cname='test_score')
        plot_gmm_components_on_feature_pca(
            df_feat2d,
            means_em,
            covs_em,
            weights=weights_em,
            ax=axes_em[1],
            n_std=1.0,
            title=f'Test score + GMM EM init{class_tag} (centers + 1σ)',
        )
        fig_em.suptitle(
            f'{target_task}: iteration {iteration} EM init{class_tag} '
            f'({refset_extension_method}, n_feat={n_features}, '
            f'n_comp={means_em.shape[0]}) | EM AIC={em_aic:.1f}, BIC={em_bic:.1f}',
            fontsize=12,
        )
        plt.tight_layout()
        save_figure(
            f'{plot_dir}/{refset_extension_method}_{dim_reduction_method}_em_init{file_suffix}.jpg'
        )
        plt.close(fig_em)

    if means_all is not None:
        fig_p, _ = plot_mcmc_gmm_parameter_distributions(
            means_all,
            covs_all,
            weights_all,
            max_draws=300,
            random_state=random_seed,
            title=(
                f'{target_task}: iteration {iteration} MCMC params{class_tag} '
                f'({refset_extension_method}, n_feat={n_features}, '
                f'n_comp={n_components})'
            ),
        )
        save_figure(
            f'{plot_dir}/{refset_extension_method}_{dim_reduction_method}_parameter{file_suffix}.jpg'
        )
        plt.close(fig_p)


