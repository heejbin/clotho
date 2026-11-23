import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from scipy.stats import pearsonr, rankdata, spearmanr

from matplotlib import colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm
from matplotlib.patches import Ellipse
from adjustText import adjust_text

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np


cmap = LinearSegmentedColormap.from_list("red_green", ["red", "green"], N=11)

# Absolute BALD / entropy-style uncertainty: fixed [0, vmax] scale
# (run script sets vmax≈0.2, not theory max log(2)).
# turbo spans a vivid rainbow; PowerNorm stretches the low end so typical
# small BALD values still differentiate instead of looking uniformly blue.
UNCERTAINTY_CMAP = "turbo"
UNCERTAINTY_POWER_GAMMA = 0.45
UNCERTAINTY_DELTA_CMAP = "RdYlBu_r"

# Ref visualization: pass=green, fail=red.
# Newly added: uncertainty → X, diversity/distance → star, other → +.
REF_PASS_COLOR = "green"
REF_FAIL_COLOR = "red"
TEST_COLOR = "orange"
PASS_THRESH_DEFAULT = 0.5
REF_NEW_LABELS = ("ref_new", "ref_new_unc", "ref_new_div")
REF_NEW_MARKERS = {
    "ref_new_unc": "X",
    "ref_new_div": "*",
    "ref_new": "P",  # random / unspecified new refs
}
REF_NEW_LEGEND = {
    "ref_new_unc": "new (uncertainty)",
    "ref_new_div": "new (diversity)",
    "ref_new": "new (other)",
}
# All experiment figures are saved as JPEG.
FIGURE_DPI = 120
FIGURE_JPEG_QUALITY = 90


def save_figure(path, fig=None, dpi=FIGURE_DPI, quality=FIGURE_JPEG_QUALITY, **kwargs):
    """Save the current (or given) figure as JPEG.

    Paths ending in ``.png``/``.jpeg`` are rewritten to ``.jpg``. Transparent
    backgrounds are flattened to white (JPEG has no alpha).
    """
    path = str(path)
    lower = path.lower()
    if lower.endswith('.png') or lower.endswith('.jpeg'):
        path = path.rsplit('.', 1)[0] + '.jpg'
    elif not lower.endswith('.jpg'):
        path = path + '.jpg'

    save_kwargs = {
        'dpi': dpi,
        'format': 'jpg',
        'facecolor': 'white',
        'edgecolor': 'white',
        'pil_kwargs': {'quality': int(quality)},
    }
    save_kwargs.update(kwargs)
    if fig is None:
        plt.savefig(path, **save_kwargs)
    else:
        fig.savefig(path, **save_kwargs)
    return path


def _split_pass_fail(df, pass_thresh=PASS_THRESH_DEFAULT):
    """Split a dataframe into passing / failing rows by test_score."""
    if df is None or len(df) == 0:
        empty = df.iloc[0:0] if df is not None else None
        return empty, empty
    if "test_score" not in df.columns:
        return df, df.iloc[0:0]
    passed = df[df["test_score"] > pass_thresh]
    failed = df[df["test_score"] <= pass_thresh]
    return passed, failed


def _scatter_new_refs(ax, df_vis, *, new_s=90, pass_thresh=PASS_THRESH_DEFAULT, zorder=3, label_suffix=''):
    """Draw newly added refs: X=uncertainty, *=diversity, P=other."""
    if df_vis is None or 'label' not in df_vis.columns:
        return False
    drew = False
    for lab, marker in REF_NEW_MARKERS.items():
        subset = df_vis[df_vis['label'] == lab]
        if len(subset) == 0:
            continue
        new_pass, new_fail = _split_pass_fail(subset, pass_thresh)
        legend_base = REF_NEW_LEGEND[lab]
        if new_pass is not None and len(new_pass) > 0:
            ax.scatter(
                new_pass['Component 1'], new_pass['Component 2'],
                c=REF_PASS_COLOR, s=new_s, marker=marker,
                edgecolors='black', linewidths=0.7,
                alpha=1.0, label=f'{legend_base} pass{label_suffix}', zorder=zorder,
            )
            drew = True
        if new_fail is not None and len(new_fail) > 0:
            ax.scatter(
                new_fail['Component 1'], new_fail['Component 2'],
                c=REF_FAIL_COLOR, s=new_s, marker=marker,
                edgecolors='black', linewidths=0.7,
                alpha=1.0, label=f'{legend_base} fail{label_suffix}', zorder=zorder,
            )
            drew = True
    return drew


def _scatter_ref_layers(ax, df_vis, *, test_s=18, ref_s=36, new_s=90, pass_thresh=PASS_THRESH_DEFAULT):
    """Draw test / old-ref / new-ref layers with pass-fail colors.

    New refs: ``ref_new_unc`` → X, ``ref_new_div`` → star, ``ref_new`` → plus.
    """
    test = df_vis[df_vis["label"] == "test"]
    old_ref = df_vis[df_vis["label"] == "ref"]
    old_pass, old_fail = _split_pass_fail(old_ref, pass_thresh)

    if len(test) > 0:
        ax.scatter(
            test["Component 1"], test["Component 2"],
            c=TEST_COLOR, s=test_s, alpha=0.55, label="test", zorder=1,
        )
    if old_pass is not None and len(old_pass) > 0:
        ax.scatter(
            old_pass["Component 1"], old_pass["Component 2"],
            c=REF_PASS_COLOR, s=ref_s, alpha=0.85, label="ref_pass", zorder=2,
        )
    if old_fail is not None and len(old_fail) > 0:
        ax.scatter(
            old_fail["Component 1"], old_fail["Component 2"],
            c=REF_FAIL_COLOR, s=ref_s, alpha=0.85, label="ref_fail", zorder=2,
        )
    _scatter_new_refs(ax, df_vis, new_s=new_s, pass_thresh=pass_thresh, zorder=3)


def plot_logprob_ranks(df_vis, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.scatterplot(data=df_vis, x='Component 1', y='Component 2', hue='logprob_rank', palette=cmap, alpha=0.7, ax=ax)
    ax.grid(alpha=0.3)
    ax.set_title('Predicted Logprobs (Normalized Ranks)')
    ax.legend([], [], frameon=False)


def plot_probability_density(df_vis, ax=None, cname='logprob', lq=0, uq=100):
    df_vis = df_vis.copy()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
    s = df_vis[cname].to_numpy()
    if lq > 0 or uq < 100:
        vmin, vmax = np.percentile(s, (lq, uq))
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        df_vis['_normed'] = norm(df_vis[cname])
    else:
        df_vis['_normed'] = df_vis[cname]

    sns.scatterplot(data=df_vis, x='Component 1', y='Component 2', hue='_normed', palette=cmap, alpha=0.7, ax=ax)
    ax.grid(alpha=0.3)
    ax.set_title('Probability Density: {}'.format(cname))
    ax.legend([], [], frameon=False)


def plot_reference_points_simple(df_vis, ax=None, pass_thresh=PASS_THRESH_DEFAULT):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Layers: test → old refs (dots) → newly added (X=unc, *=div); green=pass, red=fail.
    _scatter_ref_layers(ax, df_vis, test_s=18, ref_s=36, new_s=90, pass_thresh=pass_thresh)

    ax.grid(alpha=0.3)
    ax.set_title("Reference Points (X=uncertainty/weighted, *=diversity; green=pass, red=fail)")
    ax.legend(loc="best", fontsize=8, frameon=True)
    
    
def plot_score_distribution(df_vis, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    order = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sns.boxplot(data=df_vis, x='test_score', y='logprob', ax=ax, palette='RdYlGn', showfliers=False, order=order)
    ax.set_title('Logprob Distribution by Test Score')
    ax.set_xlabel('Test Score')
    ax.set_ylabel('Predicted Logprob')
    ax.legend([], [], frameon=False)
    

def _uncertainty_norm(vmin, vmax, *, power=True):
    """Build a color norm for absolute uncertainty scales."""
    _vmin = float(vmin)
    _vmax = float(vmax)
    if not np.isfinite(_vmin) or not np.isfinite(_vmax) or _vmax <= _vmin:
        _vmin, _vmax = 0.0, 1.0
    if power:
        return PowerNorm(
            gamma=UNCERTAINTY_POWER_GAMMA, vmin=_vmin, vmax=_vmax, clip=True
        )
    return Normalize(vmin=_vmin, vmax=_vmax, clip=True)


def plot_uncertainties(
    df_vis,
    ax=None,
    title='Uncertainty of Predictions',
    show_ref_new=True,
    use_rank=False,
    vmin=None,
    vmax=None,
    cmap_name=None,
    value_col='uncertainty',
    cbar_label=None,
    show_ood=True,
):
    """Scatter points colored by uncertainty.

    By default colors raw ``uncertainty`` values. Set ``use_rank=True`` for
    rank-based coloring (legacy). Optional ``vmin``/``vmax`` fix the color scale
    (useful when comparing two panels). Absolute scales use ``turbo`` +
    PowerNorm so low BALD values still show colorful variation within
    the fixed ``[vmin, vmax]`` window. Pass ``value_col`` to color by another
    column (e.g. aleatoric ``E[H[p_t]]``).

    If ``is_ood`` is present and ``show_ood``, overlay dual-density OOD points
    (both class GMMs jointly near-zero mass) and annotate the count.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    df_plot = df_vis.copy()
    if value_col not in df_plot.columns:
        raise KeyError(f"plot_uncertainties: missing column '{value_col}'")
    hue_col = value_col
    if use_rank:
        rank_col = f'{value_col}_rank'
        df_plot[rank_col] = rankdata(
            df_plot[value_col].to_numpy(dtype=float), method='average'
        )
        hue_col = rank_col

    vals = df_plot[hue_col].to_numpy(dtype=float)
    cmap_name = UNCERTAINTY_CMAP if cmap_name is None else cmap_name
    if cbar_label is None:
        cbar_label = hue_col

    if vmin is not None or vmax is not None:
        _vmin = float(np.nanmin(vals)) if vmin is None else float(vmin)
        _vmax = float(np.nanmax(vals)) if vmax is None else float(vmax)
        # PowerNorm only for absolute (non-rank) fixed scales.
        norm = _uncertainty_norm(_vmin, _vmax, power=not use_rank)
        sc = ax.scatter(
            df_plot['Component 1'],
            df_plot['Component 2'],
            c=vals,
            cmap=cmap_name,
            norm=norm,
            alpha=0.7,
            s=28,
            edgecolors='none',
            zorder=2,
        )
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)
        # Drop seaborn-style hue legend; colorbar is the scale.
        ax.legend([], [], frameon=False)
    else:
        sns.scatterplot(
            data=df_plot,
            x='Component 1',
            y='Component 2',
            hue=hue_col,
            palette=cmap_name,
            alpha=0.7,
            ax=ax,
        )
        ax.legend([], [], frameon=False)

    drew_new = False
    if show_ref_new and 'label' in df_plot.columns:
        drew_new = _scatter_new_refs(
            ax, df_plot, new_s=90, zorder=6, pass_thresh=PASS_THRESH_DEFAULT
        )

    ood_labeled = False
    if show_ood and 'is_ood' in df_plot.columns:
        ood_mask = df_plot['is_ood'].fillna(False).to_numpy(dtype=bool)
        n_ood = int(ood_mask.sum())
        n_total = int(len(df_plot))
        if 'label' in df_plot.columns:
            is_ref = df_plot['label'].astype(str).str.startswith('ref').to_numpy()
            n_ood_cand = int(np.sum(ood_mask & ~is_ref))
        else:
            n_ood_cand = n_ood
        if n_ood > 0:
            ood_df = df_plot.loc[ood_mask]
            ax.scatter(
                ood_df['Component 1'],
                ood_df['Component 2'],
                s=55,
                facecolors='none',
                edgecolors='black',
                linewidths=1.2,
                marker='o',
                zorder=5,
                label=f'OOD (dual dens.) n={n_ood}',
            )
            ood_labeled = True
        thresh_val = np.nan
        if 'ood_thresh' in df_plot.columns:
            thr = df_plot['ood_thresh'].to_numpy(dtype=float)
            finite = thr[np.isfinite(thr)]
            if finite.size:
                thresh_val = float(finite[0])
        thresh_txt = f'{thresh_val:.1f}' if np.isfinite(thresh_val) else 'n/a'
        ax.text(
            0.02,
            0.98,
            (
                f'OOD (both-class low dens.): {n_ood}/{n_total}'
                f' (non-ref {n_ood_cand})\n'
                f'thresh=ref p1 − 50 = {thresh_txt}'
            ),
            transform=ax.transAxes,
            va='top',
            ha='left',
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='0.6'),
            zorder=10,
        )

    if drew_new or ood_labeled:
        ax.legend(loc='best', fontsize=8, frameon=True)

    ax.grid(alpha=0.3)
    ax.set_title(title)


def _overlay_ref_new(ax, df_vis, label_suffix=''):
    """Overlay newly added refs (X=unc, *=div) from ``df_vis`` onto ``ax``."""
    return _scatter_new_refs(
        ax, df_vis, new_s=110, zorder=7, label_suffix=label_suffix
    )


def _is_ref_label(labels):
    return labels.astype(str).str.startswith('ref')


def _overlay_refs_added_between(
    ax,
    df_prev,
    df_curr,
    *,
    label_suffix='',
    pass_thresh=PASS_THRESH_DEFAULT,
    new_s=110,
):
    """Overlay all refs added between ``df_prev`` and ``df_curr`` onto ``ax``.

    Uses set difference of reference membership by ``input_index`` so that
    points added across multiple iterations between plot checkpoints (not just
    the latest ``ref_new*`` batch) are shown. Coordinates come from ``df_prev``
    (left-panel geometry).
    """
    if df_prev is None or df_curr is None:
        return False
    if 'input_index' not in df_prev.columns or 'input_index' not in df_curr.columns:
        return False
    if 'label' not in df_prev.columns or 'label' not in df_curr.columns:
        return False

    prev_ref_ids = set(df_prev.loc[_is_ref_label(df_prev['label']), 'input_index'].tolist())
    curr_ref_ids = set(df_curr.loc[_is_ref_label(df_curr['label']), 'input_index'].tolist())
    added_ids = curr_ref_ids - prev_ref_ids
    if not added_ids:
        return False

    added = df_prev[df_prev['input_index'].isin(added_ids)].copy()
    if len(added) == 0:
        # Fallback: coords from curr if missing in prev (shouldn't happen).
        added = df_curr[df_curr['input_index'].isin(added_ids)].copy()
    if len(added) == 0:
        return False

    new_pass, new_fail = _split_pass_fail(added, pass_thresh)
    drew = False
    if new_pass is not None and len(new_pass) > 0:
        ax.scatter(
            new_pass['Component 1'], new_pass['Component 2'],
            c=REF_PASS_COLOR, s=new_s, marker='X',
            edgecolors='black', linewidths=0.7,
            alpha=1.0, label=f'added ref pass{label_suffix}', zorder=7,
        )
        drew = True
    if new_fail is not None and len(new_fail) > 0:
        ax.scatter(
            new_fail['Component 1'], new_fail['Component 2'],
            c=REF_FAIL_COLOR, s=new_s, marker='X',
            edgecolors='black', linewidths=0.7,
            alpha=1.0, label=f'added ref fail{label_suffix}', zorder=7,
        )
        drew = True
    return drew


def plot_uncertainty_prediction_compare(
    df_prev,
    df_curr,
    prev_meta,
    curr_meta,
    out_path,
    unc_vmin=None,
    unc_vmax=None,
    delta_vmax=None,
):
    """Side-by-side absolute uncertainty fields for consecutive n.

    Left: prev-n uncertainty with **all** refs added between prev→curr overlaid
    (not only the latest ``ref_new*`` batch).
    Middle: curr-n uncertainty only (no ref overlay).
    Right: absolute uncertainty delta (curr − prev); red = rose.
    Left/middle share a common color scale over absolute uncertainty.
    Optional ``unc_vmin``/``unc_vmax`` fix that scale across a whole run.
    """
    fig, axes = plt.subplots(1, 3, figsize=(27, 7), sharex=False, sharey=False)
    has_prev = df_prev is not None and df_prev['uncertainty'].notna().any()

    # Shared absolute color scale across left/middle for fair comparison.
    if unc_vmin is not None and unc_vmax is not None:
        shared_vmin = float(unc_vmin)
        shared_vmax = float(unc_vmax)
        if not np.isfinite(shared_vmin) or not np.isfinite(shared_vmax) or shared_vmax <= shared_vmin:
            shared_vmin, shared_vmax = 0.0, 1.0
    else:
        unc_vals = [df_curr['uncertainty'].to_numpy(dtype=float)]
        if has_prev:
            unc_vals.append(df_prev['uncertainty'].to_numpy(dtype=float))
        unc_all = np.concatenate(unc_vals)
        unc_all = unc_all[np.isfinite(unc_all)]
        if unc_all.size:
            shared_vmin = float(np.nanmin(unc_all))
            shared_vmax = float(np.nanmax(unc_all))
            if not np.isfinite(shared_vmin) or not np.isfinite(shared_vmax) or shared_vmax <= shared_vmin:
                shared_vmin, shared_vmax = 0.0, 1.0
        else:
            shared_vmin, shared_vmax = 0.0, 1.0

    if has_prev:
        plot_uncertainties(
            df_prev,
            ax=axes[0],
            title=(
                f"Uncertainty at n={prev_meta['n_ref']} "
                f"(iter {prev_meta['iteration']}, n_feat={prev_meta['n_feat']}, "
                f"n_comp={prev_meta['n_comp']})"
            ),
            show_ref_new=False,
            use_rank=False,
            vmin=shared_vmin,
            vmax=shared_vmax,
        )
        drew = _overlay_refs_added_between(
            axes[0],
            df_prev,
            df_curr,
            label_suffix=f" (n={prev_meta['n_ref']}→{curr_meta['n_ref']})",
        )
        if drew:
            axes[0].legend(loc='best', fontsize=8, frameon=True)
    else:
        axes[0].set_title(f"Uncertainty at n=— (no previous n yet)")
        axes[0].axis('off')

    plot_uncertainties(
        df_curr,
        ax=axes[1],
        title=(
            f"Uncertainty at n={curr_meta['n_ref']} "
            f"(iter {curr_meta['iteration']}, n_feat={curr_meta['n_feat']}, "
            f"n_comp={curr_meta['n_comp']})"
        ),
        show_ref_new=False,
        use_rank=False,
        vmin=shared_vmin,
        vmax=shared_vmax,
    )

    # Right: per-point absolute uncertainty change (aligned by input_index).
    if has_prev and 'input_index' in df_prev.columns and 'input_index' in df_curr.columns:
        prev_r = df_prev[['input_index', 'Component 1', 'Component 2', 'uncertainty']].copy()
        curr_r = df_curr[['input_index', 'uncertainty']].copy()
        delta = prev_r.merge(
            curr_r.rename(columns={'uncertainty': 'uncertainty_curr'}),
            on='input_index',
            how='inner',
        )
        delta['uncertainty_delta'] = (
            delta['uncertainty_curr'].to_numpy(dtype=float)
            - delta['uncertainty'].to_numpy(dtype=float)
        )
        if delta_vmax is not None and np.isfinite(delta_vmax) and float(delta_vmax) > 0:
            vmax = float(delta_vmax)
        else:
            vmax = float(np.nanmax(np.abs(delta['uncertainty_delta'].to_numpy(dtype=float))))
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = 1.0
        sc = axes[2].scatter(
            delta['Component 1'],
            delta['Component 2'],
            c=delta['uncertainty_delta'],
            cmap=UNCERTAINTY_DELTA_CMAP,
            vmin=-vmax,
            vmax=vmax,
            alpha=0.7,
            s=18,
        )
        cbar = fig.colorbar(sc, ax=axes[2], fraction=0.046, pad=0.04)
        cbar.set_label('uncertainty(curr) − uncertainty(prev)')
        axes[2].grid(alpha=0.3)
        axes[2].set_xlabel('Component 1')
        axes[2].set_ylabel('Component 2')
        axes[2].set_title(
            f"Uncertainty Δ "
            f"(n={prev_meta['n_ref']}→{curr_meta['n_ref']})"
        )
    else:
        axes[2].set_title('Uncertainty Δ (no previous n yet)')
        axes[2].axis('off')

    fig.suptitle(
        f"{curr_meta.get('task', '')}: Uncertainty of Predictions "
        f"(n={prev_meta['n_ref']} → n={curr_meta['n_ref']})",
        fontsize=12,
    )
    plt.tight_layout()
    save_figure(out_path)
    plt.close(fig)


def _weighted_gaussian_kde(values, weights, grid):
    """Evaluate a weighted 1D Gaussian KDE on ``grid``; returns density + False if skipped."""
    from scipy.stats import gaussian_kde

    values = np.asarray(values, dtype=float).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    if values.size < 2 or not np.isfinite(values).any():
        return np.zeros_like(grid, dtype=float), False
    ok = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    values = values[ok]
    weights = weights[ok]
    if values.size < 2:
        return np.zeros_like(grid, dtype=float), False
    if float(weights.sum()) <= 0:
        weights = np.ones_like(values)
    weights = weights / weights.sum()
    # Degenerate if all mass on one point.
    if np.unique(values).size < 2:
        dens = np.zeros_like(grid, dtype=float)
        # spike near the constant value
        idx = int(np.argmin(np.abs(grid - values[0])))
        dens[idx] = 1.0
        return dens, True
    try:
        kde = gaussian_kde(values, weights=weights)
        dens = np.asarray(kde(grid), dtype=float)
        dens = np.where(np.isfinite(dens), dens, 0.0)
        return dens, True
    except Exception:
        return np.zeros_like(grid, dtype=float), False


def plot_uncertainty_marginal_dist(
    df_vis,
    out_path,
    title=None,
    n_overlay=100,
    overlay_weights=None,
    show_ref_new=False,
    show_ref_pass=True,
    show_ref_fail=False,
    pass_thresh=PASS_THRESH_DEFAULT,
    topk_uncertainty=10,
    unc_vmin=None,
    unc_vmax=None,
    scatter_values=None,
    scatter_label='likelihood',
    scatter_cmap=None,
):
    """Joint scatter + PC1/PC2 marginals in **feature-PCA** space.

    ``df_vis`` must use the same 2D coords as scoring PCA (typically
    ``LIH_rep[n_features][:, :2]``), not t-SNE / vis embeddings.

    Main scatter is colored by ``scatter_values`` when provided (e.g. mean
    predictive log-likelihood under the class GMM). Otherwise falls back to
    ``uncertainty``. Marginal overlays are KDEs weighted by ``overlay_weights``
    (typically per-draw predictive likelihoods ``p(x|θ_t)``).

    Layout (Cartesian quadrants):
      Q1 (top-right): 2D field on feature-PCA (+ optional ref markers)
      Q2 (top-left):  PC2 marginal — ``n_overlay`` transparent curves + mean
      Q4 (bottom-right): PC1 marginal — ``n_overlay`` transparent curves + mean
      Q3 (bottom-left): unused
    """
    df_plot = df_vis.copy()
    x = df_plot['Component 1'].to_numpy(dtype=float)
    y = df_plot['Component 2'].to_numpy(dtype=float)

    if scatter_values is not None:
        cvals = np.asarray(scatter_values, dtype=float).ravel()
        if cvals.shape[0] != len(df_plot):
            raise ValueError(
                f"scatter_values length {cvals.shape[0]} != n_points={len(df_plot)}"
            )
        color_by_likelihood = True
    else:
        if 'uncertainty' not in df_plot.columns or not df_plot['uncertainty'].notna().any():
            raise ValueError(
                "df_vis must contain finite 'uncertainty', or pass scatter_values."
            )
        cvals = df_plot['uncertainty'].to_numpy(dtype=float)
        color_by_likelihood = False

    unc = (
        df_plot['uncertainty'].to_numpy(dtype=float)
        if 'uncertainty' in df_plot.columns
        else np.ones(len(df_plot), dtype=float)
    )
    unc = np.where(np.isfinite(unc), np.clip(unc, 0.0, None), 0.0)
    if float(unc.sum()) <= 0:
        unc = np.ones_like(unc)

    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    x_pad = 0.05 * (x_max - x_min + 1e-9)
    y_pad = 0.05 * (y_max - y_min + 1e-9)
    x_grid = np.linspace(x_min - x_pad, x_max + x_pad, 256)
    y_grid = np.linspace(y_min - y_pad, y_max + y_pad, 256)

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=(1.0, 4.0),
        height_ratios=(4.0, 1.0),
        wspace=0.05,
        hspace=0.05,
    )
    ax_y = fig.add_subplot(gs[0, 0])       # Q2
    ax_main = fig.add_subplot(gs[0, 1])    # Q1
    ax_empty = fig.add_subplot(gs[1, 0])   # Q3
    ax_x = fig.add_subplot(gs[1, 1])       # Q4
    ax_empty.axis('off')

    # Q1: scatter colored by likelihood (preferred) or uncertainty
    if color_by_likelihood:
        finite = cvals[np.isfinite(cvals)]
        if finite.size == 0:
            _vmin, _vmax = 0.0, 1.0
        else:
            _vmin = float(np.nanpercentile(finite, 2))
            _vmax = float(np.nanpercentile(finite, 98))
            if not np.isfinite(_vmin) or not np.isfinite(_vmax) or _vmax <= _vmin:
                _vmin, _vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
                if not np.isfinite(_vmin) or not np.isfinite(_vmax) or _vmax <= _vmin:
                    _vmin, _vmax = 0.0, 1.0
        cmap_sc = UNCERTAINTY_CMAP if scatter_cmap is None else scatter_cmap
        sc = ax_main.scatter(
            x, y, c=cvals, cmap=cmap_sc, vmin=_vmin, vmax=_vmax,
            s=28, alpha=0.7, edgecolors='none', zorder=2,
        )
        cbar = plt.colorbar(sc, ax=ax_main, fraction=0.046, pad=0.04)
        cbar.set_label(scatter_label)
        ax_main.set_title(f'{scatter_label} (feature-PCA)')
        ax_main.grid(alpha=0.3)
        ax_main.legend([], [], frameon=False)
    else:
        plot_uncertainties(
            df_plot,
            ax=ax_main,
            title='Uncertainty (feature-PCA)',
            show_ref_new=show_ref_new,
            use_rank=False,
            vmin=unc_vmin,
            vmax=unc_vmax,
        )
    if show_ref_pass and 'label' in df_plot.columns and 'test_score' in df_plot.columns:
        is_ref = df_plot['label'].isin(['ref', *REF_NEW_LABELS])
        pass_ref = df_plot[is_ref & (df_plot['test_score'] > pass_thresh)]
        if len(pass_ref) > 0:
            ax_main.scatter(
                pass_ref['Component 1'],
                pass_ref['Component 2'],
                s=55,
                c=REF_PASS_COLOR,
                marker='o',
                edgecolors='black',
                linewidths=0.5,
                zorder=6,
                label='ref_pass',
            )
            handles, labels = ax_main.get_legend_handles_labels()
            # Keep hue legend entries + ref_pass (dedupe by label)
            seen = set()
            keep_h, keep_l = [], []
            for h, lab in zip(handles, labels):
                if lab in seen or lab is None:
                    continue
                seen.add(lab)
                keep_h.append(h)
                keep_l.append(lab)
            ax_main.legend(keep_h, keep_l, loc='best', fontsize=8, frameon=True)
    if show_ref_fail and 'label' in df_plot.columns and 'test_score' in df_plot.columns:
        is_ref = df_plot['label'].isin(['ref', *REF_NEW_LABELS])
        fail_ref = df_plot[is_ref & (df_plot['test_score'] <= pass_thresh)]
        if len(fail_ref) > 0:
            ax_main.scatter(
                fail_ref['Component 1'],
                fail_ref['Component 2'],
                s=55,
                c=REF_FAIL_COLOR,
                marker='o',
                edgecolors='black',
                linewidths=0.5,
                zorder=6,
                label='ref_fail',
            )
            handles, labels = ax_main.get_legend_handles_labels()
            seen = set()
            keep_h, keep_l = [], []
            for h, lab in zip(handles, labels):
                if lab in seen or lab is None:
                    continue
                seen.add(lab)
                keep_h.append(h)
                keep_l.append(lab)
            ax_main.legend(keep_h, keep_l, loc='best', fontsize=8, frameon=True)
    ax_main.set_xlim(x_min - x_pad, x_max + x_pad)
    ax_main.set_ylim(y_min - y_pad, y_max + y_pad)
    ax_main.tick_params(labelbottom=False)
    ax_main.set_xlabel('')

    # Highlight top-K uncertainty points used by uncertainty-based selection.
    if int(topk_uncertainty) > 0 and len(df_plot) > 0:
        k = min(int(topk_uncertainty), len(df_plot))
        top_idx = np.argsort(unc)[::-1][:k]
        top_df = df_plot.iloc[top_idx]
        ax_main.scatter(
            top_df['Component 1'],
            top_df['Component 2'],
            s=115,
            marker='D',
            c='none',
            edgecolors='cyan',
            linewidths=1.6,
            zorder=8,
            label=f'top-{k} uncertainty',
        )
        # If available, annotate each point with its global input index.
        if 'input_index' in top_df.columns:
            for _, row in top_df.iterrows():
                ax_main.text(
                    float(row['Component 1']) + 0.5,
                    float(row['Component 2']) + 0.5,
                    str(int(row['input_index'])),
                    fontsize=7,
                    color='cyan',
                    zorder=9,
                )
        handles, labels = ax_main.get_legend_handles_labels()
        seen = set()
        keep_h, keep_l = [], []
        for h, lab in zip(handles, labels):
            if lab in seen or lab is None:
                continue
            seen.add(lab)
            keep_h.append(h)
            keep_l.append(lab)
        ax_main.legend(keep_h, keep_l, loc='best', fontsize=8, frameon=True)

    def _overlay_marginals(ax, values, grid, *, horizontal=False):
        """Draw n_overlay transparent KDEs + their mean curve."""
        cmap_ov = plt.cm.coolwarm(np.linspace(0.15, 0.85, max(n_overlay, 1)))
        overlay_alpha = 0.18 if n_overlay >= 30 else 0.35
        dens_stack = []

        def _push(dens_k, ok_k, color):
            if not ok_k:
                return
            dens_stack.append(np.asarray(dens_k, dtype=float))
            if horizontal:
                ax.plot(dens_k, grid, color=color, alpha=overlay_alpha, lw=1.0)
            else:
                ax.plot(grid, dens_k, color=color, alpha=overlay_alpha, lw=1.0)

        if overlay_weights is not None:
            W = np.asarray(overlay_weights, dtype=float)
            if W.ndim == 1:
                W = W[None, :]
            if W.shape[1] != len(values):
                raise ValueError(
                    f"overlay_weights second dim {W.shape[1]} must match n_points={len(values)}"
                )
            n_use = min(n_overlay, W.shape[0])
            for k in range(n_use):
                dens_k, ok_k = _weighted_gaussian_kde(values, W[k], grid)
                _push(dens_k, ok_k, cmap_ov[k % len(cmap_ov)])
        else:
            ranks = rankdata(unc, method='average')
            edges = np.linspace(ranks.min(), ranks.max(), n_overlay + 1)
            for k in range(n_overlay):
                lo, hi = edges[k], edges[k + 1]
                if k < n_overlay - 1:
                    mask = (ranks >= lo) & (ranks < hi)
                else:
                    mask = (ranks >= lo) & (ranks <= hi)
                if mask.sum() < 2:
                    continue
                w = np.zeros_like(unc)
                w[mask] = unc[mask]
                dens_k, ok_k = _weighted_gaussian_kde(values, w, grid)
                _push(dens_k, ok_k, cmap_ov[k % len(cmap_ov)])

        drew = len(dens_stack)
        has_mean = False
        if drew > 0:
            dens_mean = np.mean(np.stack(dens_stack, axis=0), axis=0)
            if horizontal:
                ax.plot(
                    dens_mean, grid, color='black', lw=2.0,
                    label=f'mean of {drew} overlays',
                )
                ax.fill_betweenx(grid, 0.0, dens_mean, color='black', alpha=0.08)
            else:
                ax.plot(
                    grid, dens_mean, color='black', lw=2.0,
                    label=f'mean of {drew} overlays',
                )
                ax.fill_between(grid, dens_mean, color='black', alpha=0.08)
            has_mean = True
        return drew, has_mean

    dens_axis_label = (
        'KDE dens. (lik.-weighted)' if overlay_weights is not None else 'KDE dens.'
    )

    # Q2: feature-PCA PC2 marginal overlays
    drew_y, ok_y = _overlay_marginals(ax_y, y, y_grid, horizontal=True)
    ax_y.set_ylim(y_min - y_pad, y_max + y_pad)
    ax_y.invert_xaxis()
    ax_y.set_xlabel(dens_axis_label)
    ax_y.set_ylabel('Feature PCA PC2')
    ax_y.set_title(f'Feature-PCA PC2 marginal ({drew_y} overlays)')
    ax_y.grid(alpha=0.3)
    if ok_y:
        ax_y.legend(loc='best', fontsize=7, frameon=True)

    # Q4: feature-PCA PC1 marginal overlays
    drew_x, ok_x = _overlay_marginals(ax_x, x, x_grid, horizontal=False)
    ax_x.set_xlim(x_min - x_pad, x_max + x_pad)
    ax_x.set_xlabel('Feature PCA PC1')
    ax_x.set_ylabel(dens_axis_label)
    ax_x.set_title(f'Feature-PCA PC1 marginal ({drew_x} overlays)')
    ax_x.grid(alpha=0.3)
    if ok_x:
        ax_x.legend(loc='best', fontsize=8, frameon=True)

    ax_y.tick_params(labelleft=True)
    ax_main.tick_params(labelleft=False)
    ax_main.set_ylabel('')

    if title:
        fig.suptitle(title, fontsize=12, y=0.98)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92 if title else 0.96, bottom=0.08)
    save_figure(out_path, dpi=FIGURE_DPI)
    plt.close(fig)


def _ellipse_from_cov2d(mean_2d, cov_2d, n_std=1.0, **kwargs):
    """Build a matplotlib Ellipse for the n_std contour of a 2D Gaussian."""
    cov_2d = np.asarray(cov_2d, dtype=float)
    evals, evecs = np.linalg.eigh(cov_2d)
    evals = np.clip(evals, 1e-12, None)
    order = evals.argsort()[::-1]
    evals, evecs = evals[order], evecs[:, order]
    width, height = 2 * n_std * np.sqrt(evals)
    angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
    return Ellipse(xy=mean_2d, width=width, height=height, angle=angle, **kwargs)


def plot_gmm_components_on_feature_pca(
    df_feat2d,
    means,
    covariances,
    weights=None,
    ax=None,
    n_std=1.0,
    title="GMM (MAP/MLE) centers + 1σ",
):
    """Scatter points in feature-PCA PC1/PC2 and overlay GMM means + n_σ ellipses.

    ``means``: (k, d), ``covariances``: (k, d, d). Only the first two feature dims are drawn.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    _scatter_ref_layers(ax, df_feat2d, test_s=14, ref_s=28, new_s=70)

    means = np.asarray(means, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    if means.ndim != 2 or means.shape[1] < 2:
        raise ValueError("means must have shape (k, d) with d >= 2 for 2D overlay.")
    if covariances.ndim != 3:
        raise ValueError("covariances must have shape (k, d, d).")

    colors = plt.cm.tab10(np.linspace(0, 1, max(means.shape[0], 1)))
    for k in range(means.shape[0]):
        mu2 = means[k, :2]
        cov2 = covariances[k, :2, :2]
        w = None if weights is None else float(weights[k])
        label = f"comp {k}" + (f" (w={w:.2f})" if w is not None else "")
        ax.scatter(
            mu2[0],
            mu2[1],
            marker="X",
            s=120,
            c=[colors[k]],
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
            label=label,
        )
        fill = (*mcolors.to_rgb(colors[k]), 0.18)
        ell = _ellipse_from_cov2d(
            mu2,
            cov2,
            n_std=n_std,
            facecolor=fill,
            edgecolor=colors[k],
            linewidth=2.0,
            zorder=4,
        )
        ax.add_patch(ell)

    ax.grid(alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel("Feature PCA Component 1")
    ax.set_ylabel("Feature PCA Component 2")
    ax.legend(loc="best", fontsize=8, frameon=True)
    return ax


def plot_gmm_posterior_draws_on_feature_pca(
    df_feat2d,
    means_draws,
    covs_draws,
    weights_draws=None,
    ax=None,
    n_std=1.0,
    max_draws=100,
    random_state=42,
    title=None,
    scatter_points=True,
    ellipse_alpha=0.05,
    show_draw_centers=True,
    show_mean_centers=True,
):
    """Overlay ``max_draws`` posterior GMM ellipses on feature-PCA PC1/PC2.

    ``means_draws``: (n_draws, k, d), ``covs_draws``: (n_draws, k, d, d).
    Optionally ``weights_draws``: (n_draws, k) for legend mean weights.
    Draws are subsampled without replacement when ``n_draws > max_draws``.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if scatter_points:
        _scatter_ref_layers(ax, df_feat2d, test_s=14, ref_s=28, new_s=70)

    means_draws = np.asarray(means_draws, dtype=float)
    covs_draws = np.asarray(covs_draws, dtype=float)
    if means_draws.ndim != 3 or means_draws.shape[2] < 2:
        raise ValueError("means_draws must have shape (n_draws, k, d) with d >= 2.")
    if covs_draws.ndim != 4:
        raise ValueError("covs_draws must have shape (n_draws, k, d, d).")
    if means_draws.shape[:2] != covs_draws.shape[:2]:
        raise ValueError("means_draws and covs_draws must share (n_draws, k).")

    n_draws, k, _ = means_draws.shape
    if weights_draws is not None:
        weights_draws = np.asarray(weights_draws, dtype=float)
        if weights_draws.shape != (n_draws, k):
            raise ValueError("weights_draws must have shape (n_draws, k).")

    if n_draws > max_draws:
        rng = np.random.default_rng(random_state)
        idx = np.sort(rng.choice(n_draws, size=max_draws, replace=False))
        means_draws = means_draws[idx]
        covs_draws = covs_draws[idx]
        if weights_draws is not None:
            weights_draws = weights_draws[idx]
        n_draws = max_draws

    colors = plt.cm.tab10(np.linspace(0, 1, max(k, 1)))
    # Transparent edge/fill colors for the cloud of draws.
    edge_colors = [(*mcolors.to_rgb(colors[j]), float(ellipse_alpha)) for j in range(k)]
    fill_colors = [
        (*mcolors.to_rgb(colors[j]), float(ellipse_alpha) * 0.25) for j in range(k)
    ]

    for i in range(n_draws):
        for j in range(k):
            ell = _ellipse_from_cov2d(
                means_draws[i, j, :2],
                covs_draws[i, j, :2, :2],
                n_std=n_std,
                facecolor=fill_colors[j],
                edgecolor=edge_colors[j],
                linewidth=0.8,
                zorder=3,
            )
            ax.add_patch(ell)
        if show_draw_centers:
            for j in range(k):
                ax.scatter(
                    means_draws[i, j, 0],
                    means_draws[i, j, 1],
                    s=6,
                    c=[colors[j]],
                    alpha=max(ellipse_alpha, 0.04),
                    zorder=4,
                    linewidths=0,
                )

    if show_mean_centers:
        mean_mu = means_draws.mean(axis=0)
        mean_w = None if weights_draws is None else weights_draws.mean(axis=0)
        for j in range(k):
            w = None if mean_w is None else float(mean_w[j])
            label = f"comp {j} mean" + (f" (w̄={w:.2f})" if w is not None else "")
            ax.scatter(
                mean_mu[j, 0],
                mean_mu[j, 1],
                marker="X",
                s=120,
                c=[colors[j]],
                edgecolors="black",
                linewidths=0.8,
                zorder=6,
                label=label,
            )

    ax.grid(alpha=0.3)
    ax.set_title(title or f"MCMC posterior draws (n={n_draws}, 1σ)")
    ax.set_xlabel("Feature PCA Component 1")
    ax.set_ylabel("Feature PCA Component 2")
    ax.legend(loc="best", fontsize=8, frameon=True)
    return ax


def plot_mcmc_gmm_parameter_distributions(
    means,
    covariances,
    weights,
    ax=None,
    max_draws=300,
    random_state=42,
    title=None,
):
    """Plot MCMC posterior distributions of GMM parameters.

    Panels:
      1) posterior mean locations in feature-PCA PC1/PC2
      2) mixture-weight posteriors per component
      3) PC1/PC2 scale posteriors (sqrt of diagonal covariance entries)

    ``means``: (n_draws, k, d), ``covariances``: (n_draws, k, d, d),
    ``weights``: (n_draws, k).
    """
    means = np.asarray(means, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if means.ndim != 3 or means.shape[2] < 2:
        raise ValueError("means must have shape (n_draws, k, d) with d >= 2.")
    if covariances.ndim != 4:
        raise ValueError("covariances must have shape (n_draws, k, d, d).")
    if weights.ndim != 2 or weights.shape[:2] != means.shape[:2]:
        raise ValueError("weights must have shape (n_draws, k) matching means.")

    n_draws, k, _ = means.shape
    if n_draws > max_draws:
        rng = np.random.default_rng(random_state)
        idx = np.sort(rng.choice(n_draws, size=max_draws, replace=False))
        means = means[idx]
        covariances = covariances[idx]
        weights = weights[idx]
        n_draws = max_draws

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 3, figsize=(18, 5), sharex=False, sharey=False)
        created_fig = True
    else:
        fig = None
        ax = np.atleast_1d(ax)
        if ax.size < 3:
            raise ValueError("ax must provide at least 3 axes.")

    colors = plt.cm.tab10(np.linspace(0, 1, max(k, 1)))
    mean_of_means = means.mean(axis=0)

    # 1) Posterior mean locations in PC1/PC2
    for j in range(k):
        ax[0].scatter(
            means[:, j, 0],
            means[:, j, 1],
            s=8,
            alpha=0.25,
            c=[colors[j]],
            label=f"comp {j}",
            zorder=2,
        )
        ax[0].scatter(
            mean_of_means[j, 0],
            mean_of_means[j, 1],
            s=120,
            marker="X",
            c=[colors[j]],
            edgecolors="black",
            linewidths=0.8,
            zorder=4,
        )
    ax[0].set_title(f"Posterior means (PC1/PC2, n={n_draws})")
    ax[0].set_xlabel("Feature PCA Component 1")
    ax[0].set_ylabel("Feature PCA Component 2")
    ax[0].grid(alpha=0.3)
    ax[0].legend(loc="best", fontsize=8, frameon=True)

    # 2) Mixture weights
    for j in range(k):
        sns.kdeplot(
            weights[:, j],
            ax=ax[1],
            color=colors[j],
            fill=True,
            alpha=0.25,
            linewidth=1.5,
            label=f"comp {j} (mean={weights[:, j].mean():.2f})",
        )
    ax[1].set_title("Posterior mixture weights")
    ax[1].set_xlabel("Weight")
    ax[1].set_ylabel("Density")
    ax[1].grid(alpha=0.3)
    ax[1].legend(loc="best", fontsize=8, frameon=True)

    # 3) PC1/PC2 scales (std = sqrt diag)
    for j in range(k):
        std1 = np.sqrt(np.clip(covariances[:, j, 0, 0], 1e-12, None))
        std2 = np.sqrt(np.clip(covariances[:, j, 1, 1], 1e-12, None))
        sns.kdeplot(
            std1,
            ax=ax[2],
            color=colors[j],
            linestyle="-",
            linewidth=1.5,
            label=f"comp {j} PC1",
        )
        sns.kdeplot(
            std2,
            ax=ax[2],
            color=colors[j],
            linestyle="--",
            linewidth=1.5,
            label=f"comp {j} PC2",
        )
    ax[2].set_title("Posterior PC1/PC2 scales (std)")
    ax[2].set_xlabel("Std")
    ax[2].set_ylabel("Density")
    ax[2].grid(alpha=0.3)
    ax[2].legend(loc="best", fontsize=7, frameon=True, ncol=2)

    if title:
        if fig is None:
            ax[0].figure.suptitle(title, fontsize=12)
        else:
            fig.suptitle(title, fontsize=12)

    if created_fig:
        fig.tight_layout()
        return fig, ax
    return fig, ax


BALD_DIAG_TAIL_FRAC = 0.10
BALD_DIAG_RANDOM_REPS = 200


def _json_num(x):
    """Python float or None (JSON-safe); maps non-finite to None."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


def _corr_or_nan(x, y, method='pearson'):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3:
        return np.nan, np.nan
    xa, ya = x[m], y[m]
    if float(np.std(xa)) < 1e-15 or float(np.std(ya)) < 1e-15:
        return np.nan, np.nan
    if method == 'spearman':
        r, p = spearmanr(xa, ya)
    else:
        r, p = pearsonr(xa, ya)
    return float(r), float(p)


def compute_bald_pool_diagnostics(
    mi,
    aleatoric,
    *,
    candidate_mask,
    y_binary=None,
    selected_idx=None,
    tail_frac=BALD_DIAG_TAIL_FRAC,
    n_random_reps=BALD_DIAG_RANDOM_REPS,
    random_state=None,
):
    """Pool-level BALD decomposition diagnostics (epistemic-low + confound).

    ``mi`` is I(y; θ | x) = H[p̄] − E_θ[H[p_t]]. ``aleatoric`` is E_θ[H[p_t]].
    Stats are computed on the unlabeled pool (``candidate_mask``).

    Returns ``(summary, extras)``. ``summary`` is JSON-safe scalars;
    ``extras`` holds arrays for plotting and is not written to iteration JSON.
    """
    mi = np.asarray(mi, dtype=float).ravel()
    aleatoric = np.asarray(aleatoric, dtype=float).ravel()
    cand = np.asarray(candidate_mask, dtype=bool).ravel()
    if cand.size != mi.size or aleatoric.size != mi.size:
        raise ValueError(
            f"mi/aleatoric/candidate_mask length mismatch: "
            f"{mi.size}, {aleatoric.size}, {cand.size}"
        )

    mi_p = mi[cand]
    al_p = aleatoric[cand]
    finite = np.isfinite(mi_p) & np.isfinite(al_p)
    mi_f = mi_p[finite]
    al_f = al_p[finite]
    n_pool = int(mi_f.size)

    mi_mean = float(np.mean(mi_f)) if n_pool else np.nan
    mi_std = float(np.std(mi_f, ddof=1)) if n_pool > 1 else np.nan
    al_mean = float(np.mean(al_f)) if n_pool else np.nan
    mi_cv = (mi_std / mi_mean) if (n_pool and np.isfinite(mi_mean) and mi_mean > 0) else np.nan
    mi_over_al = (
        (mi_mean / al_mean)
        if (np.isfinite(mi_mean) and np.isfinite(al_mean) and al_mean > 0)
        else np.nan
    )

    k = max(1, int(round(float(tail_frac) * n_pool))) if n_pool else 0
    if n_pool and k > 0:
        order = np.argsort(mi_f)
        bot = mi_f[order[:k]]
        top = mi_f[order[-k:]]
        mi_top_mean = float(np.mean(top))
        mi_bot_mean = float(np.mean(bot))
        dyn_range = (
            (mi_top_mean / mi_bot_mean) if mi_bot_mean > 0 else np.inf
        )
    else:
        mi_top_mean = mi_bot_mean = dyn_range = np.nan

    pearson_r, pearson_p = _corr_or_nan(mi_f, al_f, 'pearson')
    spearman_r, spearman_p = _corr_or_nan(mi_f, al_f, 'spearman')

    pass_n = fail_n = 0
    pass_mi = fail_mi = pass_al = fail_al = np.nan
    y_pool = None
    if y_binary is not None and n_pool:
        y_all = np.asarray(y_binary, dtype=bool).ravel()
        if y_all.size == cand.size:
            y_pool = y_all[cand][finite]
            pass_n = int(np.sum(y_pool))
            fail_n = int(y_pool.size - pass_n)
            if pass_n:
                pass_mi = float(np.mean(mi_f[y_pool]))
                pass_al = float(np.mean(al_f[y_pool]))
            if fail_n:
                fail_mi = float(np.mean(mi_f[~y_pool]))
                fail_al = float(np.mean(al_f[~y_pool]))

    selected_mean = np.nan
    random_mean = np.nan
    random_std = np.nan
    selected_vs_random = np.nan
    frac_random_le_selected = np.nan
    n_selected = 0
    selected_vals = np.asarray([], dtype=float)
    random_batch_means = np.asarray([], dtype=float)
    random_example_vals = np.asarray([], dtype=float)

    if selected_idx is not None and n_pool:
        sel = np.unique(np.asarray(list(selected_idx), dtype=int).ravel())
        sel = sel[(sel >= 0) & (sel < mi.size)]
        n_selected = int(sel.size)
        if n_selected:
            selected_vals = aleatoric[sel]
            selected_vals = selected_vals[np.isfinite(selected_vals)]
            if selected_vals.size:
                selected_mean = float(np.mean(selected_vals))
            cand_idx = np.flatnonzero(cand)
            rng = np.random.default_rng(random_state)
            n_draw = min(n_selected, int(cand_idx.size))
            if n_draw >= 1 and int(cand_idx.size) >= n_draw:
                means = np.empty(int(n_random_reps), dtype=float)
                for r in range(int(n_random_reps)):
                    draw = rng.choice(cand_idx, size=n_draw, replace=False)
                    means[r] = float(np.mean(aleatoric[draw]))
                random_batch_means = means
                random_mean = float(np.mean(means))
                random_std = float(np.std(means, ddof=1)) if means.size > 1 else 0.0
                if np.isfinite(selected_mean) and random_mean > 0:
                    selected_vs_random = selected_mean / random_mean
                if np.isfinite(selected_mean):
                    frac_random_le_selected = float(
                        np.mean(means <= selected_mean)
                    )
                random_example_vals = aleatoric[
                    rng.choice(cand_idx, size=n_draw, replace=False)
                ]

    summary = {
        'n_pool': n_pool,
        'n_selected': n_selected,
        'tail_frac': float(tail_frac),
        'mi_mean': _json_num(mi_mean),
        'mi_std': _json_num(mi_std),
        'mi_cv': _json_num(mi_cv),
        'mi_top_frac_mean': _json_num(mi_top_mean),
        'mi_bottom_frac_mean': _json_num(mi_bot_mean),
        'mi_dynamic_range': _json_num(dyn_range),
        'aleatoric_mean': _json_num(al_mean),
        'mi_over_aleatoric': _json_num(mi_over_al),
        'pearson_r': _json_num(pearson_r),
        'pearson_p': _json_num(pearson_p),
        'spearman_r': _json_num(spearman_r),
        'spearman_p': _json_num(spearman_p),
        'pass_n': int(pass_n),
        'fail_n': int(fail_n),
        'pass_mi_mean': _json_num(pass_mi),
        'fail_mi_mean': _json_num(fail_mi),
        'pass_aleatoric_mean': _json_num(pass_al),
        'fail_aleatoric_mean': _json_num(fail_al),
        'selected_aleatoric_mean': _json_num(selected_mean),
        'random_aleatoric_mean': _json_num(random_mean),
        'random_aleatoric_std': _json_num(random_std),
        'selected_over_random_aleatoric': _json_num(selected_vs_random),
        'frac_random_le_selected': _json_num(frac_random_le_selected),
    }
    extras = {
        'mi_pool': mi_f,
        'aleatoric_pool': al_f,
        'y_pool': y_pool,
        'selected_aleatoric': np.asarray(selected_vals, dtype=float),
        'random_batch_means': np.asarray(random_batch_means, dtype=float),
        'random_example_values': np.asarray(random_example_vals, dtype=float),
    }
    return summary, extras


def plot_bald_decomposition_diagnostics(
    df_vis,
    metrics,
    extras,
    history,
    out_path,
    title=None,
    selected_idx=None,
    tail_frac=BALD_DIAG_TAIL_FRAC,
):
    """2×3 BALD decomposition diagnostics (every plotted n, same as other figs).

    Row 1: MI histogram + scale stats; MI vs expected-entropy scatter;
    class-wise means.
    Row 2: selected vs random aleatoric; Var_t[p_t] PCA; round-wise history.
    """
    fig, axes = plt.subplots(2, 3, figsize=(24, 12), constrained_layout=True)
    ax_hist, ax_sc, ax_cls = axes[0]
    ax_sel, ax_var, ax_hist_t = axes[1]
    metrics = metrics or {}
    extras = extras or {}
    history = list(history or [])
    pct = int(round(100 * float(tail_frac)))

    mi_p = np.asarray(extras.get('mi_pool', []), dtype=float)
    al_p = np.asarray(extras.get('aleatoric_pool', []), dtype=float)
    y_p = extras.get('y_pool', None)

    if mi_p.size:
        ax_hist.hist(mi_p, bins=40, color='C0', alpha=0.85, edgecolor='white')
    ax_hist.set_xlabel(r'MI $I(y;\theta\mid x)$ (pool)')
    ax_hist.set_ylabel('count')
    ax_hist.set_title('Pool MI distribution')
    ax_hist.grid(alpha=0.3)
    cv = metrics.get('mi_cv')
    dyn = metrics.get('mi_dynamic_range')
    ratio = metrics.get('mi_over_aleatoric')
    box = (
        f"n_pool={metrics.get('n_pool', 'n/a')}\n"
        f"mean(MI)={metrics.get('mi_mean')}\n"
        f"std(MI)={metrics.get('mi_std')}\n"
        f"CV=std/mean={_fmt(cv)}  (<0.1 ≈ flat)\n"
        f"dyn.range top/bot {pct}%={_fmt(dyn)}  (~1 ≈ random)\n"
        f"mean(MI)/mean(E[H])={_fmt(ratio)}  (<0.05 ≈ aleatoric-dominated)"
    )
    ax_hist.text(
        0.98, 0.98, box, transform=ax_hist.transAxes, va='top', ha='right',
        fontsize=8, family='monospace',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.9, edgecolor='0.6'),
    )

    if mi_p.size and al_p.size:
        if y_p is not None and len(y_p) == len(mi_p):
            y_bool = np.asarray(y_p, dtype=bool)
            ax_sc.scatter(
                al_p[~y_bool], mi_p[~y_bool], s=12, alpha=0.45, c=REF_FAIL_COLOR,
                edgecolors='none', label=f"fail n={int((~y_bool).sum())}",
            )
            ax_sc.scatter(
                al_p[y_bool], mi_p[y_bool], s=12, alpha=0.45, c=REF_PASS_COLOR,
                edgecolors='none', label=f"pass n={int(y_bool.sum())}",
            )
        else:
            ax_sc.scatter(al_p, mi_p, s=12, alpha=0.4, c='C0', edgecolors='none')
        if (
            selected_idx is not None
            and df_vis is not None
            and 'aleatoric' in df_vis.columns
            and 'uncertainty' in df_vis.columns
        ):
            sel = np.asarray(list(selected_idx), dtype=int).ravel()
            n_df = len(df_vis)
            sel = sel[(sel >= 0) & (sel < n_df)]
            if sel.size:
                ax_sc.scatter(
                    df_vis['aleatoric'].to_numpy()[sel],
                    df_vis['uncertainty'].to_numpy()[sel],
                    s=70, marker='X', c='black', linewidths=0.4,
                    label=f'selected n={int(sel.size)}', zorder=5,
                )
    ax_sc.set_xlabel(r'expected entropy $\mathbb{E}_\theta[H[p(y|x,\theta)]]$')
    ax_sc.set_ylabel(r'MI $I(y;\theta\mid x)$')
    ax_sc.set_title(
        f"MI vs aleatoric  "
        f"Pearson r={_fmt(metrics.get('pearson_r'))}  "
        f"Spearman ρ={_fmt(metrics.get('spearman_r'))}"
    )
    ax_sc.grid(alpha=0.3)
    ax_sc.legend(loc='best', fontsize=8, frameon=True)

    x = np.arange(2)
    w = 0.35
    mi_bars = [
        metrics.get('fail_mi_mean'),
        metrics.get('pass_mi_mean'),
    ]
    al_bars = [
        metrics.get('fail_aleatoric_mean'),
        metrics.get('pass_aleatoric_mean'),
    ]
    mi_plot = [0.0 if v is None else float(v) for v in mi_bars]
    al_plot = [0.0 if v is None else float(v) for v in al_bars]
    b1 = ax_cls.bar(x - w / 2, mi_plot, w, color='C0', label='mean MI')
    ax_cls2 = ax_cls.twinx()
    b2 = ax_cls2.bar(x + w / 2, al_plot, w, color='C1', label='mean E[H]')
    ax_cls.set_xticks(x)
    ax_cls.set_xticklabels([
        f"fail (n={metrics.get('fail_n', 0)})",
        f"pass (n={metrics.get('pass_n', 0)})",
    ])
    ax_cls.set_ylabel('mean MI', color='C0')
    ax_cls2.set_ylabel('mean E[H]', color='C1')
    ax_cls.set_title('Class-wise MI vs expected entropy')
    ax_cls.grid(alpha=0.3, axis='y')
    ax_cls.legend(handles=[b1, b2], loc='best', fontsize=8)

    sel_al = np.asarray(extras.get('selected_aleatoric', []), dtype=float)
    rand_means = np.asarray(extras.get('random_batch_means', []), dtype=float)
    rand_ex = np.asarray(extras.get('random_example_values', []), dtype=float)
    box_data = []
    box_labels = []
    if sel_al.size:
        box_data.append(sel_al)
        box_labels.append('BALD selected')
    if rand_ex.size:
        box_data.append(rand_ex)
        box_labels.append('one random batch')
    if rand_means.size:
        box_data.append(rand_means)
        box_labels.append('random batch-means')
    if box_data:
        ax_sel.boxplot(box_data, showfliers=False)
        ax_sel.set_xticklabels(box_labels)
        sel_m = metrics.get('selected_aleatoric_mean')
        rnd_m = metrics.get('random_aleatoric_mean')
        if sel_m is not None:
            ax_sel.axhline(sel_m, color='C0', ls='--', lw=1, alpha=0.8)
        if rnd_m is not None:
            ax_sel.axhline(rnd_m, color='C1', ls='--', lw=1, alpha=0.8)
    ax_sel.set_ylabel('expected entropy')
    ax_sel.set_title(
        f"Selected vs random E[H]  "
        f"sel={_fmt(metrics.get('selected_aleatoric_mean'))}  "
        f"rand={_fmt(metrics.get('random_aleatoric_mean'))}  "
        f"sel/rand={_fmt(metrics.get('selected_over_random_aleatoric'))}\n"
        f"P(random mean ≤ selected)={_fmt(metrics.get('frac_random_le_selected'))}  "
        f"(BALD should be lower than random)"
    )
    ax_sel.grid(alpha=0.3, axis='y')
    ax_sel.tick_params(axis='x', labelsize=8)

    if (
        df_vis is not None
        and 'p_var' in df_vis.columns
        and df_vis['p_var'].notna().any()
        and 'Component 1' in df_vis.columns
    ):
        plot_uncertainties(
            df_vis,
            ax=ax_var,
            title=r'MCMC disagreement $\mathrm{Var}_\theta[p(y{=}pass|x,\theta)]$',
            show_ref_new=False,
            use_rank=False,
            vmin=0.0,
            value_col='p_var',
            cbar_label=r'Var$_\theta$[p]',
            show_ood=False,
        )
    else:
        ax_var.set_title('MCMC disagreement (p_var unavailable)')
        ax_var.grid(alpha=0.3)

    if history:
        def _hist_series(key):
            out = []
            for h in history:
                v = h.get(key)
                out.append(np.nan if v is None else v)
            return out

        n_refs = _hist_series('n_ref')
        ax_hist_t.plot(
            n_refs, _hist_series('mi_mean'),
            marker='o', ms=4, label='mean MI', color='C0',
        )
        ax_hist_t.plot(
            n_refs, _hist_series('mi_top_frac_mean'),
            marker='s', ms=3, label=f'top-{pct}% MI', color='C0', ls='--',
        )
        ax_hist_t.set_xlabel('n_ref')
        ax_hist_t.set_ylabel('MI', color='C0')
        ax_h2 = ax_hist_t.twinx()
        ax_h2.plot(
            n_refs, _hist_series('aleatoric_mean'),
            marker='^', ms=4, label='mean E[H]', color='C1',
        )
        ax_h2.plot(
            n_refs, _hist_series('mi_over_aleatoric'),
            marker='x', ms=4, label='mean(MI)/mean(E[H])', color='C2',
        )
        ax_h2.set_ylabel('E[H]  /  MI÷E[H]')
        ax_hist_t.set_title('Round-wise pool MI (monotonic drop ⇒ converging)')
        ax_hist_t.grid(alpha=0.3)
        h1, l1 = ax_hist_t.get_legend_handles_labels()
        h2, l2 = ax_h2.get_legend_handles_labels()
        ax_hist_t.legend(h1 + h2, l1 + l2, loc='best', fontsize=8)
    else:
        ax_hist_t.set_title('Round-wise pool MI (no history yet)')
        ax_hist_t.grid(alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=12)
    save_figure(out_path, fig=fig)
    plt.close(fig)
    return out_path


def _fmt(v, digits=4):
    if v is None:
        return 'n/a'
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 'n/a'
    if not np.isfinite(x):
        return 'n/a'
    return f'{x:.{digits}g}'
