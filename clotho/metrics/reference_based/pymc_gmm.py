from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import (
    _compute_precision_cholesky,
    _estimate_gaussian_parameters,
)


class WeightedGaussianMixture(GaussianMixture):
    """sklearn diagonal/full GMM whose EM M-step accepts inverse-propensity weights.

    ``sample_weight`` is mean-normalized so the total mass stays ``n_samples``
    (MCMC priors keep the same likelihood scale). Unweighted ``fit(X)`` matches
    ``GaussianMixture``.
    """

    def fit(self, X, y=None, sample_weight=None):
        if sample_weight is None:
            self._ipw_sample_weight = None
        else:
            w = np.asarray(sample_weight, dtype=float).ravel()
            X_arr = np.asarray(X)
            if w.shape[0] != X_arr.shape[0]:
                raise ValueError(
                    f"sample_weight length {w.shape[0]} != n_samples {X_arr.shape[0]}"
                )
            w = np.clip(w, 1e-12, None)
            mean = float(w.mean()) if w.size else 1.0
            if (not np.isfinite(mean)) or mean <= 0:
                w = np.ones(w.shape[0], dtype=float)
            else:
                w = w / mean
            self._ipw_sample_weight = w
        return super().fit(X, y)

    def _e_step(self, X):
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
        w = getattr(self, "_ipw_sample_weight", None)
        if w is None:
            return np.mean(log_prob_norm), log_resp
        return float(np.dot(w, log_prob_norm) / np.sum(w)), log_resp

    def _m_step(self, X, log_resp):
        resp = np.exp(log_resp)
        w = getattr(self, "_ipw_sample_weight", None)
        if w is not None:
            resp = resp * w[:, np.newaxis]
        self.weights_, self.means_, self.covariances_ = _estimate_gaussian_parameters(
            X, resp, self.reg_covar, self.covariance_type
        )
        self.weights_ /= self.weights_.sum()
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type
        )

    def _initialize(self, X, resp):
        w = getattr(self, "_ipw_sample_weight", None)
        if w is not None and resp is not None:
            resp = np.asarray(resp, dtype=float) * w[:, np.newaxis]
        super()._initialize(X, resp)



FloatDType = Literal["float32", "float64"]


@dataclass
class PyMCGMMConfig:
    """Config for fixed-k diagonal BlackJAX/PyMC GMM (EM-initialized on cold start).

    Component count is chosen outside this class (e.g. weight-perplexity heuristic
    in the experiment loop). Default ``fit`` runs sklearn EM init + MCMC or JAX ADVI.
    With ``--d`` and unchanged ``k``, the experiment loop warm-starts (reuse last
    posterior-mean / EM params, skip EM). Default sampler is BlackJAX on CPU with
    ``chain_method='parallel'`` (``jax.pmap`` over host CPU devices). ADVI uses
    the same jaxified logp.
    """

    n_components: int = 5
    random_state: int = 42
    # Match Clotho ``GMMScorerConfig`` EM defaults (also used for cov regularization).
    reg_covar: float = 1e-6
    sklearn_n_init: int = 30
    sklearn_max_iter: int = 500
    sklearn_tol: float = 1e-3
    mcmc_draws: int = 1000
    mcmc_tune: int = 1000
    mcmc_chains: int = 2
    nuts_sampler: str = "blackjax"
    nuts_sampler_kwargs: dict = field(default_factory=lambda: {"chain_method": "parallel"})
    target_accept: float = 0.9
    # Posterior inference: NUTS (``mcmc``) or JAX ADVI (mean-field / full-rank).
    inference: Literal["mcmc", "advi"] = "mcmc"
    advi_method: Literal["advi", "fullrank_advi"] = "advi"
    advi_n_iter: int = 20000
    # Adam step size for JAX ADVI (unconstrained space).
    advi_learning_rate: float = 0.01
    # Independent draws stored from the variational approximation (then thinned
    # to ``n_uncertainty_draws`` for BALD). Unlike NUTS, these are not autocorrelated.
    advi_draws: int = 200
    mu_prior_sigma: float = 1.0
    sigma_prior_sigma: float = 1.5
    # If True, center μ / scale / weight priors on EM (or warm-start) init_params
    # instead of N(0,·) / HalfNormal(·) / Dirichlet(1,…).
    em_centered_prior: bool = True
    # Dirichlet concentration for EM-centered weights: Dirichlet(c * w_EM).
    weight_prior_concentration: float = 10.0
    # Epistemic multi-start: N loose sklearn EMs (different seeds, fewer n_init
    # than the scoring EM) become NUTS chain inits with uncentered priors.
    epistemic_multistart: bool = False
    epistemic_em_inits: int = 4
    epistemic_em_n_init: int = 3
    floatx: FloatDType = "float32"
    jax_platform: str = "cpu"
    disable_xla_preallocate: bool = True
    n_uncertainty_draws: int = 100
    # If set (e.g. ``--fast``), BALD uses every ``stride``-th flattened draw
    # instead of targeting ``n_uncertainty_draws``.
    uncertainty_thin_stride: int | None = None
    verbose: bool = False


def _diagonal_gmm_loglik(X, mu, sigma, w, sample_weight=None):
    """Log-likelihood of a diagonal-covariance Gaussian mixture.

    Uses independent Normal factors per feature (no dense cov / MvNormal path):
    ``log p(x|k) = sum_j log N(x_j | mu_kj, sigma_kj)``.
    ``X`` is ``(n, d)``; ``mu``/``sigma`` are ``(k, d)``; ``w`` is ``(k,)``.
    Optional ``sample_weight`` ``(n,)`` implements IPW (``sum_i w_i log p(x_i)``).
    """
    diff = (X[:, None, :] - mu[None, :, :]) / sigma[None, :, :]
    log_comp = (
        -0.5 * pt.sum(pt.sqr(diff), axis=-1)
        - pt.sum(pt.log(sigma), axis=-1)
        - 0.5 * X.shape[1] * pt.log(2.0 * np.pi)
    )
    log_mix = pm.math.logsumexp(pt.log(w)[None, :] + log_comp, axis=1)
    if sample_weight is None:
        return pt.sum(log_mix)
    return pt.sum(log_mix * sample_weight)


def _gauss_logpdf(X, means, variances) -> np.ndarray:
    """Diagonal Gaussian log-densities. Returns (n, k).

    ``variances`` may be ``(k, d)`` or full mats ``(k, d, d)`` (diag extracted).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    means = np.asarray(means, dtype=float)
    variances = np.asarray(variances, dtype=float)
    if variances.ndim == 3:
        variances = np.diagonal(variances, axis1=-2, axis2=-1)
    diff = X[:, None, :] - means[None, :, :]
    log_norm = -0.5 * (np.log(2 * np.pi) + np.log(variances[None, :, :]))
    return np.sum(-0.5 * diff**2 / variances[None, :, :] + log_norm, axis=2)


_NUMBA_MIXTURE_LL = None


def _diag_mixture_loglik_numba(X, means, vars_, log_w):
    """Numba-parallel mixture LL. ``X (n,d)``, rest ``(S,k,*)``. Returns ``(S, n)``."""
    global _NUMBA_MIXTURE_LL
    if _NUMBA_MIXTURE_LL is None:
        from numba import njit, prange

        @njit(parallel=True, fastmath=True, cache=True)
        def _kernel(X_, means_, vars_, log_w_):
            n_draws, n_comp, n_features = means_.shape
            n_samples = X_.shape[0]
            out = np.empty((n_draws, n_samples), dtype=np.float64)
            log_2pi = np.log(2.0 * np.pi)
            for s in prange(n_draws):
                comps = np.empty(n_comp, dtype=np.float64)
                for i in range(n_samples):
                    max_c = -1.0e300
                    for j in range(n_comp):
                        acc = log_w_[s, j]
                        for t in range(n_features):
                            var = vars_[s, j, t]
                            if var < 1e-12:
                                var = 1e-12
                            diff = X_[i, t] - means_[s, j, t]
                            acc += -0.5 * (log_2pi + np.log(var) + (diff * diff) / var)
                        comps[j] = acc
                        if acc > max_c:
                            max_c = acc
                    total = 0.0
                    for j in range(n_comp):
                        total += np.exp(comps[j] - max_c)
                    out[s, i] = max_c + np.log(total)
            return out

        _NUMBA_MIXTURE_LL = _kernel
    return _NUMBA_MIXTURE_LL(X, means, vars_, log_w)


def _diag_mixture_loglik_batched(X, means, variances, weights) -> np.ndarray:
    """Mixture log-likelihood for every posterior draw. Returns ``(S, n)``.

    ``X`` is ``(n, d)``; ``means``/``variances`` are ``(S, k, d)`` (or
    ``variances`` ``(S, k, d, d)`` with diag extracted); ``weights`` is ``(S, k)``.
    Uses a Numba-parallel kernel over draws when available; otherwise a
    component-wise NumPy broadcast.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    means = np.asarray(means)
    variances = np.asarray(variances)
    weights = np.asarray(weights)
    if means.ndim != 3:
        raise ValueError(f"means must be (S, k, d), got {means.shape}.")
    if weights.ndim != 2 or weights.shape[:2] != means.shape[:2]:
        raise ValueError(
            f"weights must be (S, k) matching means {means.shape[:2]}, got {weights.shape}."
        )
    if variances.ndim == 4:
        variances = np.diagonal(variances, axis1=-2, axis2=-1)
    if variances.shape != means.shape:
        raise ValueError(
            f"variances shape {variances.shape} must match means {means.shape}."
        )
    vars_ = np.ascontiguousarray(
        np.maximum(np.asarray(variances, dtype=np.float64), 1e-12)
    )
    X64 = np.ascontiguousarray(X, dtype=np.float64)
    means64 = np.ascontiguousarray(means, dtype=np.float64)
    log_w = np.ascontiguousarray(
        np.log(np.asarray(weights, dtype=np.float64) + 1e-300)
    )
    try:
        return _diag_mixture_loglik_numba(X64, means64, vars_, log_w)
    except Exception:
        n_samples = int(X64.shape[0])
        n_draws, n_comp, _ = means64.shape
        log_comp = np.empty((n_draws, n_samples, n_comp), dtype=np.float64)
        half_log_2pi = 0.5 * np.log(2.0 * np.pi)
        for j in range(n_comp):
            mu_j = means64[:, j, :][:, None, :]
            v_j = vars_[:, j, :][:, None, :]
            diff = X64[None, :, :] - mu_j
            log_comp[:, :, j] = np.sum(
                -(half_log_2pi + 0.5 * np.log(v_j) + 0.5 * np.square(diff) / v_j),
                axis=-1,
            )
        return logsumexp(log_w[:, None, :] + log_comp, axis=-1)


def _mixture_responsibilities(X, means, variances, weights) -> np.ndarray:
    log_weights = np.log(np.asarray(weights) + 1e-300)
    log_comp = _gauss_logpdf(X, means, variances)
    log_resp = log_weights[None, :] + log_comp
    log_resp -= logsumexp(log_resp, axis=1, keepdims=True)
    return np.exp(log_resp)


def diag_gmm_n_parameters(n_components: int, n_features: int) -> int:
    """Free-parameter count for a diagonal-covariance GMM (sklearn convention)."""
    return int(n_components * n_features + n_components * n_features + n_components - 1)


def diag_gmm_aic_bic(
    X: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    """AIC/BIC for diagonal-covariance GMM params (same formulas as sklearn)."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}.")
    means = np.asarray(means, dtype=float)
    variances = np.asarray(variances, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if variances.ndim == 3:
        variances = np.diagonal(variances, axis1=-2, axis2=-1)
    n_samples, n_features = X.shape
    n_components = means.shape[0]
    n_params = diag_gmm_n_parameters(n_components, n_features)
    log_weights = np.log(weights + 1e-300)
    log_lik = float(
        logsumexp(
            log_weights[None, :] + _gauss_logpdf(X, means, variances),
            axis=1,
        ).sum()
    )
    aic = -2.0 * log_lik + 2.0 * n_params
    bic = -2.0 * log_lik + n_params * np.log(n_samples)
    return float(aic), float(bic)


def _hungarian_permutation(means: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return permutation of draw components that best matches reference labels."""
    cost = ((means[:, None, :] - reference[None, :, :]) ** 2).sum(axis=-1)
    _, col = linear_sum_assignment(cost)
    return np.argsort(col)


def _thin_indices(n_total: int, n_keep: int) -> np.ndarray:
    """Evenly spaced thinning indices over ``[0, n_total)`` (length ``n_keep``)."""
    n_total = int(n_total)
    n_keep = int(n_keep)
    if n_keep < 1:
        raise ValueError(f"n_keep must be >= 1, got {n_keep}.")
    if n_keep > n_total:
        raise ValueError(f"n_keep={n_keep} exceeds n_total={n_total}.")
    if n_keep == n_total:
        return np.arange(n_total, dtype=int)
    # stride thinning: 0, floor(stride), floor(2*stride), ...
    stride = n_total / float(n_keep)
    indices = (np.arange(n_keep, dtype=float) * stride).astype(int)
    return np.clip(indices, 0, n_total - 1)


def _ensure_floatx_runtime(
    floatx: FloatDType,
    disable_xla_preallocate: bool,
    jax_platform: str | None = "cpu",
    jax_cpu_device_count: int | None = None,
) -> None:
    # Harmless JAX noise under float32 + BlackJAX: PyTensor may request float64
    # accumulators, and buffer donation may be skipped.
    warnings.filterwarnings(
        "ignore",
        message=r"Explicitly requested dtype float64.*",
        category=UserWarning,
        module=r"jax\..*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Some donated buffers were not usable:.*",
        category=UserWarning,
        module=r"jax\..*",
    )

    if disable_xla_preallocate:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    # Force JAX device before backend init (BlackJAX path). Does not affect PyTorch CUDA.
    if jax_platform:
        os.environ["JAX_PLATFORMS"] = jax_platform
    # BlackJAX chain_method='parallel' uses jax.pmap and needs one host device per chain.
    if jax_platform == "cpu" and jax_cpu_device_count is not None and jax_cpu_device_count > 0:
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={int(jax_cpu_device_count)}"
    # Prefer updating pytensor config if already imported; otherwise set flags for later imports.
    # ``fast_compile`` skips the heavy C rewrite pipeline that dominated ADVI ``pm.fit``.
    try:
        import pytensor

        pytensor.config.floatX = floatx
        pytensor.config.optimizer = "fast_compile"
    except Exception:
        os.environ["PYTENSOR_FLAGS"] = f"floatX={floatx},optimizer=fast_compile"

    if floatx == "float32" or jax_platform:
        try:
            import jax

            if floatx == "float32":
                jax.config.update("jax_enable_x64", False)
            if jax_platform:
                # Best-effort if JAX was not yet fully initialized.
                try:
                    jax.config.update("jax_platforms", jax_platform)
                except Exception:
                    pass
        except Exception:
            pass


class _JaxADVIApprox:
    """Lightweight stand-in for PyMC's Approximation (stores ELBO hist + q params)."""

    def __init__(self, hist: np.ndarray, loc: np.ndarray, method: str):
        self.hist = np.asarray(hist)
        self.loc = np.asarray(loc)
        self.method = method


def _ravel_leaves(leaves: list) -> tuple:
    import jax.numpy as jnp

    flats = [jnp.ravel(x) for x in leaves]
    sizes = tuple(int(x.size) for x in leaves)
    shapes = tuple(tuple(int(s) for s in x.shape) for x in leaves)
    return jnp.concatenate(flats), sizes, shapes


def _unravel_flat(flat, sizes: tuple, shapes: tuple) -> list:
    import jax.numpy as jnp

    parts = []
    offset = 0
    for size, shape in zip(sizes, shapes):
        parts.append(flat[offset : offset + size].reshape(shape))
        offset += size
    return parts


def _fit_jax_advi(
    model: pm.Model,
    *,
    n_iter: int,
    n_draws: int,
    method: str,
    random_seed: int,
    learning_rate: float = 0.01,
) -> tuple:
    """Mean-field / full-rank ADVI in unconstrained space, compiled with JAX.

    Uses the same ``get_jaxified_logp`` path as BlackJAX NUTS, then Adam on the
    ELBO. Draws are mapped back to constrained RVs (``mu``, ``sigma``, ``w``)
    via PyMC's JAX postprocessing so ``idata_`` matches the NUTS layout.
    """
    import arviz as az
    import jax
    import jax.numpy as jnp
    from pymc.sampling.jax import (
        _postprocess_samples,
        coords_and_dims_for_inferencedata,
        get_jaxified_graph,
        get_jaxified_logp,
    )
    from pymc.backends.arviz import find_constants, find_observations
    from pymc.util import get_default_varnames

    compile_t0 = time.time()
    logp_fn = get_jaxified_logp(model, negative_logp=True)

    ip = model.initial_point()
    init_leaves = [np.asarray(ip[v.name]) for v in model.value_vars]
    flat0, sizes, shapes = _ravel_leaves(init_leaves)
    n_dim = int(flat0.shape[0])
    n_off = n_dim * (n_dim - 1) // 2
    use_fullrank = method == "fullrank_advi" and n_off > 0

    # Warm up logp + XLA so the scan does not hide compile time.
    logp_fn(init_leaves)
    jaxify_s = time.time() - compile_t0

    def _scale_from_rho(rho):
        return jax.nn.softplus(rho) + jnp.asarray(1e-5, rho.dtype)

    def _pack_scale_tril(rho_diag, off):
        diag = _scale_from_rho(rho_diag)
        L = jnp.diag(diag)
        if n_off > 0:
            L = L.at[jnp.tril_indices(n_dim, k=-1)].set(off)
        return L

    def _neg_elbo(params, key):
        eps = jax.random.normal(key, (n_dim,), dtype=flat0.dtype)
        if use_fullrank:
            loc, rho_diag, off = params
            L = _pack_scale_tril(rho_diag, off)
            z = loc + L @ eps
            log_q = -0.5 * jnp.sum(jnp.square(eps)) - jnp.sum(
                jnp.log(jnp.diag(L))
            ) - 0.5 * n_dim * jnp.log(2.0 * jnp.pi)
        else:
            loc, rho = params
            scale = _scale_from_rho(rho)
            z = loc + scale * eps
            log_q = -0.5 * jnp.sum(jnp.square(eps) + 2.0 * jnp.log(scale) + jnp.log(2.0 * jnp.pi))
        log_p = logp_fn(_unravel_flat(z, sizes, shapes))
        return -(log_p - log_q)

    def _adam_step(state, key, lr=learning_rate, b1=0.9, b2=0.999, eps=1e-8):
        params, m, v, t = state
        loss, grads = jax.value_and_grad(_neg_elbo)(params, key)
        t = t + 1
        m = jax.tree.map(lambda mi, g: b1 * mi + (1.0 - b1) * g, m, grads)
        v = jax.tree.map(lambda vi, g: b2 * vi + (1.0 - b2) * jnp.square(g), v, grads)
        mhat = jax.tree.map(lambda mi: mi / (1.0 - b1**t), m)
        vhat = jax.tree.map(lambda vi: vi / (1.0 - b2**t), v)
        params = jax.tree.map(
            lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps),
            params,
            mhat,
            vhat,
        )
        elbo = -loss
        return (params, m, v, t), elbo

    if use_fullrank:
        # softplus(rho)≈0.1
        rho0 = jnp.full((n_dim,), -2.252, dtype=flat0.dtype)
        off0 = jnp.zeros((n_off,), dtype=flat0.dtype)
        params0 = (flat0, rho0, off0)
    else:
        rho0 = jnp.full((n_dim,), -2.252, dtype=flat0.dtype)
        params0 = (flat0, rho0)
    zeros = jax.tree.map(jnp.zeros_like, params0)
    init_state = (params0, zeros, zeros, jnp.asarray(0, dtype=jnp.int32))

    rng0 = jax.random.PRNGKey(int(random_seed))
    rng0, train_key, draw_key = jax.random.split(rng0, 3)
    train_keys = jax.random.split(train_key, n_iter)

    def _train(state, keys):
        return jax.lax.scan(_adam_step, state, keys)

    train_t0 = time.time()
    (params_hat, _, _, _), elbo_hist = jax.jit(_train)(init_state, train_keys)
    params_hat = jax.tree.map(lambda x: np.asarray(x), params_hat)
    elbo_hist = np.asarray(elbo_hist)
    fit_s = time.time() - train_t0

    sample_t0 = time.time()
    loc = jnp.asarray(params_hat[0])
    if use_fullrank:
        L = _pack_scale_tril(jnp.asarray(params_hat[1]), jnp.asarray(params_hat[2]))

        def _one_draw(key):
            eps = jax.random.normal(key, (n_dim,), dtype=loc.dtype)
            return tuple(_unravel_flat(loc + L @ eps, sizes, shapes))
    else:
        scale = _scale_from_rho(jnp.asarray(params_hat[1]))

        def _one_draw(key):
            eps = jax.random.normal(key, (n_dim,), dtype=loc.dtype)
            return tuple(_unravel_flat(loc + scale * eps, sizes, shapes))

    draw_keys = jax.random.split(draw_key, n_draws)
    raw_draws = jax.jit(jax.vmap(_one_draw))(draw_keys)
    # (n_draws, *shape) -> (chain=1, n_draws, *shape) to match NUTS layout.
    raw_mcmc_samples = [np.asarray(s)[None, ...] for s in raw_draws]

    vars_to_sample = list(
        get_default_varnames(model.unobserved_value_vars, include_transformed=False)
    )
    jax_fn = get_jaxified_graph(inputs=model.value_vars, outputs=vars_to_sample)
    result = _postprocess_samples(
        jax_fn,
        raw_mcmc_samples,
        postprocessing_vectorize="vmap",
        donate_samples=False,
    )
    posterior = {v.name: np.asarray(r) for v, r in zip(vars_to_sample, result)}
    coords, dims = coords_and_dims_for_inferencedata(model)
    idata = az.from_dict(
        posterior=posterior,
        coords=coords,
        dims=dims,
        observed_data=find_observations(model),
        constant_data=find_constants(model),
    )
    sample_s = time.time() - sample_t0
    approx = _JaxADVIApprox(hist=elbo_hist, loc=np.asarray(params_hat[0]), method=method)
    times = {
        "advi_jaxify": jaxify_s,
        "advi_fit": fit_s,
        "advi_sample": sample_s,
        "advi_compile": jaxify_s,
    }
    return idata, approx, times


class PyMCGaussianMixture:
    """Fixed-k diagonal GMM fitted with PyMC (NUTS or ADVI).

    sklearn EM initializes inference and, when ``em_centered_prior=True`` (default),
    also centers the μ / scale / weight priors. Density scores use EM point
    estimates; draw-wise LLs come from the posterior (NUTS samples or ADVI draws).
    """

    def __init__(self, config: PyMCGMMConfig | None = None):
        self.config = config or PyMCGMMConfig()
        self.n_components: int | None = None
        self.reg_covar = self.config.reg_covar
        self.means_: np.ndarray | None = None
        # Diagonal variances, shape (k, d).
        self.covariances_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self.idata_ = None
        self._pymc_model = None
        self.em_init_: GaussianMixture | None = None
        self.bic_: float | None = None  # EM-init BIC (always set when fit)
        self.em_aic_: float | None = None
        self.mcmc_aic_: float | None = None  # posterior-mean AIC/BIC
        self.mcmc_bic_: float | None = None
        self.mcmc_means_: np.ndarray | None = None
        self.mcmc_covariances_: np.ndarray | None = None
        self.mcmc_weights_: np.ndarray | None = None
        self._reference_means_: np.ndarray | None = None
        self.approx_ = None
        self.warm_start_: bool = False
        self.last_fit_profile_: dict[str, float] | None = None
        self.inference_ = str(self.config.inference)
        # Optional scoring overlay (MAP MCMC draw). Warm-start still uses EM ``means_``.
        self.score_means_: np.ndarray | None = None
        self.score_covariances_: np.ndarray | None = None
        self.score_weights_: np.ndarray | None = None
        self.map_draw_index_: int | None = None
        self.map_train_ll_: float | None = None
        self.epistemic_em_inits_: list[dict[str, np.ndarray]] | None = None

    @property
    def _dtype(self) -> np.dtype:
        """Numpy dtype matching ``config.floatx`` (default float32)."""
        return np.dtype(np.float32 if self.config.floatx == "float32" else np.float64)


    def fit(
        self,
        X: np.ndarray,
        init_params: dict[str, np.ndarray] | None = None,
        *,
        warm_start: bool = False,
        run_mcmc: bool = True,
        sample_weight: np.ndarray | None = None,
    ):
        inference = str(getattr(self.config, "inference", "mcmc"))
        self.inference_ = inference
        chain_method = None
        if self.config.nuts_sampler_kwargs:
            chain_method = self.config.nuts_sampler_kwargs.get("chain_method")
        jax_cpu_devices = None
        run_nuts = bool(run_mcmc and inference == "mcmc")
        run_advi = bool(run_mcmc and inference == "advi")
        if (
            run_nuts
            and self.config.jax_platform == "cpu"
            and self.config.nuts_sampler == "blackjax"
            and chain_method == "parallel"
        ):
            # pmap needs >= chains host devices; cap by available CPU cores.
            n_cores = os.cpu_count() or 1
            n_chains_eff = int(self.config.mcmc_chains)
            if bool(getattr(self.config, "epistemic_multistart", False)):
                n_chains_eff = max(
                    n_chains_eff, int(getattr(self.config, "epistemic_em_inits", 4))
                )
            jax_cpu_devices = max(1, min(n_chains_eff, int(n_cores)))
        if run_nuts or run_advi:
            _ensure_floatx_runtime(
                self.config.floatx,
                self.config.disable_xla_preallocate,
                getattr(self.config, "jax_platform", "cpu"),
                jax_cpu_device_count=jax_cpu_devices,
            )
        elif self.config.floatx:
            try:
                import pytensor

                pytensor.config.floatX = self.config.floatx
                pytensor.config.optimizer = "fast_compile"
            except Exception:
                os.environ.setdefault(
                    "PYTENSOR_FLAGS",
                    f"floatX={self.config.floatx},optimizer=fast_compile",
                )
        dtype = self._dtype

        X = np.asarray(X, dtype=dtype)
        if X.ndim == 1:
            X = X[:, None]
        if X.ndim != 2:
            raise ValueError("Training data must be a 2D array of shape (n_samples, n_features).")

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.warm_start_ = bool(warm_start and init_params is not None)
        self.score_means_ = None
        self.score_covariances_ = None
        self.score_weights_ = None
        self.map_draw_index_ = None
        self.map_train_ll_ = None
        self.epistemic_em_inits_ = None
        fit_t0 = time.time()
        try:
            import jax

            jax_dev_info = f"{jax.default_backend()}:{len(jax.devices())}"
        except Exception:
            jax_dev_info = "unknown"

        profile: dict[str, float] = {"warm_start": 1.0 if self.warm_start_ else 0.0}
        # Cold-start: sklearn EM seeds scoring (+ optional MCMC/ADVI).
        # Warm-start skips EM and reuses last params as initvals / prior centers.
        if self.warm_start_:
            k = int(np.asarray(init_params["mu"]).shape[0])
        else:
            k = int(self.config.n_components)
        if k < 1:
            raise ValueError(f"n_components must be >= 1, got {k}.")
        if n_samples < k:
            raise ValueError(
                f"Number of samples ({n_samples}) must be >= n_components ({k})."
            )

        X_real = X
        if self.warm_start_:
            self.em_init_ = None
            self.bic_ = None
            self.em_aic_ = None
            profile["em_init"] = 0.0
            mu = np.asarray(init_params["mu"], dtype=dtype)
            weights = np.asarray(init_params["w"], dtype=dtype)
            if mu.ndim != 2 or mu.shape[1] != n_features:
                raise ValueError(
                    f"Warm-start mu shape {mu.shape} incompatible with n_features={n_features}."
                )
            if weights.shape != (mu.shape[0],):
                raise ValueError(
                    f"Warm-start w shape {weights.shape} must be ({mu.shape[0]},)."
                )
            if "sigma" in init_params:
                sigma = np.asarray(init_params["sigma"], dtype=dtype)
            elif "cov" in init_params:
                covs = np.asarray(init_params["cov"], dtype=dtype)
                if covs.ndim == 3:
                    vars_ = np.diagonal(covs, axis1=-2, axis2=-1)
                elif covs.ndim == 2:
                    vars_ = covs
                else:
                    raise ValueError(f"Warm-start cov shape {covs.shape} not supported.")
                sigma = np.sqrt(np.maximum(vars_, 1e-12)).astype(dtype)
            else:
                raise ValueError("Warm-start init_params require 'sigma' or 'cov'.")
            if sigma.shape != mu.shape:
                raise ValueError(
                    f"Warm-start sigma shape {sigma.shape} must match mu shape {mu.shape}."
                )
            self.n_components = int(mu.shape[0])
            init_params = {"mu": mu, "sigma": sigma, "w": weights}
        else:
            em_t0 = time.time()
            em_init = WeightedGaussianMixture(
                n_components=k,
                covariance_type="diag",
                reg_covar=self.config.reg_covar,
                n_init=self.config.sklearn_n_init,
                max_iter=self.config.sklearn_max_iter,
                tol=self.config.sklearn_tol,
                random_state=self.config.random_state,
            ).fit(X_real, sample_weight=sample_weight)
            self.em_init_ = em_init
            self.bic_ = float(em_init.bic(X_real))
            self.em_aic_ = float(em_init.aic(X_real))
            self.n_components = k
            profile["em_init"] = time.time() - em_t0
            sigma = np.sqrt(np.maximum(em_init.covariances_, 1e-12)).astype(dtype)
            init_params = {
                "mu": em_init.means_.astype(dtype),
                "sigma": sigma,
                "w": em_init.weights_.astype(dtype),
            }

        if n_samples < self.n_components:
            raise ValueError(
                f"Number of samples ({n_samples}) must be >= n_components ({self.n_components})."
            )

        # Density score / rank always use EM (or warm-start) point estimates.
        self._set_em_point_estimates(init_params)
        self._reference_means_ = np.asarray(init_params["mu"], dtype=dtype)

        if not run_mcmc:
            # Keep any previous posterior (idata_) for uncertainty reuse unless cleared by caller.
            self.mcmc_aic_ = None
            self.mcmc_bic_ = None
            self.mcmc_means_ = None
            self.mcmc_covariances_ = None
            self.mcmc_weights_ = None
            profile["fit_total"] = time.time() - fit_t0
            self.last_fit_profile_ = profile
            return self

        use_epimulti = bool(getattr(self.config, "epistemic_multistart", False))
        if use_epimulti and inference != "mcmc":
            raise ValueError("epistemic_multistart requires inference='mcmc'")
        mcmc_init = init_params
        if use_epimulti:
            epi_t0 = time.time()
            self.epistemic_em_inits_ = self._fit_epistemic_em_inits(
                X_real, self.n_components, sample_weight=sample_weight
            )
            profile["epistemic_em_inits"] = float(len(self.epistemic_em_inits_))
            profile["epistemic_em_n_init"] = float(
                getattr(self.config, "epistemic_em_n_init", 3)
            )
            profile["epistemic_em_fit"] = time.time() - epi_t0
            mcmc_init = self.epistemic_em_inits_[0]

        build_t0 = time.time()
        self._pymc_model = self._build_multivariate_model(
            X_real, self.n_components, mcmc_init, sample_weight=sample_weight
        )
        profile["build_pymc_model"] = time.time() - build_t0
        profile["inference"] = inference

        if inference == "advi":
            profile.update(self._run_advi())
        elif inference == "mcmc":
            profile.update(self._run_mcmc())
        else:
            raise ValueError(
                f"Unknown inference={inference!r}; expected 'mcmc' or 'advi'."
            )

        relabel_t0 = time.time()
        self._relabel_posterior_to_reference()
        profile["relabel_posterior"] = time.time() - relabel_t0

        if inference == "mcmc":
            conv_t0 = time.time()
            self._run_convergence_checks()
            profile["convergence_checks"] = time.time() - conv_t0
        else:
            profile["convergence_checks"] = 0.0

        pe_t0 = time.time()
        self._set_mcmc_point_estimates()
        self.mcmc_aic_, self.mcmc_bic_ = diag_gmm_aic_bic(
            X_real, self.mcmc_means_, self.mcmc_covariances_, self.mcmc_weights_
        )
        profile["point_estimates"] = time.time() - pe_t0
        profile["fit_total"] = time.time() - fit_t0
        self.last_fit_profile_ = profile
        return self

    def _fit_epistemic_em_inits(
        self,
        X: np.ndarray,
        k: int,
        sample_weight: np.ndarray | None = None,
    ) -> list[dict[str, np.ndarray]]:
        """Loose sklearn EMs with distinct seeds; used only as uncentered NUTS inits.

        Scoring still uses the rigorous EM in ``fit``. ``n_init`` per start is
        ``epistemic_em_n_init`` (default 3 vs scoring's 30).
        """
        cfg = self.config
        dtype = self._dtype
        n_starts = int(getattr(cfg, "epistemic_em_inits", 4))
        if n_starts < 1:
            raise ValueError(f"epistemic_em_inits must be >= 1, got {n_starts}")
        n_init_loose = int(getattr(cfg, "epistemic_em_n_init", 3))
        if n_init_loose < 1:
            raise ValueError(f"epistemic_em_n_init must be >= 1, got {n_init_loose}")
        base_seed = int(cfg.random_state)
        inits: list[dict[str, np.ndarray]] = []
        for i in range(n_starts):
            em = WeightedGaussianMixture(
                n_components=int(k),
                covariance_type="diag",
                reg_covar=cfg.reg_covar,
                n_init=n_init_loose,
                max_iter=cfg.sklearn_max_iter,
                tol=cfg.sklearn_tol,
                random_state=base_seed + 1009 * (i + 1),
            ).fit(X, sample_weight=sample_weight)
            sigma = np.sqrt(np.maximum(em.covariances_, 1e-12)).astype(dtype)
            w = np.asarray(em.weights_, dtype=dtype)
            w = np.clip(w, 1e-6, None)
            w = (w / w.sum()).astype(dtype)
            inits.append(
                {
                    "mu": np.asarray(em.means_, dtype=dtype),
                    "sigma": sigma,
                    "w": w,
                }
            )
        return inits

    def _set_em_point_estimates(self, init_params: dict[str, np.ndarray]) -> None:
        """Set scoring params from EM / warm-start init (not posterior mean)."""
        dtype = self._dtype
        mu = np.asarray(init_params["mu"], dtype=dtype)
        weights = np.asarray(init_params["w"], dtype=dtype)
        self.means_ = mu
        self.weights_ = (weights / weights.sum()).astype(dtype)
        if "sigma" in init_params:
            sigma = np.asarray(init_params["sigma"], dtype=dtype)
            self.covariances_ = (
                np.square(np.maximum(sigma, 1e-6)) + dtype.type(self.reg_covar)
            ).astype(dtype)
        else:
            cov = np.asarray(init_params["cov"], dtype=dtype)
            if cov.ndim == 3:
                cov = np.diagonal(cov, axis1=-2, axis2=-1)
            self.covariances_ = (
                np.maximum(cov, 1e-12) + dtype.type(self.reg_covar)
            ).astype(dtype)

    def _build_multivariate_model(
        self,
        x: np.ndarray,
        k: int,
        init_params: dict[str, np.ndarray],
        sample_weight: np.ndarray | None = None,
    ) -> pm.Model:
        cfg = self.config
        dtype = self._dtype
        x = np.asarray(x, dtype=dtype)
        _, n_features = x.shape
        use_em_prior = bool(cfg.em_centered_prior) and not bool(
            getattr(cfg, "epistemic_multistart", False)
        )

        mu_em = np.asarray(init_params["mu"], dtype=dtype)
        w_em = np.asarray(init_params["w"], dtype=dtype)
        w_em = np.clip(w_em, 1e-6, None)
        w_em = (w_em / w_em.sum()).astype(dtype)
        sigma_em = np.asarray(init_params["sigma"], dtype=dtype)
        sigma_em = np.maximum(sigma_em, 1e-6).astype(dtype)

        with pm.Model() as model:
            if use_em_prior:
                mu = pm.Normal(
                    "mu",
                    mu=mu_em,
                    sigma=cfg.mu_prior_sigma,
                    shape=(k, n_features),
                    initval=mu_em,
                )
                w_alpha = np.maximum(
                    cfg.weight_prior_concentration * w_em,
                    1e-3,
                ).astype(dtype)
                weights = pm.Dirichlet("w", w_alpha, initval=w_em)
                # TruncatedNormal around EM scales (HalfNormal is always centered at 0).
                sigma = pm.TruncatedNormal(
                    "sigma",
                    mu=sigma_em,
                    sigma=cfg.sigma_prior_sigma,
                    lower=1e-6,
                    shape=(k, n_features),
                    initval=sigma_em,
                )
            else:
                mu = pm.Normal(
                    "mu",
                    mu=0,
                    sigma=cfg.mu_prior_sigma,
                    shape=(k, n_features),
                    initval=mu_em,
                )
                weights = pm.Dirichlet("w", np.ones(k, dtype=dtype), initval=w_em)
                sigma = pm.HalfNormal(
                    "sigma",
                    sigma=cfg.sigma_prior_sigma,
                    shape=(k, n_features),
                    initval=sigma_em,
                )
            sw_tensor = None
            if sample_weight is not None:
                sw = np.asarray(sample_weight, dtype=dtype).ravel()
                if sw.shape[0] != x.shape[0]:
                    raise ValueError(
                        f"sample_weight length {sw.shape[0]} != n_samples {x.shape[0]}"
                    )
                sw = np.clip(sw, 1e-12, None)
                mean = float(sw.mean()) if sw.size else 1.0
                if (not np.isfinite(mean)) or mean <= 0:
                    sw = np.ones(sw.shape[0], dtype=dtype)
                else:
                    sw = (sw / mean).astype(dtype)
                sw_tensor = pt.as_tensor_variable(sw)
            pm.Potential(
                "obs_loglik",
                _diagonal_gmm_loglik(
                    pt.as_tensor_variable(x),
                    mu,
                    sigma,
                    weights,
                    sample_weight=sw_tensor,
                ),
            )
        return model

    def _run_mcmc(self) -> dict[str, float]:
        cfg = self.config
        # Warm-start uses the same tune/draws as cold-start (defaults: 1000/1000).
        tune = cfg.mcmc_tune
        # BlackJAX vectorized sampling cannot use the progress bar (IO effect under vmap).
        use_progressbar = bool(cfg.verbose)
        chain_method = None
        if cfg.nuts_sampler_kwargs:
            chain_method = cfg.nuts_sampler_kwargs.get("chain_method")
        if cfg.nuts_sampler == "blackjax" and chain_method == "vectorized":
            use_progressbar = False

        n_chains = int(cfg.mcmc_chains)
        initvals = None
        epi = getattr(self, "epistemic_em_inits_", None)
        if bool(getattr(cfg, "epistemic_multistart", False)) and epi:
            n_chains = int(len(epi))
            initvals = [
                {
                    "mu": np.asarray(p["mu"]),
                    "w": np.asarray(p["w"]),
                    "sigma": np.asarray(p["sigma"]),
                }
                for p in epi
            ]

        if (
            cfg.nuts_sampler == "blackjax"
            and chain_method == "parallel"
            and cfg.jax_platform == "cpu"
        ):
            n_cores = os.cpu_count() or 1
            if n_chains > n_cores:
                raise ValueError(
                    f"BlackJAX parallel on CPU needs mcmc_chains <= CPU cores "
                    f"({n_chains} > {n_cores})."
                )

        sample_t0 = time.time()
        profile: dict[str, float] = {}
        sample_kwargs = {
            "draws": cfg.mcmc_draws,
            "tune": tune,
            "chains": n_chains,
            "random_seed": cfg.random_state,
            "progressbar": use_progressbar,
            "nuts_sampler": cfg.nuts_sampler,
            "target_accept": cfg.target_accept,
            "compute_convergence_checks": False,
        }
        if initvals is not None:
            sample_kwargs["initvals"] = initvals
        if cfg.nuts_sampler_kwargs:
            sample_kwargs["nuts_sampler_kwargs"] = cfg.nuts_sampler_kwargs
        with self._pymc_model:
            self.idata_ = pm.sample(**sample_kwargs)
        # Compile and sampling are not separable via pm.sample for JAX backends.
        profile["mcmc_compile"] = 0.0
        profile["mcmc_sample"] = time.time() - sample_t0
        profile["inference"] = "mcmc"
        return profile

    def _run_advi(self) -> dict[str, float]:
        """Fit JAX ADVI and draw independent posterior samples.

        Mean-field (``advi``) or full-rank (``fullrank_advi``) in unconstrained
        space, using the same jaxified logp as BlackJAX NUTS. Samples are stored
        in ``idata_`` so BALD / BatchBALD reuse the NUTS draw-wise LL path.
        """
        cfg = self.config
        n_iter = int(getattr(cfg, "advi_n_iter", 20000))
        if n_iter < 1:
            raise ValueError(f"advi_n_iter must be >= 1, got {n_iter}")
        n_unc = int(cfg.n_uncertainty_draws)
        n_draws = int(max(n_unc, int(getattr(cfg, "advi_draws", n_unc))))
        method = str(getattr(cfg, "advi_method", "advi"))
        if method not in ("advi", "fullrank_advi"):
            raise ValueError(
                f"advi_method must be 'advi' or 'fullrank_advi', got {method!r}"
            )
        lr = float(getattr(cfg, "advi_learning_rate", 0.01))
        if lr <= 0:
            raise ValueError(f"advi_learning_rate must be > 0, got {lr}")

        idata, approx, times = _fit_jax_advi(
            self._pymc_model,
            n_iter=n_iter,
            n_draws=n_draws,
            method=method,
            random_seed=int(cfg.random_state),
            learning_rate=lr,
        )
        self.idata_ = idata
        self.approx_ = approx
        if "chain" not in self.idata_.posterior.dims:
            self.idata_.posterior = self.idata_.posterior.expand_dims(chain=1)

        elbo = None
        hist = getattr(approx, "hist", None)
        if hist is not None and len(hist) > 0:
            elbo = float(np.asarray(hist).ravel()[-1])

        advi_fit = float(times["advi_fit"])
        advi_sample = float(times["advi_sample"])
        advi_compile = float(times["advi_compile"])
        profile: dict[str, float] = {
            "inference": "advi",
            "advi_method": method,
            "advi_backend": "jax",
            "advi_n_iter": float(n_iter),
            "advi_draws": float(n_draws),
            "advi_learning_rate": lr,
            "advi_compile": advi_compile,
            "advi_jaxify": float(times["advi_jaxify"]),
            "advi_fit": advi_fit,
            "advi_sample": advi_sample,
            "mcmc_compile": advi_compile,
            "mcmc_sample": advi_fit + advi_sample,
        }
        if elbo is not None:
            profile["advi_elbo"] = elbo
        return profile


    def _relabel_posterior_to_reference(self) -> None:
        """Align every posterior draw's component labels to the EM initialization."""
        if self.idata_ is None or self._reference_means_ is None:
            raise ValueError("Posterior samples and reference means are required for relabeling.")
        if self.n_components is None or self.n_features_in_ is None:
            raise ValueError("Model dimensions are required for relabeling.")

        mu_da = self.idata_.posterior["mu"]
        weight_da = self.idata_.posterior["w"]
        sigma_da = self.idata_.posterior["sigma"]
        mu_values = np.array(mu_da.values, copy=True)
        weight_values = np.array(weight_da.values, copy=True)
        sigma_values = np.array(sigma_da.values, copy=True)
        n_chains, n_draws = mu_values.shape[:2]
        means = mu_values.reshape(-1, *mu_values.shape[2:])
        weights = weight_values.reshape(-1, *weight_values.shape[2:])
        sigmas = sigma_values.reshape(-1, *sigma_values.shape[2:])
        means, sigmas, weights = self.relabel_params_to_reference(
            means, sigmas, weights
        )
        self.idata_.posterior["mu"] = (
            mu_da.dims,
            means.reshape(n_chains, n_draws, *means.shape[1:]),
        )
        self.idata_.posterior["w"] = (
            weight_da.dims,
            weights.reshape(n_chains, n_draws, *weights.shape[1:]),
        )
        self.idata_.posterior["sigma"] = (
            sigma_da.dims,
            sigmas.reshape(n_chains, n_draws, *sigmas.shape[1:]),
        )

    def _run_convergence_checks(self) -> None:
        """Run PyMC's default convergence diagnostics (R-hat threshold 1.01)."""
        from pymc.stats.convergence import log_warnings, run_convergence_checks

        log_warnings(run_convergence_checks(self.idata_, self._pymc_model))

    def _set_mcmc_point_estimates(self):
        """Store posterior-mean params for diagnostics; scoring still uses EM ``means_``."""
        dtype = self._dtype
        mu = np.asarray(self.idata_.posterior["mu"].mean(("chain", "draw")), dtype=dtype)
        weights = np.asarray(self.idata_.posterior["w"].mean(("chain", "draw")), dtype=dtype)
        sigma = np.asarray(self.idata_.posterior["sigma"].mean(("chain", "draw")), dtype=dtype)
        self.mcmc_means_ = mu
        self.mcmc_weights_ = (weights / weights.sum()).astype(dtype)
        self.mcmc_covariances_ = (
            np.square(np.maximum(sigma, 1e-6)) + dtype.type(self.reg_covar)
        ).astype(dtype)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        means = self.score_means_ if self.score_means_ is not None else self.means_
        covs = self.score_covariances_ if self.score_covariances_ is not None else self.covariances_
        weights = self.score_weights_ if self.score_weights_ is not None else self.weights_
        if means is None or covs is None or weights is None:
            raise ValueError("Model has not been fitted yet.")
        log_weights = np.log(weights + 1e-300)
        log_comp = _gauss_logpdf(X, means, covs)
        return logsumexp(log_weights[None, :] + log_comp, axis=1)

    def set_map_scoring_params(self, X: np.ndarray) -> dict:
        """Use the posterior draw with highest training log-likelihood for ``score_samples``.

        Searches all flattened MCMC/ADVI draws (typically 1000). Does not change
        EM ``means_`` used for warm-start.
        """
        if self.idata_ is None:
            raise ValueError("Posterior draws are required to set MAP scoring params.")
        X = np.asarray(X, dtype=self._dtype)
        if X.ndim == 1:
            X = X[:, None]
        mu, covs, weights = self._flattened_posterior_params()
        n_total = int(mu.shape[0])
        ll = np.asarray(
            self.predictive_log_likelihoods_over_draws(X, n_draws=n_total),
            dtype=float,
        )
        train_ll = np.sum(ll, axis=1)
        best = int(np.argmax(train_ll))
        dtype = self._dtype
        self.score_means_ = np.asarray(mu[best], dtype=dtype)
        cov = np.asarray(covs[best], dtype=dtype)
        if cov.ndim == 3:
            cov = np.diagonal(cov, axis1=-2, axis2=-1)
        self.score_covariances_ = cov
        w = np.asarray(weights[best], dtype=dtype)
        self.score_weights_ = (w / np.maximum(w.sum(), 1e-300)).astype(dtype)
        self.map_draw_index_ = best
        self.map_train_ll_ = float(train_ll[best])
        return {
            "map_draw_index": best,
            "map_train_ll": self.map_train_ll_,
            "n_draws": n_total,
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.means_ is None or self.covariances_ is None or self.weights_ is None:
            raise ValueError("Model has not been fitted yet.")
        return _mixture_responsibilities(X, self.means_, self.covariances_, self.weights_)

    def _flattened_posterior_params(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return flattened posterior draws as ``(means, covmats, weights)``.

        ``covmats`` have shape ``(n_draws, k, d, d)`` (diagonal embedded) for
        plot compatibility.
        """
        if self.idata_ is None or self.n_components is None or self.n_features_in_ is None:
            raise ValueError("Model has not been fitted yet.")
        mu = np.asarray(self.idata_.posterior["mu"].values)
        weights = np.asarray(self.idata_.posterior["w"].values)
        sigma = np.asarray(self.idata_.posterior["sigma"].values)
        mu = mu.reshape(-1, *mu.shape[2:])
        weights = weights.reshape(-1, *weights.shape[2:])
        sigma = sigma.reshape(-1, *sigma.shape[2:])
        variances = np.square(np.maximum(sigma, 1e-6)) + self.reg_covar
        n_d, k_d, d_d = variances.shape
        covs = np.zeros((n_d, k_d, d_d, d_d), dtype=variances.dtype)
        idx = np.arange(d_d)
        covs[:, :, idx, idx] = variances
        return mu, covs, weights

    def sample_posterior_params(self, n_draws: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return posterior parameter sets via thinning.

        Flattens ``(chain, draw)`` then subsamples. Default is evenly spaced
        ``n_uncertainty_draws``. If ``uncertainty_thin_stride`` is set and
        ``n_draws`` is omitted, keep every ``stride``-th draw instead.
        """
        mu, covs, weights = self._flattened_posterior_params()
        n_total = int(mu.shape[0])
        stride = getattr(self.config, "uncertainty_thin_stride", None)
        if n_draws is None and stride is not None and int(stride) > 1:
            stride = int(stride)
            indices = np.arange(0, n_total, stride, dtype=int)
            if indices.size < 1:
                raise ValueError(
                    f"thin stride={stride} produced no draws from n_total={n_total}."
                )
            return mu[indices], covs[indices], weights[indices]
        n_draws = self.config.n_uncertainty_draws if n_draws is None else int(n_draws)
        if n_draws < 1:
            raise ValueError(f"n_draws must be >= 1, got {n_draws}.")
        if n_draws > n_total:
            raise ValueError(
                f"Requested n_draws={n_draws} but only {n_total} posterior samples exist."
            )
        indices = _thin_indices(n_total, n_draws)
        return mu[indices], covs[indices], weights[indices]

    def relabel_params_to_reference(
        self,
        means: np.ndarray,
        cov_or_sigma: np.ndarray,
        weights: np.ndarray,
        reference: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        reference = self._reference_means_ if reference is None else reference
        if reference is None:
            raise ValueError("No reference means available for relabeling.")

        means_out = np.empty_like(means)
        cov_out = np.empty_like(cov_or_sigma)
        weights_out = np.empty_like(weights)
        for i in range(means.shape[0]):
            perm = _hungarian_permutation(means[i], reference)
            means_out[i] = means[i][perm]
            cov_out[i] = cov_or_sigma[i][perm]
            weights_out[i] = weights[i][perm]
        return means_out, cov_out, weights_out

    def predictive_log_likelihoods_over_draws(
        self,
        X: np.ndarray,
        n_draws: int | None = None,
    ) -> np.ndarray:
        """Predictive log-likelihood ``log p(x|θ)`` under ``n_draws`` posterior sets.

        Mixture likelihood is invariant to component label switching, so draws are
        not relabeled. Returns array of shape ``(n_draws, n_samples)``.
        """
        means, covs, weights = self.sample_posterior_params(n_draws=n_draws)

        X = np.asarray(X, dtype=self._dtype)
        if X.ndim == 1:
            X = X[:, None]
        return np.asarray(
            _diag_mixture_loglik_batched(X, means, covs, weights),
            dtype=self._dtype,
        )

    def get_init_params(self) -> dict[str, np.ndarray]:
        if self.means_ is None or self.weights_ is None or self.covariances_ is None:
            return {}
        dtype = self._dtype
        return {
            "mu": np.asarray(self.means_, dtype=dtype),
            "cov": np.asarray(self.covariances_, dtype=dtype),
            "w": np.asarray(self.weights_, dtype=dtype),
            "sigma": np.sqrt(
                np.maximum(self.covariances_ - self.reg_covar, 1e-12)
            ).astype(dtype),
        }

