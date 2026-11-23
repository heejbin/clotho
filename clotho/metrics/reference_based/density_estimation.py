
from sklearn.utils.validation import check_is_fitted
from scipy.stats import entropy, rankdata
import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

from clotho.metrics.reference_based.pymc_gmm import (
    PyMCGMMConfig,
    PyMCGaussianMixture,
    WeightedGaussianMixture,
)


def _align_sample_weight(sample_weight, labels, n_train):
    """Slice IPW to GMM training rows (``labels==1``) and mean-normalize.

    Mean-normalization keeps total mass = ``n_train`` so EM / MCMC priors
    stay on the same scale as the unweighted model.
    """
    if sample_weight is None:
        return None
    w = np.asarray(sample_weight, dtype=float).ravel()
    if labels is None:
        if w.shape[0] != n_train:
            raise ValueError(
                f"sample_weight length {w.shape[0]} != n_samples {n_train}"
            )
    else:
        labels = np.asarray(labels)
        if w.shape[0] != labels.shape[0]:
            raise ValueError(
                f"sample_weight length {w.shape[0]} != labels length {labels.shape[0]}"
            )
        w = w[labels == 1]
        if w.shape[0] != n_train:
            raise ValueError(
                f"sample_weight after label filter has length {w.shape[0]} != n_train {n_train}"
            )
    w = np.clip(w, 1e-12, None)
    mean = float(w.mean()) if w.size else 1.0
    if (not np.isfinite(mean)) or mean <= 0:
        return np.ones(w.shape[0], dtype=float)
    return (w / mean).astype(float)


class DensityEstimator(ABC):
    @abstractmethod
    def fit(self, ref_set, labels=None):
        """Fit the density estimator to the reference set."""
        pass
    
    @abstractmethod
    def score(self, target_vectors):
        """Compute the density score for the target vectors."""
        pass
    
    @abstractmethod
    def rank(self, target_vectors):
        """Rank the target vectors based on their density scores."""
        pass

@dataclass
class GMMScorerConfig:
    n_components: int = 10
    random_state: int = 42
    warm_start: bool = False
    verbose: bool = False
    max_iter: int = 500
    reg_covar: float = 1e-6
    tol: float = 1e-3
    n_init: int = 30
    covariance_type: str = "diag"


@dataclass
class MCMCGMMScorerConfig:
    """Config for PyMC/BlackJAX MCMC GMM scorer with fixed ``n_components``."""

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
    inference: Literal["mcmc", "advi"] = "mcmc"
    advi_method: Literal["advi", "fullrank_advi"] = "advi"
    advi_n_iter: int = 20000
    advi_learning_rate: float = 0.01
    advi_draws: int = 200
    mu_prior_sigma: float = 1.0
    sigma_prior_sigma: float = 1.5
    em_centered_prior: bool = True
    weight_prior_concentration: float = 10.0
    epistemic_multistart: bool = False
    epistemic_em_inits: int = 4
    epistemic_em_n_init: int = 3
    floatx: Literal["float32", "float64"] = "float32"
    jax_platform: str = "cpu"
    disable_xla_preallocate: bool = True
    # How many thinned posterior draws to use for BALD / uncertainty scoring.
    n_uncertainty_draws: int = 100
    uncertainty_thin_stride: int | None = None
    verbose: bool = False


class GMMScorer(DensityEstimator):
    """Original Clotho sklearn GMM density scorer."""

    def __init__(self, config: GMMScorerConfig | None = None):
        config = config or GMMScorerConfig()
        self.config = config
        self.n_components = config.n_components
        self.model = None
        verbose_option = 2 if config.verbose else 0
        self.model = WeightedGaussianMixture(
            n_components=config.n_components,
            covariance_type=config.covariance_type,
            random_state=config.random_state,
            warm_start=config.warm_start,
            verbose=verbose_option,
            verbose_interval=1,
            max_iter=config.max_iter,
            reg_covar=config.reg_covar,
            tol=config.tol,
            n_init=config.n_init,
        )

    def fit(self, ref_set, labels=None, *, sample_weight=None):
        """Fit the GMM to the embeddings of the reference test cases. The embeddings should be reduced to a lower dimension."""
        labels_for_weight = None if labels is None else np.asarray(labels)
        if labels is None:
            _ref_set_effective = ref_set
        elif all(label in [0, 1] for label in labels):
            labels = np.asarray(labels)
            _ref_set_effective = ref_set[labels == 1]
        else:
            raise ValueError("Labels should be binary (0 or 1) if provided.")

        _ref_set_effective = np.asarray(_ref_set_effective)
        sw = _align_sample_weight(sample_weight, labels_for_weight, len(_ref_set_effective))
        self.model.fit(_ref_set_effective, sample_weight=sw)

    def score(self, target_vectors):
        """Compute the density score for the target vectors."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        
        logprobs = self.model.score_samples(np.asarray(target_vectors))
        return logprobs

    def predict_proba(self, target_vectors):
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        return self.model.predict_proba(np.asarray(target_vectors))
    
    def rank(self, target_vectors, normalize=True):
        """Rank the target vectors based on their density scores."""
        scores = self.score(target_vectors)
        ranks = rankdata(scores, method='average')
        if normalize:
            ranks = (ranks - 1) / (len(ranks) - 1)  # Normalize ranks to [0, 1]
        return ranks
    
    def check_is_fitted(self):
        if self.model is None:
            return False
        try:
            check_is_fitted(self.model)
        except Exception:
            return False
        return True

    def _n_features(self):
        return self.model.n_features_in_
    
    def _n_components(self):
        return self.model.n_components
    
    def get_uncertainties(self, target_vectors):
        """Get uncertainties for the target vectors based on the model's component assignment entropy."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        
        probs = self.predict_proba(target_vectors)
        return entropy(probs, axis=1)


class MCMCGMMScorer(DensityEstimator):
    """PyMC GMM scorer used by ``scoring_model='MCMCGMM'`` / ``'ADVIGMM'``.

    ``n_components`` is fixed in config; the experiment loop may update it via the
    weight-perplexity heuristic on mixture weights. Posterior draws come from
    NUTS (``inference='mcmc'``) or JAX ADVI (``inference='advi'``).
    """

    def __init__(self, config: MCMCGMMScorerConfig | None = None):
        self.config = config or MCMCGMMScorerConfig()
        self.model = PyMCGaussianMixture(
            PyMCGMMConfig(
                n_components=self.config.n_components,
                random_state=self.config.random_state,
                reg_covar=self.config.reg_covar,
                sklearn_n_init=self.config.sklearn_n_init,
                sklearn_max_iter=self.config.sklearn_max_iter,
                sklearn_tol=self.config.sklearn_tol,
                mcmc_draws=self.config.mcmc_draws,
                mcmc_tune=self.config.mcmc_tune,
                mcmc_chains=self.config.mcmc_chains,
                nuts_sampler=self.config.nuts_sampler,
                nuts_sampler_kwargs=dict(self.config.nuts_sampler_kwargs),
                target_accept=self.config.target_accept,
                inference=self.config.inference,
                advi_method=self.config.advi_method,
                advi_n_iter=self.config.advi_n_iter,
                advi_learning_rate=self.config.advi_learning_rate,
                advi_draws=self.config.advi_draws,
                mu_prior_sigma=self.config.mu_prior_sigma,
                sigma_prior_sigma=self.config.sigma_prior_sigma,
                em_centered_prior=self.config.em_centered_prior,
                weight_prior_concentration=self.config.weight_prior_concentration,
                epistemic_multistart=self.config.epistemic_multistart,
                epistemic_em_inits=self.config.epistemic_em_inits,
                epistemic_em_n_init=self.config.epistemic_em_n_init,
                floatx=self.config.floatx,
                jax_platform=self.config.jax_platform,
                disable_xla_preallocate=self.config.disable_xla_preallocate,
                n_uncertainty_draws=self.config.n_uncertainty_draws,
                uncertainty_thin_stride=self.config.uncertainty_thin_stride,
                verbose=self.config.verbose,
            )
        )
        self.n_components = self.config.n_components
        self.feature_mean_: np.ndarray | None = None
        self.feature_scale_: np.ndarray | None = None

    @property
    def _dtype(self) -> np.dtype:
        return np.dtype(np.float32 if self.config.floatx == "float32" else np.float64)

    def _fit_feature_scaler(self, X: np.ndarray) -> None:
        dtype = self._dtype
        X = np.asarray(X, dtype=dtype)
        if X.ndim == 1:
            X = X[:, None]
        self.feature_mean_ = np.zeros(X.shape[1], dtype=dtype)
        self.feature_scale_ = np.ones(X.shape[1], dtype=dtype)

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        dtype = self._dtype
        X = np.asarray(X, dtype=dtype)
        if X.ndim == 1:
            X = X[:, None]
        return X

    def inverse_transform_gmm_params(
        self,
        means: np.ndarray,
        covariances: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Identity map (no feature standardization for diagonal GMM)."""
        dtype = self._dtype
        return np.asarray(means, dtype=dtype), np.asarray(covariances, dtype=dtype)


    def fit(self, ref_set, labels=None, init_params=None, *, warm_start: bool = False, run_mcmc: bool = True, sample_weight=None):
        labels_for_weight = None if labels is None else np.asarray(labels)
        if labels is None:
            _ref_set_effective = ref_set
        elif all(label in [0, 1] for label in labels):
            labels = np.asarray(labels)
            _ref_set_effective = np.asarray(ref_set)[labels == 1]
        else:
            raise ValueError("Labels should be binary (0 or 1) if provided.")

        _ref_set_effective = np.asarray(_ref_set_effective, dtype=self._dtype)
        self._fit_feature_scaler(_ref_set_effective)
        X_fit = self.transform_features(_ref_set_effective)
        sw = _align_sample_weight(sample_weight, labels_for_weight, len(_ref_set_effective))
        # init_params (if any) are assumed to already live in model/feature space.
        self.model.fit(
            X_fit,
            init_params=init_params,
            warm_start=warm_start,
            run_mcmc=run_mcmc,
            sample_weight=sw,
        )
        self.n_components = self.model.n_components
        self.last_fit_profile_ = getattr(self.model, "last_fit_profile_", None)
        return self

    def has_mcmc_posterior(self) -> bool:
        return getattr(self.model, "idata_", None) is not None

    def get_init_params(self) -> dict:
        return self.model.get_init_params()

    def get_fit_profile(self) -> dict | None:
        return getattr(self, "last_fit_profile_", None) or getattr(
            self.model, "last_fit_profile_", None
        )

    def score(self, target_vectors):
        if self.model is None or self.model.weights_ is None:
            raise ValueError("Model has not been fitted yet.")
        return self.model.score_samples(self.transform_features(target_vectors))

    def set_map_scoring_params(self, target_vectors) -> dict:
        """Score with the highest-likelihood MCMC draw on ``target_vectors``."""
        if not self.has_mcmc_posterior():
            raise ValueError("Posterior draws are required to set MAP scoring params.")
        return self.model.set_map_scoring_params(self.transform_features(target_vectors))

    def predict_proba(self, target_vectors):
        if self.model is None or self.model.weights_ is None:
            raise ValueError("Model has not been fitted yet.")
        return self.model.predict_proba(self.transform_features(target_vectors))

    def rank(self, target_vectors, normalize=True):
        scores = self.score(target_vectors)
        ranks = rankdata(scores, method="average")
        if normalize:
            ranks = (ranks - 1) / (len(ranks) - 1)
        return ranks

    def check_is_fitted(self):
        return self.model is not None and self.model.weights_ is not None

    def _n_features(self):
        return self.model.n_features_in_

    def _n_components(self):
        return self.model.n_components

    def get_uncertainties(self, target_vectors):
        """Epistemic uncertainty as Var[p(x|θ)] over posterior parameter draws."""
        if not self.check_is_fitted():
            raise ValueError("Model has not been fitted yet.")
        if not self.has_mcmc_posterior():
            raise ValueError(
                "Posterior draws are required for predictive-variance uncertainty. "
                "Call fit(..., run_mcmc=True) first."
            )
        n_draws = (
            None
            if getattr(self.config, "uncertainty_thin_stride", None)
            else self.config.n_uncertainty_draws
        )
        logp = self.model.predictive_log_likelihoods_over_draws(
            self.transform_features(target_vectors),
            n_draws=n_draws,
        )
        return np.var(np.exp(logp), axis=0)

    def predictive_log_likelihoods_over_draws(self, target_vectors, n_draws=None):
        """``log p(x|θ)`` under posterior draws; shape ``(n_draws, n_samples)``."""
        if not self.check_is_fitted():
            raise ValueError("Model has not been fitted yet.")
        if not self.has_mcmc_posterior():
            raise ValueError(
                "Posterior draws are required for predictive log-likelihoods over draws. "
                "Call fit(..., run_mcmc=True) first."
            )
        return self.model.predictive_log_likelihoods_over_draws(
            self.transform_features(target_vectors),
            n_draws=self.config.n_uncertainty_draws if n_draws is None else n_draws,
        )

def build_sklearn_gmm_for_cv(
    n_components: int,
    random_state: int = 42,
    reg_covar: float = 1e-6,
    n_init: int = 10,
    covariance_type: str = "diag",
):
    """Cheap sklearn GMM used inside PCA dimension CV for MCMCGMM."""
    return GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        reg_covar=reg_covar,
        n_init=n_init,
    )


def get_highest_likelihood_gmm_params(
    scorer,
    X_ref: np.ndarray,
    posterior_params: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(means, covariances, weights)`` for the highest-likelihood parameter set.

    - sklearn / point-estimate GMM: fitted MLE parameters.
    - MCMCGMM with posterior samples: posterior draw maximizing train log-likelihood.
    Covariances are always returned as ``(k, d, d)`` full matrices for plotting.

    ``posterior_params`` may supply a precomputed ``(means_all, covs_all, weights_all)``
    tuple from ``model._flattened_posterior_params()`` to avoid recomputing it.
    """
    from scipy.special import logsumexp
    from clotho.metrics.reference_based.pymc_gmm import _gauss_logpdf

    model = scorer.model
    X_ref = np.asarray(X_ref, dtype=float)
    if X_ref.ndim == 1:
        X_ref = X_ref[:, None]

    if posterior_params is not None:
        means_all, covs_all, weights_all = posterior_params
    elif getattr(model, "idata_", None) is not None:
        means_all, covs_all, weights_all = model._flattened_posterior_params()
    else:
        means_all = covs_all = weights_all = None

    if means_all is not None:
        best_ll = -np.inf
        best_idx = 0
        for i in range(means_all.shape[0]):
            log_weights = np.log(np.asarray(weights_all[i]) + 1e-300)
            log_comp = _gauss_logpdf(X_ref, means_all[i], covs_all[i], diag=False)
            ll = float(logsumexp(log_weights[None, :] + log_comp, axis=1).sum())
            if ll > best_ll:
                best_ll = ll
                best_idx = i
        means = np.asarray(means_all[best_idx], dtype=float)
        covariances = np.asarray(covs_all[best_idx], dtype=float)
        weights = np.asarray(weights_all[best_idx], dtype=float)
        return means, covariances, weights / weights.sum()

    means = np.asarray(model.means_, dtype=float)
    weights = np.asarray(model.weights_, dtype=float)
    cov = np.asarray(model.covariances_, dtype=float)
    if cov.ndim == 2:
        covariances = np.array([np.diag(v) for v in cov])
    elif cov.ndim == 3:
        covariances = cov
    else:
        raise ValueError(f"Unsupported covariances_ shape: {cov.shape}")
    return means, covariances, weights / weights.sum()
