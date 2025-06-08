"""Debug script for loo_moment_match based on the PyMC vignette."""

import logging
import os
import tempfile

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from cmdstanpy import CmdStanModel
from scipy import stats

from arviz_stats.loo import loo, loo_moment_match

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

STAN_CODE = """
data {
  int<lower=1> K;
  int<lower=1> N;
  matrix[N,K] x;
  array[N] int y;
  vector[N] log_exposure;

  real beta_prior_scale;
  real alpha_prior_scale;
}
parameters {
  vector[K] beta;
  real intercept;
}
model {
  y ~ poisson(exp(x * beta + intercept + log_exposure));
  beta ~ normal(0, beta_prior_scale);
  intercept ~ normal(0, alpha_prior_scale);
}
generated quantities {
  vector[N] log_lik;
  for (n in 1:N)
    log_lik[n] = poisson_lpmf(y[n] | exp(x[n] * beta + intercept + log_exposure[n]));
}
"""

roaches = pd.read_csv("roaches.csv")
roaches["roach1"] = np.sqrt(roaches["roach1"])

y = roaches["y"].values
X = roaches[["roach1", "treatment", "senior"]].values
log_exposure = np.log(roaches["exposure2"].values)

n, k = X.shape

stan_data = {
    "N": n,
    "K": k,
    "x": X,
    "y": y.astype(int),
    "log_exposure": log_exposure,
    "beta_prior_scale": 2.5,
    "alpha_prior_scale": 5.0,
}

with tempfile.NamedTemporaryFile(mode="w", suffix=".stan", delete=False) as f:
    f.write(STAN_CODE)
    stan_file = f.name

try:
    model = CmdStanModel(stan_file=stan_file)

    fit = model.sample(
        data=stan_data,
        chains=4,
        iter_warmup=1000,
        iter_sampling=1000,
        seed=9547,
        show_progress=True,
        save_warmup=False,
    )

    idata = az.from_cmdstanpy(
        posterior=fit, log_likelihood={"y_obs": "log_lik"}, observed_data={"y": y}
    )

finally:
    os.unlink(stan_file)

loo_orig = loo(idata, var_name="y_obs", pointwise=True)
# print("\nOriginal LOO result:")
# print(f"ELPD: {loo_orig.elpd:.1f}, p_loo: {loo_orig.p:.1f}")
# print(f"Observations with k > 0.7: {np.sum(loo_orig.pareto_k.values > 0.7)}")

posterior = idata.posterior

# print("\nDimensions of posterior variables:")
# print(f"intercept dims: {posterior['intercept'].dims}")
# print(f"beta dims: {posterior['beta'].dims}")

intercept_vals = posterior["intercept"].values[..., np.newaxis]
beta_vals = posterior["beta"].values

upars_vals = np.concatenate([intercept_vals, beta_vals], axis=-1)

upars = xr.DataArray(
    upars_vals,
    dims=["chain", "draw", "param"],
    coords={
        "chain": posterior.chain,
        "draw": posterior.draw,
        "param": np.arange(upars_vals.shape[-1]),
    },
)

# print(f"\nupars dimensions: {upars.dims}")
# print(f"upars shape: {upars.shape}")


def log_prob_upars(upars_da):
    """Use CmdStanPy's log_prob method for accurate calculation."""
    n_chains, n_draws, _ = upars_da.shape
    logp_values = np.empty((n_chains, n_draws))

    for chain_idx in range(n_chains):
        for draw_idx in range(n_draws):
            params = upars_da.isel(chain=chain_idx, draw=draw_idx).values

            intercept = params[0]
            beta = params[1:]

            eta = X @ beta + intercept + log_exposure
            mu = np.exp(eta)
            log_lik = np.sum(stats.poisson.logpmf(y, mu))

            log_prior_intercept = stats.norm.logpdf(intercept, 0, 5.0)
            log_prior_beta = np.sum(stats.norm.logpdf(beta, 0, 2.5))

            logp_values[chain_idx, draw_idx] = log_lik + log_prior_intercept + log_prior_beta

    return xr.DataArray(
        logp_values,
        dims=["chain", "draw"],
        coords={"chain": upars_da.coords["chain"], "draw": upars_da.coords["draw"]},
    )


def log_lik_i_upars(upars_da, i):
    """Compute log likelihood for observation i."""
    intercept_values = upars_da.isel(param=0).values
    beta_values = upars_da.isel(param=slice(1, None)).values

    eta_i = intercept_values + np.dot(beta_values, X[i]) + log_exposure[i]
    mu_i = np.exp(eta_i)

    loglik_values = stats.poisson.logpmf(y[i], mu_i)

    return xr.DataArray(
        loglik_values,
        dims=["chain", "draw"],
        coords={"chain": upars_da.coords["chain"], "draw": upars_da.coords["draw"]},
    )


loo_mm_default = loo_moment_match(
    idata,
    loo_orig,
    upars=upars,
    log_prob_upars_fn=log_prob_upars,
    log_lik_i_upars_fn=log_lik_i_upars,
    var_name="y_obs",
    max_iters=30,
    split=True,
    cov=True,
    pointwise=True,
)

# print("\nMoment matching result:")
# print(f"ELPD: {loo_mm_default.elpd:.1f}, p_loo: {loo_mm_default.p:.1f}")
# print(f"Observations with k > 0.7: {np.sum(loo_mm_default.pareto_k.values > 0.7)}")

# print("\n" + "=" * 60)
# print("DETAILED COMPARISON WITH R VIGNETTE RESULTS")
# print("=" * 60)

# print("\nExpected from R vignette:")
# print("  Original ELPD: -5457.8")
# print("  After MM ELPD: -5478.5")
# print("  Original high k: 19 observations")
# print("  After MM high k: 0 observations")

# print("\nOur results:")
# print(f"  Original ELPD: {loo_orig.elpd:.1f}")
# print(f"  After MM ELPD: {loo_mm_default.elpd:.1f}")
# print(f"  Original high k: {np.sum(loo_orig.pareto_k.values > 0.7)}")
# print(f"  After MM high k: {np.sum(loo_mm_default.pareto_k.values > 0.7)}")

# high_k_orig = np.where(loo_orig.pareto_k.values > 0.7)[0]
# if len(high_k_orig) > 0:
#     print(f"\nProblematic observation indices: {high_k_orig}")
#     print("\nOriginal vs MM Pareto k values for these observations:")
#     for idx in high_k_orig[:5]:
#         orig_k = loo_orig.pareto_k.values[idx]
#         mm_k = loo_mm_default.pareto_k.values[idx]
#         print(f"  Obs {idx}: {orig_k:.3f} -> {mm_k:.3f}")

# if hasattr(loo_mm_default, "p_loo_i") and loo_mm_default.p_loo_i is not None:
#     nan_count = np.isnan(loo_mm_default.p_loo_i.values).sum()
#     neg_count = (loo_mm_default.p_loo_i.values < 0).sum()
#     print("\nDiagnostics:")
#     print(f"  NaN values in p_loo_i: {nan_count}")
#     print(f"  Negative values in p_loo_i: {neg_count}")

# print(loo_mm_default.p_loo_i)
