import bilby
import numpy as np
import pytest

from pocomc_bilby.prior import PriorWrapper


def model(x, m, c):
    return m * x + c


@pytest.fixture()
def bilby_likelihood():
    bilby.core.utils.random.seed(42)
    rng = bilby.core.utils.random.rng
    x = np.linspace(0, 10, 100)
    injection_parameters = dict(m=0.5, c=0.2)
    sigma = 0.1
    y = model(x, **injection_parameters) + rng.normal(0.0, sigma, len(x))
    likelihood = bilby.likelihood.GaussianLikelihood(x, y, model, sigma)
    return likelihood


@pytest.fixture()
def bilby_priors():
    priors = bilby.core.prior.PriorDict()
    priors["m"] = bilby.core.prior.Uniform(0, 5, boundary="periodic")
    priors["c"] = bilby.core.prior.Uniform(-2, 2, boundary="reflective")
    return priors


@pytest.fixture()
def sampler_kwargs():
    return dict(
        n_active=1000,
    )


@pytest.mark.parametrize("n", [1, 10])
def test_prior(bilby_priors, n):

    prior = PriorWrapper(bilby_priors, bilby_priors.non_fixed_keys)

    x = prior.rvs(n)
    assert x.shape == (n, 2)

    log_prob = prior.logpdf(x)
    assert len(log_prob) == n


def test_run_sampler(bilby_likelihood, bilby_priors, tmp_path, sampler_kwargs):
    outdir = tmp_path / "test_run_sampler"

    bilby.run_sampler(
        likelihood=bilby_likelihood,
        priors=bilby_priors,
        sampler="pocomc",
        outdir=outdir,
        **sampler_kwargs,
    )
