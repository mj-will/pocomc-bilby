import bilby
import numpy as np
import pytest


def model(x, m, c):
    return m * x + c


@pytest.fixture()
def bilby_likelihood():
    bilby.core.utils.random.seed(42)
    rng = bilby.core.utils.random.rng
    x = np.linspace(0, 1, 11)
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
    # Any keyword arguments that need to be set of you want to test
    return dict(
        ninitial=100,
    )


def test_run_sampler(bilby_likelihood, bilby_priors, tmp_path, sampler_kwargs):
    outdir = tmp_path / "test_run_sampler"

    bilby.run_sampler(
        likelihood=bilby_likelihood,
        priors=bilby_priors,
        sampler="demo_sampler",    # This should match the name of the sampler
        outdir=outdir,
    )
