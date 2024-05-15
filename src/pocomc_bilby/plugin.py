"""Example plugin for using a sampler in bilby.

Here we demonstrate the how to implement the class.
"""
import bilby
import inspect
import numpy as np
import os
import pocomc


class PocoMC(bilby.core.sampler.Sampler):
    """Wrapper for pocomc.

    See the documentation for details: https://pocomc.readthedocs.io/
    """
    sampler_name = "pocomc"

    @property
    def init_kwargs(self):
        params = inspect.signature(pocomc.Sampler).parameters
        kwargs = {
            key: param.default
            for key, param in params.items()
            if param.default != param.empty
        }
        kwargs["n_particles"] = 1000
        kwargs.pop("vectorize_likelihood")
        return kwargs

    @property
    def run_kwargs(self):
        params = inspect.signature(pocomc.Sampler.run).parameters
        kwargs = {
            key: param.default
            for key, param in params.items()
            if param.default != param.empty
        }
        kwargs.pop("prior_samples")
        return kwargs

    @property
    def default_kwargs(self):
        kwargs = self.init_kwargs
        kwargs.update(self.run_kwargs)
        kwargs["n_final"] = None
        return kwargs

    def run_sampler(self):

        n_final = self.kwargs.pop("n_final")

        init_kwargs = {k: self.kwargs.get(k) for k in self.init_kwargs.keys()}
        run_kwargs = {k: self.kwargs.get(k) for k in self.run_kwargs.keys()}

        sampler = pocomc.Sampler(
            n_dim=self.ndim,
            log_likelihood=self.log_likelihood,
            log_prior=self.log_prior,
            vectorize_likelihood=False,
            **init_kwargs,
        )
        prior_samples = self.priors.sample(self.kwargs["n_particles"])
        prior_samples = np.array(
            [prior_samples[key] for key in self.search_parameter_keys]
        ).T

        sampler.run(prior_samples, **run_kwargs)
        if n_final is not None:
            sampler.add_samples(n_final - self.kwargs["n_particles"])

        results = sampler.results

        if self.plot:
            fig = pocomc.plotting.run(sampler.results)
            fig.savefig(os.path.join(self.outdir, "pocomc_plot.png"))

        self.result.samples = results["samples"]
        self.result.log_likelihood_evaluations = results["loglikelihood"]
        self.result.log_prior_evaluations = results["logprior"]
        self.result.log_evidence = results["logz"][-1]

        return self.result

    @classmethod
    def get_expected_outputs(cls, outdir=None, label=None):
        filenames = [
            "pocomc_plot.png",
        ]
        return filenames, []
