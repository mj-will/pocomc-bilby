"""Example plugin for using a sampler in bilby.

Here we demonstrate the how to implement the class.
"""
import bilby
import numpy as np
import os
import pocomc


class PocoMC(bilby.core.sampler.Sampler):
    """Wrapper for pocomc.

    See the documentation for details: https://pocomc.readthedocs.io/
    """
    default_kwargs = dict(
        n_particles=1000,
        threshold=1.0,
        periodic=None,
        flow_config=None,
        train_config=None,
        n_final=None
    )

    def run_sampler(self):

        n_final = self.kwargs.pop("n_final")

        sampler = pocomc.Sampler(
            n_dim=self.ndim,
            log_likelihood=self.log_likelihood,
            log_prior=self.log_prior,
            vectorize_likelihood=False,
            **self.kwargs
        )
        prior_samples = self.priors.sample(self.kwargs["n_particles"])
        prior_samples = np.array([prior_samples[key] for key in self.search_parameter_keys]).T

        sampler.run(prior_samples)
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
