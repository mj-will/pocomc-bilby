"""Example plugin for using a sampler in bilby.

Here we demonstrate the how to implement the class.
"""
import bilby
from bilby.core.sampler.base_sampler import MCMCSampler, NestedSampler
import numpy as np


class DemoSampler(NestedSampler):
    """Bilby wrapper for your sampler.

    This class should inherit from :code:`MCMCSampler` or :code:`NestedSampler`
    """

    @property
    def external_sampler_name(self) -> str:
        """The name of package that provides the sampler."""
        # In this template we do not require any external codes, so we just
        # use bilby. You should change this.
        return "bilby"

    @property
    def default_kwargs(self) -> dict:
        """Dictionary of default keyword arguments.

        Any arguments not included here will be removed before calling the
        sampler.
        """
        return dict(
            ninitial=100,
        )

    def run_sampler(self) -> dict:
        """Run the sampler.

        This method should run the sampler and update the result object.
        It should also return the result object.
        """

        # The code below shows how you can call different methods.
        # Replace this code with calls to your sampler

        # Keyword arguments are stored in self.kwargs
        prior_samples = np.array(
            list(self.priors.sample(self.kwargs["ninitial"]).values()),
        ).T
        # We can evaluate the log-prior and log-likelihood
        logl = np.empty(len(prior_samples))
        logp = np.empty(len(prior_samples))
        for i, sample in enumerate(prior_samples):
            logl[i] = self.log_likelihood(sample)
            logp[i] = self.log_prior(sample)

        # Generate posterior samples
        logw = logl.copy() - logl.max()
        keep = logw > np.log(np.random.rand(len(logw)))
        posterior_samples = prior_samples[keep]

        # The result object is created automatically
        # So we just have to populate the different methods
        # Add the posterior samples to the result object
        # This should be a numpy array of shape (# samples x # parameters)
        self.result.samples = posterior_samples
        # We can also store the log-likelihood and log-prior values for each
        # posterior sample
        self.result.log_likelihood_evaluations = logl[keep]
        self.result.log_prior_evaluations = logp[keep]
        # If it is a nested sampler, we can add the nested samples
        self.result.nested_samples = prior_samples
        # We can also add the log-evidence and the error
        # These can be NaNs for samplers that no not estimate the evidence
        self.result.log_evidence = np.mean(logl)
        self.result.log_evidence_err = np.std(logl)

        # Must return the result object
        return self.result
