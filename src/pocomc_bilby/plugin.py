import bilby
import inspect
import os
import pocomc

from .prior import PriorWrapper


def _log_likelihood_wrapper(theta):
    """Wrapper to the log likelihood. Needed for multiprocessing."""
    from bilby.core.sampler.base_sampler import _sampling_convenience_dump

    if _sampling_convenience_dump.use_ratio:
        return _sampling_convenience_dump.likelihood.log_likelihood_ratio()
    else:
        return _sampling_convenience_dump.likelihood.log_likelihood()


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
        kwargs["n_active"] = 1000
        not_allowed = [
            "vectorize",
            "output_dir",
            "output_label",
            "n_dim",
            "pool",
        ]
        for key in not_allowed:
            kwargs.pop(key)
        return kwargs

    @property
    def run_kwargs(self):
        params = inspect.signature(pocomc.Sampler.run).parameters
        kwargs = {
            key: param.default
            for key, param in params.items()
            if param.default != param.empty
        }
        return kwargs

    @property
    def default_kwargs(self):
        kwargs = self.init_kwargs
        kwargs.update(self.run_kwargs)
        return kwargs

    def run_sampler(self):

        init_kwargs = {k: self.kwargs.get(k) for k in self.init_kwargs.keys()}
        run_kwargs = {k: self.kwargs.get(k) for k in self.run_kwargs.keys()}

        prior = PriorWrapper(self.priors, self.search_parameter_keys)

        output_dir = os.path.join(
            self.outdir, f"{self.sampler_name}_{self.label}", "",
        )

        self._setup_pool()

        sampler = pocomc.Sampler(
            prior=prior,
            likelihood=_log_likelihood_wrapper,
            vectorize=False,
            output_label=self.label,
            output_dir=output_dir,
            n_dim=self.ndim,
            pool=self.pool,
            **init_kwargs,
        )

        sampler.run(**run_kwargs)

        samples, weights, logl, logp = sampler.posterior()
        logz, logz_err = sampler.evidence()

        posterior_samples = bilby.core.result.rejection_sample(
            samples, weights
        )

        self.result.samples = posterior_samples
        self.result.log_evidence = logz
        self.result.log_evidence_error = logz_err
        return self.result
