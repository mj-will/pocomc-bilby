import bilby
from bilby.core.utils.log import logger
import inspect
import numpy as np
import os
from pathlib import Path
import pocomc

from .prior import PriorWrapper


def _log_likelihood_wrapper(theta):
    """Wrapper to the log likelihood. Needed for multiprocessing."""
    from bilby.core.sampler.base_sampler import _sampling_convenience_dump

    theta = {
        key: theta[ii]
        for ii, key in enumerate(_sampling_convenience_dump.search_parameter_keys)
    }

    _sampling_convenience_dump.likelihood.parameters.update(theta)

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
        kwargs["save_every"] = 5
        return kwargs

    @property
    def default_kwargs(self):
        kwargs = self.init_kwargs
        kwargs.update(self.run_kwargs)
        kwargs["resume"] = True
        return kwargs

    def run_sampler(self):

        init_kwargs = {k: self.kwargs.get(k) for k in self.init_kwargs.keys()}
        run_kwargs = {k: self.kwargs.get(k) for k in self.run_kwargs.keys()}

        prior = PriorWrapper(self.priors, self.search_parameter_keys)

        output_dir = \
            Path(self.outdir) / f"{self.sampler_name}_{self.label}" / ""
        os.makedirs(output_dir, exist_ok=True)

        self._setup_pool()
        pool = self.kwargs.pop("pool", None)
        resume = self.kwargs.pop("resume", False)

        sampler = pocomc.Sampler(
            prior=prior,
            likelihood=_log_likelihood_wrapper,
            vectorize=False,
            output_label=self.label,
            output_dir=output_dir,
            n_dim=self.ndim,
            pool=pool,
            **init_kwargs,
        )

        if resume and run_kwargs["resume_state_path"] is None:
            files = output_dir.glob("*.state")
            t_values = [int(file.stem.split("_")[-1]) for file in files]
            if len(t_values):
                t_max = max(t_values)
                state_path = output_dir / f"{self.label}_{t_max}.state"
                logger.info(f"Resuming pocomc from: {state_path}")
                run_kwargs["resume_state_path"] = state_path
            else:
                logger.debug("No files to resume from")

        sampler.run(**run_kwargs)

        samples, weights, logl, logp = sampler.posterior()
        logz, logz_err = sampler.evidence()

        posterior_samples = bilby.core.result.rejection_sample(
            samples, weights
        )

        self.result.samples = posterior_samples
        self.result.log_evidence = logz
        self.result.log_evidence_error = logz_err
        self.result.num_likelihood_evaluations = sampler.results["calls"][-1]
        return self.result