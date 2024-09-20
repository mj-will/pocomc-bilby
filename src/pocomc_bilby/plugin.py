import bilby
from bilby.core.utils.log import logger
import inspect
import numpy as np
from pathlib import Path
import pocomc

from .prior import PriorWrapper


def _log_likelihood_wrapper(theta):
    """Wrapper to the log likelihood.

    Does not evaluate the prior constraints.

    Needed for multiprocessing.
    """
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


def _log_likelihood_wrapper_with_constraints(theta):
    """Wrapper to the log likelihood that evaluates the prior constraints.

    Needed for multiprocessing."""
    from bilby.core.sampler.base_sampler import _sampling_convenience_dump

    theta = {
        key: theta[ii]
        for ii, key in enumerate(_sampling_convenience_dump.search_parameter_keys)
    }

    if not _sampling_convenience_dump.priors.evaluate_constraints(theta):
        return -np.inf
    _sampling_convenience_dump.likelihood.parameters.update(theta)

    if _sampling_convenience_dump.use_ratio:
        return _sampling_convenience_dump.likelihood.log_likelihood_ratio()
    else:
        return _sampling_convenience_dump.likelihood.log_likelihood()


class PocoMC(bilby.core.sampler.Sampler):
    """Wrapper for pocomc.

    See the documentation for details: https://pocomc.readthedocs.io/

    Outputs from the sampler will be saved in :code:`<outdir>/pocomc_<label>/.

    This implementation includes an additional option,
    :code:`evaluate_constraints_in_prior`, that determines if the prior
    prior constraints are evaluated when computing the log-likelihood
    (:code:`False`) or when evaluating the log-prior(:code:`True`).

    Some settings are automatically set based on the the bilby likelihood and
    prior that are provided.

    Supports multiprocessing via the bilby-supplied pool.
    """
    sampler_name = "pocomc"

    sampling_seed_key = "random_state"

    @property
    def init_kwargs(self):
        params = inspect.signature(pocomc.Sampler).parameters
        kwargs = {
            key: param.default
            for key, param in params.items()
            if param.default != param.empty
        }
        not_allowed = [
            "vectorize",
            "output_dir",
            "output_label",
            "n_dim",
            "pool",
            "reflective",  # Set automatically
            "periodic",    # Set automatically
        ]
        for key in not_allowed:
            kwargs.pop(key)
        kwargs["evaluate_constraints_in_prior"] = True
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
        kwargs["npool"] = None
        return kwargs

    def _translate_kwargs(self, kwargs):
        """Translate the keyword arguments"""
        if "npool" not in kwargs:
            for equiv in self.npool_equiv_kwargs:
                if equiv in kwargs:
                    kwargs["npool"] = kwargs.pop(equiv)
                    break
            # If nothing was found, set to npool but only if it is larger
            # than 1
            else:
                if self._npool > 1:
                    kwargs["npool"] = self._npool
        super()._translate_kwargs(kwargs)

    def _verify_kwargs_against_default_kwargs(self):
        super()._verify_kwargs_against_default_kwargs()
        n_active = self.kwargs.get("n_active")
        n_effective = self.kwargs.get("n_effective")
        if n_active >= n_effective:
            logger.warning(
                "Running with n_active > n_effective is not recommended"
            )

    def _get_pocomc_boundaries(self, key):
        # Based on the equivalent method for dynesty
        selected = list()
        for ii, param in enumerate(self.search_parameter_keys):
            if self.priors[param].boundary == key:
                logger.debug(f"Setting {key} boundary for {param}")
                selected.append(ii)
        if len(selected) == 0:
            selected = None
        return selected

    @staticmethod
    def _get_log_likelihood_fn(evaluate_constraints):
        if evaluate_constraints:
            return _log_likelihood_wrapper_with_constraints
        else:
            return _log_likelihood_wrapper

    def run_sampler(self):

        init_kwargs = {k: self.kwargs.get(k) for k in self.init_kwargs.keys()}
        run_kwargs = {k: self.kwargs.get(k) for k in self.run_kwargs.keys()}

        evaluate_constraints_in_prior = init_kwargs.pop(
            "evaluate_constraints_in_prior",
        )

        prior = PriorWrapper(
            self.priors,
            self.search_parameter_keys,
            evaluate_constraints=evaluate_constraints_in_prior,
        )

        output_dir = \
            Path(self.outdir) / f"{self.sampler_name}_{self.label}" / ""
        output_dir.mkdir(parents=True, exist_ok=True)

        self._setup_pool()
        pool = self.kwargs.pop("pool", None)
        resume = self.kwargs.pop("resume", False)

        # Set the boundary conditions
        for key in ["reflective", "periodic"]:
            init_kwargs[key] = self._get_pocomc_boundaries(key)

        sampler = pocomc.Sampler(
            prior=prior,
            likelihood=self._get_log_likelihood_fn(not evaluate_constraints_in_prior),
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

        # Want i.i.d samples without duplicates
        posterior_samples = bilby.core.result.rejection_sample(
            samples, weights
        )

        self._close_pool()

        self.result.samples = posterior_samples
        self.result.log_evidence = logz
        self.result.log_evidence_error = logz_err
        self.result.num_likelihood_evaluations = sampler.results["calls"][-1]
        return self.result
