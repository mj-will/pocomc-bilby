import datetime
import inspect
import time
from pathlib import Path

import bilby
import numpy as np
import pandas as pd
import pocomc
from bilby.core.sampler.base_sampler import signal_wrapper
from bilby.core.utils.log import logger

from .prior import PriorWrapper


def _log_likelihood_wrapper(theta):
    """Wrapper to the log likelihood.

    Does not evaluate the prior constraints.

    Needed for multiprocessing.
    """
    from bilby.core.sampler.base_sampler import _sampling_convenience_dump

    theta = {
        key: theta[ii]
        for ii, key in enumerate(
            _sampling_convenience_dump.search_parameter_keys
        )
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
        for ii, key in enumerate(
            _sampling_convenience_dump.search_parameter_keys
        )
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

    This implementation includes some additional keyword arguments:

    - :code:`evaluate_constraints_in_prior`, that determines if the prior
    prior constraints are evaluated when computing the log-likelihood
    (:code:`False`) or when evaluating the log-prior(:code:`True`).

    - :code:`track_sampling_time`, that determines if the total sampling time
    is tracked and saved in a file. The file is saved in the same output
    directory as the sampler outputs. If false, the sampling time reported in
    the result file will not account for checkpointing.

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
            "periodic",  # Set automatically
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
    def time_file_path(self):
        """Path to the file that stores the total sampling time."""
        return (
            Path(self.outdir)
            / f"{self.sampler_name}_{self.label}"
            / "sampling_time.dat"
        )

    @property
    def default_kwargs(self):
        kwargs = self.init_kwargs
        kwargs.update(self.run_kwargs)
        kwargs["resume"] = True
        kwargs["npool"] = None
        kwargs["track_sampling_time"] = False
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

    @signal_wrapper
    def run_sampler(self):
        self.track_sampling_time = self.kwargs.pop(
            "track_sampling_time", False
        )
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

        output_dir = (
            Path(self.outdir) / f"{self.sampler_name}_{self.label}" / ""
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        self._setup_pool()
        pool = self.kwargs.pop("pool", None)
        resume = self.kwargs.pop("resume", False)

        # Set the boundary conditions
        for key in ["reflective", "periodic"]:
            init_kwargs[key] = self._get_pocomc_boundaries(key)

        if resume and run_kwargs["resume_state_path"] is None:
            resume_state_path = self._find_resume_state_path(output_dir)
            if resume_state_path is not None:
                logger.info(f"Resuming pocomc from: {resume_state_path}")
                run_kwargs["resume_state_path"] = resume_state_path
            else:
                logger.debug("No files to resume from")
        self._check_and_load_sampling_time(resume=resume)
        self.start_time = time.time()

        sampler = pocomc.Sampler(
            prior=prior,
            likelihood=self._get_log_likelihood_fn(
                not evaluate_constraints_in_prior
            ),
            vectorize=False,
            output_label=self.label,
            output_dir=output_dir,
            n_dim=self.ndim,
            pool=pool,
            **init_kwargs,
        )

        sampler.run(**run_kwargs)

        samples, weights, logl, logp = sampler.posterior()
        logz, logz_err = sampler.evidence()

        # Include the log likelihood and log prior in the samples
        # so that we can populate the result object correctly
        samples = pd.DataFrame(samples, columns=self.search_parameter_keys)
        samples["log_likelihood"] = logl
        samples["log_prior"] = logp
        # Want i.i.d samples without duplicates
        posterior_samples = bilby.core.result.rejection_sample(
            samples, weights
        )
        if self.track_sampling_time:
            self._calculate_and_save_sampling_time()
        self._close_pool()

        self.result.samples = posterior_samples.drop(
            columns=["log_likelihood", "log_prior"]
        ).values
        self.result.log_likelihood_evaluations = posterior_samples[
            "log_likelihood"
        ].values
        self.result.log_prior_evaluations = posterior_samples[
            "log_prior"
        ].values
        self.result.log_evidence = logz
        self.result.log_evidence_err = logz_err
        self.result.num_likelihood_evaluations = sampler.results["calls"][-1]
        if self.track_sampling_time:
            self.result.sampling_time = datetime.timedelta(
                seconds=self.total_sampling_time
            )
        return self.result

    def _find_resume_state_path(self, output_dir):
        """Find the state file to resume from.

        If the final state file is found, it is used. Otherwise, the state file
        with the largest t value is used.
        """
        files = list(output_dir.glob("*.state"))
        for file in files:
            if "final" in file.stem:
                logger.info("Found final state file")
                return file
        t_values = [int(file.stem.split("_")[-1]) for file in files]
        if len(t_values):
            t_max = max(t_values)
            state_path = output_dir / f"{self.label}_{t_max}.state"
            return state_path
        else:
            return None

    def _check_and_load_sampling_time(self, resume: bool = False):
        """Check if the sampling time file exists and load the total
        sampling time.

        If resume is False, the total sampling time is set to 0.0 and any
        existing sampling time file is overwritten.
        """
        if not resume:
            self.total_sampling_time = 0.0
            if self.time_file_path.exists():
                logger.debug("Overwriting existing sampling time file")
                with open(self.time_file_path, "w") as f:
                    f.write(f"{self.total_sampling_time}\n")
        else:
            if self.time_file_path.exists():
                with open(self.time_file_path, "r") as time_file:
                    self.total_sampling_time = float(time_file.readline())
            else:
                self.total_sampling_time = 0.0

    def _calculate_and_save_sampling_time(self):
        current_time = time.time()
        new_sampling_time = current_time - self.start_time
        self.total_sampling_time += new_sampling_time
        with open(self.time_file_path, "w") as f:
            f.write(f"{self.total_sampling_time}\n")
        self.start_time = time.time()

    def write_current_state(self):
        # Can currently manually checkpoint pocomc
        pass

    def write_current_state_and_exit(self, signum=None, frame=None):
        # We implement this here since we want to log the information
        # irrespective of the state of the pool.
        if self.track_sampling_time:
            self._calculate_and_save_sampling_time()
        super().write_current_state_and_exit(signum=signum, frame=frame)
