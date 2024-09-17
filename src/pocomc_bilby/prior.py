import numpy as np
from pocomc.prior import Prior


class PriorWrapper(Prior):
    """Wrapper for bilby prior class to make it compatible with pocomc.

    Parameters
    ----------
    bilby_priors: bilby.core.prior.PriorDict
        Bilby prior dictionary.
    sampling_parameters: list
        List of parameters that are being sampled.
    evaluate_constraints: bool, optional
        If True, any prior constraints are evaluated when computing the log-
        prior PDF.
    """

    logpdf = None
    """Log-prior probability density function.

    Is set when the class is initialized to either based on the value of
    :code:`evaluate_constraints`.
    """

    rvs = None
    """Function for drawing random samples from the prior.

    Is set when the class is initialized to either based on the value of
    :code:`evaluate_constraints`.
    """

    def __init__(
        self,
        bilby_priors,
        sampling_parameters,
        evaluate_constraints=True,
    ):
        self.bilby_priors = bilby_priors
        self.sampling_parameters = sampling_parameters
        self.evaluate_constraints = evaluate_constraints

        if self.evaluate_constraints:
            self.logpdf = self._logpdf_with_constraints
            self.rvs = self._rvs_with_constraints
        else:
            self.logpdf = self._logpdf_without_constraints
            self.rvs = self._rvs_without_constraints

    def to_dict(self, x):
        return {k: x[..., i] for i, k in enumerate(self.sampling_parameters)}

    def from_dict(self, x, keys=None):
        if keys is None:
            keys = self.sampling_parameters
        return np.array([x[v] for v in keys]).T

    def _logpdf_with_constraints(self, x):
        x_dict = self.to_dict(x)
        # The priors already include the constraints
        return self.bilby_priors.ln_prob(x_dict, axis=0)

    def _logpdf_without_constraints(self, x):
        x_dict = self.to_dict(x)
        return np.sum(
            [self.bilby_priors[key].ln_prob(x_dict[key]) for key in x_dict],
            axis=0,
        )

    def _rvs_with_constraints(self, size=1):
        return self.from_dict(
            self.bilby_priors.sample_subset_constrained(
                keys=list(self.bilby_priors.keys()), size=size
            ),
            self.sampling_parameters,
        )

    def _rvs_without_constraints(self, size=1):
        return self.from_dict(
            self.bilby_priors.sample_subset(
                keys=list(self.bilby_priors.keys()), size=size
            ),
            self.sampling_parameters,
        )

    @property
    def bounds(self):
        bounds = []
        for key in self.bilby_priors.non_fixed_keys:
            bounds.append(
                [
                    self.bilby_priors[key].minimum,
                    self.bilby_priors[key].maximum,
                ]
            )
        return np.array(bounds, dtype=float)

    @property
    def dim(self):
        return len(self.sampling_parameters)
