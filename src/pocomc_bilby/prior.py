import numpy as np
from pocomc.prior import Prior


class PriorWrapper(Prior):
    """Wrapper for bilby prior class to make it compatible with pocomc"""
    def __init__(self, bilby_priors, sampling_parameters):
        self.bilby_priors = bilby_priors
        self.sampling_parameters = sampling_parameters

    def to_dict(self, x):
        return {k: x[..., i] for i, k in enumerate(self.sampling_parameters)}

    def from_dict(self, x, keys=None):
        if keys is None:
            keys = self.sampling_parameters
        return np.array([x[v] for v in keys]).T

    def logpdf(self, x):
        return self.bilby_priors.ln_prob(self.to_dict(x), axis=0)

    def rvs(self, size=1):
        return self.from_dict(
            self.bilby_priors.sample(size),
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
