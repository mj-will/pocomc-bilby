[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15543647.svg)](https://doi.org/10.5281/zenodo.15543647)
[![PyPI](https://img.shields.io/pypi/v/pocomc-bilby)](https://pypi.org/project/pocomc-bilby/)

# pocomc-bilby

`pocomc` plugin for `bilby`.

This package provides a plugin for using `pocomc` with `bilby` via the [sampler plugins interface](https://bilby-dev.github.io/bilby/plugins.html#sampler-plugins). `pocomc` is Sequential Monte Carlo sampler
that implements a specific flavour of SMC called [Persistent Sampling](https://arxiv.org/abs/2407.20722). For more details,
see the [`pocomc` documentation](https://pocomc.readthedocs.io/en/latest/index.html).

## Installation

**Note:** since `pocomc` depends on `torch` we recommend installing it first using
the instructions on the [PyTorch website](https://pytorch.org/).

`pocomc-bilby` is can be installed via `pip`:

```
pip install pococmc-bilby
```

**Note:** this plugin only supports `pocomc>=1.2.6`.

## Usage

Once installed, `pocomc` can be used as you would any other sampler in `bilby`:

```python
import bilby

# Define likelihood & priors as normal
priors = ...
likelihood = ...

bilby.run_sampler(
    sampler="pocomc",
    ...
)
```

`pocomc` has two different types of settings, those specified when initializing the
sampler and those specified when calling the `run` method. When using `pocomc` via
`bilby` both types of settings can be passed to `run_sampler` and these will automatically
be passed to the correct method.

For details about the various settings, see the [`pocomc` documentation](https://pocomc.readthedocs.io/en/latest/index.html).

This plugin is also compatible with `bilby_pipe`, including support for checkpointing.

## Attribution & Citation

If you use `pocomc-bilby` in your own work please cite the [DOI for this package](https://doi.org/10.5281/zenodo.15543647), corresponding paper (to be added) and the
citations for `pocomc`,
see [Attribution & Citation in the `pocomc` documentation](https://pocomc.readthedocs.io/en/latest/index.html#attribution-citation)
