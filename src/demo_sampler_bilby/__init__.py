"""An example of how to implement a sampler plugin in for bilby.

This package provides the 'demo_sampler' sampler.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"
