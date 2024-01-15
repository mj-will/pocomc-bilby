# sampler-template

Template for sampler plugins in bilby

There are three main components to a sampler plugin for bilby:

- The sampler class
- The `pyproject.toml`
- The test suite


## The sampler class

This is the interface between the external sampling code and bilby, in this
template repo is it located in `src/demo_sampler_bilby/plugin.py`.

The class should inherit from one of:

- `bilby.core.sampler.Sampler`
- `bilby.core.sampler.MCMCSampler`
- `bilby.core.sampler.NestedSampler`

The sampler is run in the `run_sampler` method


## pyproject.toml

Various fields in the `pyproject.toml` need to be set, these are:

* `name`: this is the name Python package, we recommend using `<name of your sampler>_bilby`
* `author`: this should include any authors
* `dependencies`: any dependencies should be included here, this must include `bilby`
* `[project.entry-points."bilby.samplers"]` see below


### Adding the entry points

This section of the `pyproject.toml` makes the sampler 'visible' within bilby.

```
[project.entry-points."bilby.samplers"]
demo_sampler = "demo_sampler_bilby.plugin:DemoSampler"
```

The name of the sampler within bilby is determined based on the name used here
(e.g. `demo_sampler` in this case).

The string should points to the file and, after the colon, the sampler class.


## Tests

The plugin should include a test suite.

`tests/test_bilby_integration.py` includes a standard test that
all samplers should pass but other tests can be included in here. The file should
be updated name matches the name of the sampler provided by the plugin and any keyword
arguments should be set.


### Continuous Integration

The tests are run automatically via GitHub Actions. These are configured in
`.github/workflow/test.yaml`. You may wish to change the version of Python
being tested or included additional configuration, e.g. for more complex installation
processes.


## Plugin in an existing package

It is also possible to include the plugin in an existing package rather
than creating a separate plugin package. In this case, you need to define
the sampler class somewhere within the existing package and then add an entry
point.
