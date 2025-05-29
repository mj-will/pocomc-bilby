#!/usr/bin/env python
"""
Example of performing GW inference with pocomc.

This example is on the 'fast_tutorial.py' from bilby.
"""

import bilby

# Set up the parameters for the simulated data
duration = 4.0
sampling_frequency = 2048.0
minimum_frequency = 20

outdir = "outdir"
label = "pocomc_example"
bilby.core.utils.setup_logger(outdir=outdir, label=label)

bilby.core.utils.random.seed(88170235)

# Define the injection parameters, this is a GW150914-like injection
injection_parameters = dict(
    mass_1=36.0,
    mass_2=29.0,
    a_1=0.4,
    a_2=0.3,
    tilt_1=0.5,
    tilt_2=1.0,
    phi_12=1.7,
    phi_jl=0.3,
    luminosity_distance=2000.0,
    theta_jn=0.4,
    psi=2.659,
    phase=1.3,
    geocent_time=1126259642.413,
    ra=1.375,
    dec=-1.2108,
)

# Define the waveform generator
waveform_arguments = dict(
    waveform_approximant="IMRPhenomPv2",
    reference_frequency=50.0,
    minimum_frequency=minimum_frequency,
)

waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments,
)

ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=injection_parameters["geocent_time"] - 2,
)
ifos.inject_signal(
    waveform_generator=waveform_generator, parameters=injection_parameters
)

# Define the priors
priors = bilby.gw.prior.BBHPriorDict()
# Fix various parameters to reduce the dimensionality of the problem
for key in [
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "psi",
    "ra",
    "dec",
    "geocent_time",
    "phase",
]:
    priors[key] = injection_parameters[key]

priors.validate_prior(duration, minimum_frequency)

# Define the likelihood
likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator
)

# Run pocomc
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="pocomc",
    n_active=1000,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
    n_pool=4,  # pocomc supports multiprocessing
)

result.plot_corner()
