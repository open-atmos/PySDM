"""
Created at 08.08.2019
"""

import numpy as np

from PySDM.particles_builder import ParticlesBuilder
from PySDM.environments import Box
from PySDM.dynamics import Coalescence
from PySDM.initialisation.spectral_sampling import constant_multiplicity
from PySDM.dynamics.coalescence.kernels import Gravitational
from PySDM_examples.Berry_1967_Fig_5.setup import Setup
from PySDM_examples.Berry_1967_Fig_5.spectrum_plotter import SpectrumPlotter
from PySDM.state.products.particles_volume_spectrum import ParticlesVolumeSpectrum
from PySDM.attributes.droplet.terminal_velocity import gunn_and_kinzer


def run(setup):
    particles_builder = ParticlesBuilder(n_sd=setup.n_sd, backend=setup.backend)
    particles_builder.set_environment(Box, {"dv": setup.dv, "dt": setup.dt})
    particles_builder.register_dynamic(Coalescence, {"kernel": setup.kernel})
    attributes = {}
    attributes['volume'], attributes['n'] = constant_multiplicity(setup.n_sd, setup.spectrum,
                                                                  (setup.init_x_min, setup.init_x_max))
    products = {ParticlesVolumeSpectrum: {}}
    particles = particles_builder.get_particles(attributes, products)
    particles.state.whole_attributes['terminal velocity'].approximation = setup.u_term_approx(particles)
    particles.dynamics[str(Coalescence)].adaptive = setup.adaptive

    vals = {}
    for step in setup.steps:
        particles.run(step - particles.n_steps)
        vals[step] = particles.products['dv/dlnr'].get(setup.radius_bins_edges)
        vals[step][:] *= setup.rho

    return vals, particles.stats


def main(plot: bool):
    with np.errstate(all='ignore'):

        u_term_approxs = (gunn_and_kinzer.Interpolation,)
        dts = (1, 10, 'adaptive',)
        setups = {}

        for u_term_approx in u_term_approxs:
            setups[u_term_approx] = {}
            for dt in dts:
                setups[u_term_approx][dt] = {}

                sweepout = Setup()
                sweepout.u_term_approx = u_term_approx
                sweepout.dt = 10 if dt == 'adaptive' else dt
                sweepout.adaptive = dt == 'adaptive'
                sweepout.kernel = Gravitational(collection_efficiency=1)
                sweepout.steps = [0, 100, 200, 300, 400, 500, 600, 700, 750, 800, 850]
                sweepout.steps = [int(step/sweepout.dt) for step in sweepout.steps]
                setups[u_term_approx][dt]['sweepout'] = sweepout

                electro = Setup()
                electro.u_term_approx = u_term_approx
                electro.dt = 10 if dt == 'adaptive' else dt
                electro.adaptive = dt == 'adaptive'
                electro.kernel = Gravitational(collection_efficiency='3000V/cm')
                electro.steps = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
                electro.steps = [int(step / electro.dt) for step in electro.steps]
                setups[u_term_approx][dt]['3000V/cm'] = electro

                hydro = Setup()
                hydro.u_term_approx = u_term_approx
                hydro.dt = 10 if dt == 'adaptive' else dt
                hydro.adaptive = dt == 'adaptive'
                hydro.kernel = Gravitational(collection_efficiency='hydrodynamic')
                hydro.steps = [1600, 1800, 2000, 2200]
                hydro.steps = [int(step / hydro.dt) for step in hydro.steps]
                setups[u_term_approx][dt]['hydrodynamic'] = hydro

        states = {}
        for u_term_approx in setups:
            states[u_term_approx] = {}
            for dt in setups[u_term_approx]:
                states[u_term_approx][dt] = {}
                for kernel in setups[u_term_approx][dt]:
                    states[u_term_approx][dt][kernel] = run(setups[u_term_approx][dt][kernel])[0]

    if plot:
        for u_term_approx in setups:
            for dt in setups[u_term_approx]:
                for kernel in setups[u_term_approx][dt]:
                    plotter = SpectrumPlotter(setups[u_term_approx][dt][kernel])
                    for step, vals in states[u_term_approx][dt][kernel].items():
                        plotter.plot(vals, step * setups[u_term_approx][dt][kernel].dt)
                    plotter.show(title=f"{u_term_approx.__name__} {dt} {kernel}", legend=True)


if __name__ == '__main__':
    main(plot=True)
