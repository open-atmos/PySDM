"""
Created at 08.08.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np

from PySDM.particles_builder import ParticlesBuilder
from PySDM.environments import Box
from PySDM.dynamics import Coalescence
from PySDM.initialisation.spectral_sampling import constant_multiplicity
from PySDM.dynamics.coalescence.kernels import Golovin, Gravitational

from PySDM_examples.Berry_1967_Fig_5.setup import Setup
from PySDM_examples.Berry_1967_Fig_5.spectrum_plotter import SpectrumPlotter
from PySDM.state.products.particles_volume_spectrum import ParticlesVolumeSpectrum


def run(setup):
    particles_builder = ParticlesBuilder(n_sd=setup.n_sd, backend=setup.backend)
    particles_builder.set_environment(Box, {"dv": setup.dv, "dt": setup.dt})
    particles_builder.register_dynamic(Coalescence, {"kernel": setup.kernel})
    attributes = {}
    attributes['volume'], attributes['n'] = constant_multiplicity(setup.n_sd, setup.spectrum,
                                                                  (setup.init_x_min, setup.init_x_max))
    products = {ParticlesVolumeSpectrum: {}}
    particles = particles_builder.get_particles(attributes, products)
    particles.state.whole_attributes['terminal velocity'].atype = setup.atype

    vals = {}
    for step in setup.steps:
        particles.run(step - particles.n_steps)
        vals[step] = particles.products['dv/dlnr'].get(setup.radius_bins_edges)
        vals[step][:] *= setup.rho

    return vals, particles.stats


def main(plot: bool):
    with np.errstate(all='raise'):

        atypes = ('inter',)
        dts = (1,)
        setups = {}

        for atype in atypes:
            setups[atype] = {}
            for dt in dts:
                setups[atype][dt] = {}

                sweepout = Setup()
                sweepout.atype = atype
                sweepout.dt = dt
                sweepout.kernel = Gravitational(collection_efficiency=1)
                sweepout.steps = [0, 100, 200, 300, 400, 500, 600, 700, 750, 800, 850]
                sweepout.steps = [int(step/sweepout.dt) for step in sweepout.steps]
                setups[atype][dt]['sweepout'] = sweepout

                electro = Setup()
                electro.atype = atype
                electro.dt = dt
                electro.kernel = Gravitational(collection_efficiency='3000V/cm')
                electro.steps = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
                electro.steps = [int(step / electro.dt) for step in electro.steps]
                setups[atype][dt]['3000V/cm'] = electro

                hydro = Setup()
                hydro.atype = atype
                hydro.dt = dt
                hydro.kernel = Gravitational(collection_efficiency='hydrodynamic')
                hydro.steps = [1600, 1800, 2000, 2200]
                hydro.steps = [int(step / hydro.dt) for step in hydro.steps]
                setups[atype][dt]['hydrodynamic'] = hydro

        states = {}
        for atype in setups:
            states[atype] = {}
            for dt in setups[atype]:
                states[atype][dt] = {}
                for kernel in setups[atype][dt]:
                    states[atype][dt][kernel] = run(setups[atype][dt][kernel])[0]

    with np.errstate(invalid='ignore'):
        for atype in setups:
            for dt in setups[atype]:
                for kernel in setups[atype][dt]:
                    plotter = SpectrumPlotter(setups[atype][dt][kernel])
                    for step, vals in states[atype][dt][kernel].items():
                        plotter.plot(vals, step * setups[atype][dt][kernel].dt)
                    plotter.show(title=f"{atype[:3]}{dt}{kernel[:3]}", legend=True)


if __name__ == '__main__':
    main(plot=True)
