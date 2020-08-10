"""
Created at 08.08.2019
"""

import numpy as np

from PySDM.builder import Builder
from PySDM.environments import Box
from PySDM.dynamics import Coalescence
from PySDM.initialisation.spectral_sampling import constant_multiplicity

from PySDM_examples.Shima_et_al_2009_Fig_2.setup import SetupA
from PySDM_examples.Shima_et_al_2009_Fig_2.spectrum_plotter import SpectrumPlotter
from PySDM.state.products.particles_volume_spectrum import ParticlesVolumeSpectrum


def run(setup, observers=()):
    builder = Builder(n_sd=setup.n_sd, backend=setup.backend)
    builder.set_environment(Box(dv=setup.dv, dt=setup.dt))
    attributes = {}
    attributes['volume'], attributes['n'] = constant_multiplicity(setup.n_sd, setup.spectrum,
                                                                  (setup.init_x_min, setup.init_x_max))
    coalescence = Coalescence(setup.kernel)
    coalescence.adaptive = setup.adaptive
    builder.add_dynamic(coalescence)
    products = [ParticlesVolumeSpectrum()]
    particles = builder.build(attributes, products)
    if hasattr(setup, 'u_term') and 'terminal velocity' in particles.state.attributes:
        particles.state.attributes['terminal velocity'].approximation = setup.u_term(particles)

    for observer in observers:
        particles.observers.append(observer)

    vals = {}
    for step in setup.steps:
        particles.run(step - particles.n_steps)
        vals[step] = particles.products['dv/dlnr'].get(setup.radius_bins_edges)
        vals[step][:] *= setup.rho

    return vals, particles.stats


def main(plot: bool, save: str):
    with np.errstate(all='raise'):
        setup = SetupA()

        setup.n_sd = 2 ** 15

        states, _ = run(setup)

    with np.errstate(invalid='ignore'):
        plotter = SpectrumPlotter(setup)
        plotter.smooth = True
        for step, vals in states.items():
            plotter.plot(vals, step * setup.dt)
        if save is not None:
            n_sd = setup.n_sd
            plotter.save(save + "/" +
                         f"{n_sd}_shima_fig_2" +
                         "." + plotter.format)
        if plot:
            plotter.show()


if __name__ == '__main__':
    main(plot=True, save=None)
