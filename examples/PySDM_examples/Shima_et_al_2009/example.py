import os
from typing import Optional

import numpy as np
from PySDM_examples.Shima_et_al_2009.settings import Settings
from PySDM_examples.Shima_et_al_2009.spectrum_plotter import SpectrumPlotter

from PySDM.backends import CPU
from PySDM.builder import Builder
from PySDM.dynamics import Coalescence
from PySDM.environments import Box
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity
from PySDM.products import ParticleVolumeVersusRadiusLogarithmSpectrum, WallTime


def run(settings, backend=CPU, observers=()):
    env = Box(dv=settings.dv, dt=settings.dt)
    builder = Builder(
        n_sd=settings.n_sd, backend=backend(formulae=settings.formulae), environment=env
    )
    attributes = {}
    sampling = ConstantMultiplicity(settings.spectrum)
    attributes["volume"], attributes["multiplicity"] = sampling.sample(settings.n_sd)
    coalescence = Coalescence(
        collision_kernel=settings.kernel, adaptive=settings.adaptive
    )
    builder.add_dynamic(coalescence)
    products = (
        ParticleVolumeVersusRadiusLogarithmSpectrum(
            settings.radius_bins_edges, name="dv/dlnr"
        ),
        WallTime(),
    )
    particulator = builder.build(attributes, products)

    for observer in observers:
        particulator.observers.append(observer)

    vals = {}
    particulator.products["wall time"].reset()
    for step in settings.output_steps:
        particulator.run(step - particulator.n_steps)
        vals[step] = particulator.products["dv/dlnr"].get()[0]
        vals[step][:] *= settings.rho

    exec_time = particulator.products["wall time"].get()
    return vals, exec_time


def main(plot: bool, save: Optional[str]):
    with np.errstate(all="raise"):
        settings = Settings()

        settings.n_sd = 2**15

        states, _ = run(settings)

    with np.errstate(invalid="ignore"):
        plotter = SpectrumPlotter(settings)
        plotter.smooth = True
        for step, vals in states.items():
            _ = plotter.plot(vals, step * settings.dt)
            # assert _ < 200  # TODO #327
        if save is not None:
            n_sd = settings.n_sd
            plotter.save(save + "/" + f"{n_sd}_shima_fig_2" + "." + plotter.format)
        if plot:
            plotter.show()


if __name__ == "__main__":
    main(plot="CI" not in os.environ, save=None)
