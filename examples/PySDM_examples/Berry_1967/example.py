import os

import numpy as np
from PySDM_examples.Berry_1967.settings import Settings
from PySDM_examples.Berry_1967.spectrum_plotter import SpectrumPlotter
from PySDM_examples.Shima_et_al_2009.example import run

from PySDM.dynamics.collisions.collision_kernels import (
    Electric,
    Geometric,
    Hydrodynamic,
)


def main(plot: bool, save):
    with np.errstate(all="ignore"):
        u_term_approxs = ("GunnKinzer1949",)
        dts = (1, 10, "adaptive")
        setup_prop = {
            Geometric: (0, 100, 200, 300, 400, 500, 600, 700, 750, 800, 850),
            Electric: (0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000),
            Hydrodynamic: (0, 1600, 1800, 2000, 2200),
        }
        setups = {}

        for u_term_approx in u_term_approxs:
            setups[u_term_approx] = {}
            for dt in dts:
                setups[u_term_approx][dt] = {}

                for kernel_type, steps in setup_prop.items():
                    s = Settings(terminal_velocity_variant=u_term_approx)
                    s.dt = 10 if dt == "adaptive" else dt
                    s.adaptive = dt == "adaptive"
                    s.kernel = kernel_type()
                    s._steps = steps
                    setups[u_term_approx][dt][kernel_type] = s

        states = {}
        for u_term_approx, setup in setups.items():
            states[u_term_approx] = {}
            for dt in setup:
                states[u_term_approx][dt] = {}
                for kernel in setup[dt]:
                    states[u_term_approx][dt][kernel], _ = run(setup[dt][kernel])

    if plot or save is not None:
        for u_term_approx, setup in setups.items():
            for dt in setup:
                for kernel in setup[dt]:
                    plotter = SpectrumPlotter(setup[dt][kernel], legend=True)
                    for step, vals in states[u_term_approx][dt][kernel].items():
                        plotter.plot(vals, step * setup[dt][kernel].dt)
                    if save is not None:
                        n_sd = setup[dt][kernel].n_sd
                        plotter.save(
                            save
                            + "/"
                            + f"{n_sd}_{u_term_approx}_{dt}_{kernel.__name__}"
                            + "."
                            + plotter.format
                        )
                    if plot:
                        plotter.show()


if __name__ == "__main__":
    main(plot="CI" not in os.environ, save=".")
