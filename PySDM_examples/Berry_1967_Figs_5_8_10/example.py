"""
Created at 08.08.2019
"""

import numpy as np

from PySDM.dynamics.coalescence.kernels import Geometric, Hydrodynamic, Electric
from PySDM_examples.Berry_1967_Figs_5_8_10.settings import Settings
from PySDM_examples.Berry_1967_Figs_5_8_10.spectrum_plotter import SpectrumPlotter
from PySDM.attributes.droplet.terminal_velocity import gunn_and_kinzer
from PySDM_examples.Shima_et_al_2009_Fig_2.example import run


def main(plot: bool, save):
    with np.errstate(all='ignore'):

        u_term_approxs = (gunn_and_kinzer.Interpolation,)
        dts = (1, 10, 'adaptive')
        setup_prop = {Geometric: (0, 100, 200, 300, 400, 500, 600, 700, 750, 800, 850),
                      Electric: (0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000),
                      Hydrodynamic: (0, 1600, 1800, 2000, 2200)
                      }
        setups = {}

        for u_term_approx in u_term_approxs:
            setups[u_term_approx] = {}
            for dt in dts:
                setups[u_term_approx][dt] = {}

                for kernel in setup_prop:
                    s = Settings()
                    s.u_term = u_term_approx
                    s.dt = 10 if dt == 'adaptive' else dt
                    s.adaptive = dt == 'adaptive'
                    s.kernel = kernel()
                    s._steps = setup_prop[kernel]
                    setups[u_term_approx][dt][kernel] = s

        states = {}
        for u_term_approx in setups:
            states[u_term_approx] = {}
            for dt in setups[u_term_approx]:
                states[u_term_approx][dt] = {}
                for kernel in setups[u_term_approx][dt]:
                    states[u_term_approx][dt][kernel], _ = run(setups[u_term_approx][dt][kernel])

    if plot or save is not None:
        for u_term_approx in setups:
            for dt in setups[u_term_approx]:
                for kernel in setups[u_term_approx][dt]:
                    plotter = SpectrumPlotter(setups[u_term_approx][dt][kernel], legend=True)
                    for step, vals in states[u_term_approx][dt][kernel].items():
                        plotter.plot(vals, step * setups[u_term_approx][dt][kernel].dt)
                    if save is not None:
                        n_sd = setups[u_term_approx][dt][kernel].n_sd
                        plotter.save(save + "/" +
                                     f"{n_sd}_{u_term_approx.__name__}_{dt}_{kernel.__name__}" +
                                     "." + plotter.format)
                    if plot:
                        plotter.show()


if __name__ == '__main__':
    main(plot=False, save=".")
