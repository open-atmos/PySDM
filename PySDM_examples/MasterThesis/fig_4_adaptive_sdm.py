"""
Created at 20.08.2020
"""

import numpy as np
from PySDM_examples.Shima_et_al_2009_Fig_2.setup import SetupA
from PySDM_examples.Shima_et_al_2009_Fig_2.example import run
from PySDM_examples.Shima_et_al_2009_Fig_2.spectrum_plotter import SpectrumPlotter
from matplotlib import pyplot as plt


def main(plot: bool, save: str):
    n_sds = (13, 15, 17)
    n_sds = (2**i for i in n_sds)
    dts = (10, 5, 1, 'adaptive')

    fig, axs = plt.subplots(len(dts), len(n_sds),
                            sharex=True, sharey=True, figsize=(10, 13))

    for dt in dts:
        for n_sd in n_sds:
            setup = SetupA()

            setup.n_sd = n_sd
            setup.dt = dt if dt != 'adaptive' else 10
            setup.adaptive = dt == 'adaptive'

            states, _ = run(setup)

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