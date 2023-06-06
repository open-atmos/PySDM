import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from PySDM_examples.Arabas_and_Shima_2017.settings import setups
from PySDM_examples.Arabas_and_Shima_2017.simulation import Simulation

from PySDM.backends import CPU, GPU
from PySDM.backends.impl_numba.test_helpers import scipy_ode_condensation_solver


def data(n_output, rtols, schemes, setups_num):
    resultant_data = {}
    for scheme in schemes:
        resultant_data[scheme] = {}
        if scheme == "SciPy":
            for rtol in rtols:
                resultant_data[scheme][rtol] = []
            for settings_idx in range(setups_num):
                settings = setups[settings_idx]
                settings.n_output = n_output
                simulation = Simulation(settings)
                scipy_ode_condensation_solver.patch_particulator(
                    simulation.particulator
                )
                results = simulation.run()
                for rtol in rtols:
                    resultant_data[scheme][rtol].append(results)
        else:
            for rtol in rtols:
                resultant_data[scheme][rtol] = []
                for settings_idx in range(setups_num):
                    settings = setups[settings_idx]
                    settings.rtol_x = rtol
                    settings.rtol_thd = rtol
                    settings.n_output = n_output
                    simulation = Simulation(
                        settings, backend=CPU if scheme == "CPU" else GPU
                    )
                    results = simulation.run()
                    resultant_data[scheme][rtol].append(results)
    return resultant_data


def add_color_line(fig, ax, x, y, z):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    z = np.array(z)
    vmin = min(np.nanmin(z), np.nanmax(z) / 2)
    lc = LineCollection(
        segments,
        cmap=plt.get_cmap("plasma"),
        norm=matplotlib.colors.LogNorm(vmax=1, vmin=vmin),
    )
    lc.set_array(z)
    lc.set_linewidth(3)

    ax.add_collection(lc)
    fig.colorbar(lc, ax=ax)


def plot(plot_data, rtols, schemes, setups_num, show_plot, path=None):
    _rtol = "$r_{tol}$"

    plt.ioff()
    fig, axs = plt.subplots(
        setups_num, len(rtols), sharex=True, sharey=True, figsize=(10, 13)
    )
    for settings_idx in range(setups_num):
        SCIPY_S = None
        PySDM_S = None
        for rtol_idx, _rtol in enumerate(rtols):
            ax = axs[settings_idx, rtol_idx]
            for scheme in schemes:
                datum = plot_data[scheme][_rtol][settings_idx]
                S = datum["S"]
                z = datum["z"]
                dt = datum["dt_cond_min"]
                if scheme == "SciPy":
                    ax.plot(S, z, label=scheme, color="grey")
                    SCIPY_S = np.array(S)
                else:
                    add_color_line(fig, ax, S, z, dt)
                    PySDM_S = np.array(S)
            if SCIPY_S is not None and PySDM_S is not None:
                mae = np.mean(np.abs(SCIPY_S - PySDM_S))
                ax.set_title(f"MAE: {mae:.4E}")
            ax.set_xlim(-7.5e-3, 7.5e-3)
            ax.set_ylim(0, 180)
            ax.get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
            ax.grid()
            plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")
    for i, ax in enumerate(axs[:, 0]):
        ax.set(ylabel=r"$\bf{settings: " + str(i) + "}$\ndisplacement [m]")
    for i, ax in enumerate(axs[-1, :]):
        ax.set(xlabel="supersaturation\n" + r"$\bf{r_{tol}: " + str(rtols[i]) + "}$")

    plt.tight_layout()

    if path is not None:
        plt.savefig(path + ".pdf", format="pdf")
    if show_plot:
        plt.show()


def main(save=None, show_plot=True):
    rtols = [1e-7, 1e-11]
    schemes = ["CPU", "SciPy"]
    setups_num = len(setups)
    input_data = data(80, rtols, schemes, setups_num)
    plot(input_data, rtols, schemes, setups_num, show_plot, save)


if __name__ == "__main__":
    main("SCIPY_VS_ADAPTIVE", show_plot="CI" not in os.environ)
