# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
from matplotlib import pyplot
from PySDM_examples.Lowe_et_al_2019 import Settings, Simulation
from PySDM_examples.Lowe_et_al_2019.aerosol import AerosolBoreal, AerosolMarine
from PySDM_examples.Lowe_et_al_2019.constants_def import LOWE_CONSTS

from PySDM import Formulae
from PySDM.initialisation.sampling import spectral_sampling as spec_sampling
from PySDM.physics import si

FORMULAE = Formulae(constants=LOWE_CONSTS)
WATER_MOLAR_VOLUME = FORMULAE.constants.water_molar_volume


def test_zero_forg(plot=False):  # pylint: disable=too-many-locals
    nRes = 3
    updraft_list = np.geomspace(0.1, 10, nRes)

    subplot_list = ["a", "b", "c", "d"]
    models = ("Constant", "CompressedFilmOvadnevaite")

    Acc = {"a": 30, "b": 134, "c": 160, "d": 540}

    cdnc_compare = np.zeros((len(models), len(subplot_list), len(updraft_list)))
    for i, w in enumerate(updraft_list):
        for k, subplot in enumerate(subplot_list):
            for m, model in enumerate(models):
                settings = Settings(
                    dz=1 * si.m,
                    n_sd_per_mode=20,
                    model=model,
                    aerosol={
                        "a": AerosolMarine(
                            water_molar_volume=WATER_MOLAR_VOLUME,
                            Forg=0,
                            Acc_N2=Acc["a"],
                        ),
                        "b": AerosolMarine(
                            water_molar_volume=WATER_MOLAR_VOLUME,
                            Forg=0,
                            Acc_N2=Acc["b"],
                        ),
                        "c": AerosolBoreal(
                            water_molar_volume=WATER_MOLAR_VOLUME,
                            Forg=0,
                            Acc_N2=Acc["c"],
                        ),
                        "d": AerosolBoreal(
                            water_molar_volume=WATER_MOLAR_VOLUME,
                            Forg=0,
                            Acc_N2=Acc["d"],
                        ),
                    }[subplot],
                    w=w * si.m / si.s,
                    spectral_sampling=spec_sampling.ConstantMultiplicity,
                )
                simulation = Simulation(settings)
                output = simulation.run()
                cdnc_compare[m, k, i] = np.array(output["CDNC_cm3"])[-1]

    mrkr = ["o", "s", "*", "v", "^", "D", "h", "x", "+", "8", "p", "<", ">", "d", "H"]
    _, axes = pyplot.subplots(
        1,
        len(subplot_list),
        figsize=(len(subplot_list) * 4, 4),
        constrained_layout=True,
        sharex=True,
        sharey=False,
    )

    for k, subplot in enumerate(subplot_list):
        if len(subplot_list) > 1:
            ax = axes[k]
        else:
            ax = axes

        for m, model in enumerate(models):
            for i, w in enumerate(updraft_list):
                if i == 0:
                    ax.scatter(
                        w * (1 + 0.1 * m),
                        cdnc_compare[m, k, i],
                        color="C" + str(m),
                        marker=mrkr[m],
                        label=model,
                    )
                else:
                    ax.scatter(
                        w * (1 + 0.1 * m),
                        cdnc_compare[m, k, i],
                        color="C" + str(m),
                        marker=mrkr[m],
                    )
        ax.set_xscale("log")

        ax.set_title(subplot, loc="left", weight="bold")
        ax.set_xlabel("Updraft velocity, w [m s$^{-1}$]")
        if k == 0:
            ax.set_ylabel("CDNC [cm$^{-3}$]")
            ax.legend()

    pyplot.suptitle("zero organics")

    if plot:
        pyplot.show()

    np.testing.assert_allclose(
        cdnc_compare[0, :, :],
        cdnc_compare[1, :, :],
        rtol=1e-2,
    )
