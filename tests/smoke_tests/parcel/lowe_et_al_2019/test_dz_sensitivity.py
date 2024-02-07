""" checks how parcel equilibrium supersaturation depends on dz """

import numpy as np
from matplotlib import pyplot
from PySDM_examples.Lowe_et_al_2019 import Settings, Simulation
from PySDM_examples.Lowe_et_al_2019.aerosol_code import AerosolMarine
from PySDM_examples.Lowe_et_al_2019.constants_def import LOWE_CONSTS

from PySDM import Formulae
from PySDM.initialisation.sampling import spectral_sampling as spec_sampling
from PySDM.physics import si

FORMULAE = Formulae(constants=LOWE_CONSTS)
WATER_MOLAR_VOLUME = FORMULAE.constants.water_molar_volume


def test_dz_sensitivity(
    plot=False,
):  # pylint: disable=too-many-locals,too-many-branches
    # arrange
    output = {}
    aerosol = AerosolMarine(water_molar_volume=WATER_MOLAR_VOLUME)
    model = "Constant"

    # act
    for i, dz_test in enumerate((0.1, 1, 10)):
        key = f"{dz_test}"
        settings = Settings(
            dz=dz_test * si.m,
            n_sd_per_mode=200,
            model=model,
            aerosol=aerosol,
            spectral_sampling=spec_sampling.ConstantMultiplicity,
        )
        simulation = Simulation(settings)
        output[key] = simulation.run()
        output[key]["color"] = "C" + str(i)

    # plot
    pyplot.rc("font", size=14)
    _, axs = pyplot.subplots(1, 2, figsize=(11, 4), sharey=True)
    vlist = ("S_max", "CDNC_cm3")

    for idx, var in enumerate(vlist):
        for key, out_item in output.items():
            Y = np.asarray(out_item["z"])
            X = out_item[var]
            axs[idx].plot(
                X, Y, label=f"dz={key} m", color=out_item["color"], linestyle="-"
            )

        axs[idx].set_ylabel("Displacement [m]")
        if var == "S_max":
            axs[idx].set_xlabel("Supersaturation [%]")
            axs[idx].set_xlim(0)
        elif var == "CDNC_cm3":
            axs[idx].set_xlabel("Cloud droplet concentration [cm$^{-3}$]")
        else:
            assert False

    for ax in axs:
        ax.grid()
    axs[0].legend(fontsize=12)

    if plot:
        pyplot.show()
    else:
        pyplot.clf()

    # assert
    for i, out_item in enumerate(output.values()):
        if i == 0:
            x = out_item["S_max"][-1]
        np.testing.assert_approx_equal(x, out_item["S_max"][-1], significant=1)
