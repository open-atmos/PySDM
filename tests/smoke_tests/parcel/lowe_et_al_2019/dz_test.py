import numpy as np
from matplotlib import pyplot
from PySDM_examples.Lowe_et_al_2019 import Settings, Simulation
from PySDM_examples.Lowe_et_al_2019.aerosol_code import AerosolMarine

from PySDM.initialisation.sampling import spectral_sampling as spec_sampling
from PySDM.initialisation.spectra import Sum
from PySDM.physics import si


def test_dz(plot=False):
    consts = {
        "delta_min": 0.1,
        "MAC": 1,
        "HAC": 1,
        "c_pd": 1006 * si.joule / si.kilogram / si.kelvin,
        "g_std": 9.81 * si.m / si.s**2,
        "scipy_ode_solver": False,
    }

    output = {}
    aerosol = AerosolMarine()
    model = "Constant"
    for i, dz_test in enumerate((0.1, 1, 10)):
        key = f"{dz_test}"
        settings = Settings(
            dz=dz_test * si.m,
            n_sd_per_mode=200,
            model=model,
            aerosol=aerosol,
            spectral_sampling=spec_sampling.ConstantMultiplicity,
            **consts,
        )
        simulation = Simulation(settings)
        output[key] = simulation.run()
        output[key]["color"] = "C" + str(i)

    for i, out_item in enumerate(output.values()):
        if i == 0:
            x = out_item["S_max"][-1]
        np.testing.assert_approx_equal(x, out_item["S_max"][-1], significant=1)

    if plot:
        pyplot.rc("font", size=14)
        fig, axs = pyplot.subplots(1, 2, figsize=(11, 4), sharey=True)
        vlist = ("S_max", "n_c_cm3")

        for idx, var in enumerate(vlist):
            for key, out_item in output.items():
                Y = np.asarray(out_item["z"])
                if var == "RH":
                    X = np.asarray(out_item[var]) - 100
                else:
                    X = out_item[var]
                axs[idx].plot(
                    X, Y, label=f"dz={key} m", color=out_item["color"], linestyle="-"
                )
            # axs[idx].set_ylim(0, 210)

            axs[idx].set_ylabel("Displacement [m]")
            if var == "S_max":
                axs[idx].set_xlabel("Supersaturation [%]")
                axs[idx].set_xlim(0)
            elif var == "n_c_cm3":
                axs[idx].set_xlabel("Cloud droplet concentration [cm$^{-3}$]")
            else:
                assert False

        for ax in axs:
            ax.grid()
        axs[0].legend(fontsize=12)
        pyplot.show()
