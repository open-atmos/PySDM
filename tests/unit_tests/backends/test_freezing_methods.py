# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
from matplotlib import pylab

from PySDM import Builder, Formulae
from PySDM.backends import CPU
from PySDM.dynamics import Freezing
from PySDM.environments import Box
from PySDM.physics import constants_defaults as const
from PySDM.products import IceWaterContent

from ...backends_fixture import backend_class  # TODO #599

assert hasattr(backend_class, "_pytestfixturefunction")


class TestFreezingMethods:
    # TODO #599
    @staticmethod
    # pylint: disable=redefined-outer-name
    def test_freeze_singular(backend_class):
        pass

    @staticmethod
    # pylint: disable=too-many-locals
    def test_freeze_time_dependent(plot=False):
        # Arrange
        cases = (
            {"dt": 5e5, "N": 1},
            {"dt": 1e6, "N": 1},
            {"dt": 5e5, "N": 8},
            {"dt": 1e6, "N": 8},
            {"dt": 5e5, "N": 16},
            {"dt": 1e6, "N": 16},
        )
        rate = 1e-9
        immersed_surface_area = 1

        number_of_real_droplets = 1024
        total_time = (
            2e9  # effectively interpretted here as seconds, i.e. cycle = 1 * si.s
        )

        # dummy (but must-be-set) values
        vol = (
            44  # for sign flip (ice water has negative volumes), value does not matter
        )
        d_v = 666  # products use conc., dividing there, multiplying here, value does not matter

        def hgh(t):
            return np.exp(-0.8 * rate * (t - total_time / 10))

        def low(t):
            return np.exp(-1.2 * rate * (t + total_time / 10))

        # Act
        output = {}

        for case in cases:
            n_sd = int(number_of_real_droplets // case["N"])
            assert n_sd == number_of_real_droplets / case["N"]
            assert total_time // case["dt"] == total_time / case["dt"]

            key = f"{case['dt']}:{case['N']}"
            output[key] = {"unfrozen_fraction": [], "dt": case["dt"], "N": case["N"]}

            formulae = Formulae(
                heterogeneous_ice_nucleation_rate="Constant",
                constants={"J_HET": rate / immersed_surface_area},
            )
            builder = Builder(n_sd=n_sd, backend=CPU(formulae=formulae))
            env = Box(dt=case["dt"], dv=d_v)
            builder.set_environment(env)
            builder.add_dynamic(Freezing(singular=False))
            attributes = {
                "n": np.full(n_sd, int(case["N"])),
                "immersed surface area": np.full(n_sd, immersed_surface_area),
                "volume": np.full(n_sd, vol),
            }
            products = (IceWaterContent(name="qi"),)
            particulator = builder.build(attributes=attributes, products=products)

            env["a_w_ice"] = np.nan

            cell_id = 0
            for i in range(int(total_time / case["dt"]) + 1):
                particulator.run(0 if i == 0 else 1)

                ice_mass_per_volume = particulator.products["qi"].get()[cell_id]
                ice_mass = ice_mass_per_volume * d_v
                ice_number = ice_mass / (const.rho_w * vol)
                unfrozen_fraction = 1 - ice_number / number_of_real_droplets
                output[key]["unfrozen_fraction"].append(unfrozen_fraction)

        # Plot
        fit_x = np.linspace(0, total_time, num=100)
        fit_y = np.exp(-rate * fit_x)

        for out in output.values():
            pylab.step(
                out["dt"] * np.arange(len(out["unfrozen_fraction"])),
                out["unfrozen_fraction"],
                label=f"dt={out['dt']:.2g} / N={out['N']}",
                marker=".",
                linewidth=1 + out["N"] // 8,
            )

        _plot_fit(fit_x, fit_y, low, hgh, total_time)
        if plot:
            pylab.show()

        # Assert
        for out in output.values():
            data = np.asarray(out["unfrozen_fraction"])
            arg = out["dt"] * np.arange(len(data))
            np.testing.assert_array_less(data, hgh(arg))
            np.testing.assert_array_less(low(arg), data)


def _plot_fit(fit_x, fit_y, low, hgh, total_time):
    pylab.plot(fit_x, fit_y, color="black", linestyle="--", label="theory", linewidth=5)
    pylab.plot(
        fit_x, hgh(fit_x), color="black", linestyle=":", label="assert upper bound"
    )
    pylab.plot(
        fit_x, low(fit_x), color="black", linestyle=":", label="assert lower bound"
    )
    pylab.legend()
    pylab.yscale("log")
    pylab.ylim(fit_y[-1], fit_y[0])
    pylab.xlim(None, total_time)
    pylab.xlabel("time")
    pylab.ylabel("unfrozen fraction")
    pylab.grid()
