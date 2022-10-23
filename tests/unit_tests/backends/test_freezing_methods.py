# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
from matplotlib import pyplot

from PySDM import Builder, Formulae
from PySDM.dynamics import Freezing
from PySDM.environments import Box
from PySDM.physics import constants_defaults as const
from PySDM.physics import si
from PySDM.products import IceWaterContent

from ...backends_fixture import backend_class

assert hasattr(backend_class, "_pytestfixturefunction")


class TestFreezingMethods:
    # TODO #599
    def test_record_freezing_temperature_on_time_dependent_freeze(self):
        pass

    # TODO #599
    def test_no_subsaturated_freezing(self):
        pass

    @staticmethod
    # pylint: disable=redefined-outer-name
    def test_freeze_singular(backend_class):
        # arrange
        n_sd = 44
        dt = 1 * si.s
        dv = 1 * si.m**3
        T_fz = 250 * si.K
        vol = 1 * si.um**3
        multiplicity = 1e10
        steps = 1

        formulae = Formulae()
        builder = Builder(n_sd=n_sd, backend=backend_class(formulae=formulae))
        env = Box(dt=dt, dv=dv)
        builder.set_environment(env)
        builder.add_dynamic(Freezing(singular=True))
        attributes = {
            "n": np.full(n_sd, multiplicity),
            "freezing temperature": np.full(n_sd, T_fz),
            "volume": np.full(n_sd, vol),
        }
        products = (IceWaterContent(name="qi"),)
        particulator = builder.build(attributes=attributes, products=products)
        env["T"] = T_fz
        env["RH"] = 1.000001

        # act
        particulator.run(steps=steps)

        # assert
        np.testing.assert_almost_equal(
            np.asarray(particulator.products["qi"].get()),
            [n_sd * multiplicity * vol * const.rho_w / dv],
        )

    @staticmethod
    # pylint: disable=too-many-locals,redefined-outer-name
    def test_freeze_time_dependent(backend_class, plot=False):
        # Arrange
        seed = 44
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
            0.25e9  # effectively interpreted here as seconds, i.e. cycle = 1 * si.s
        )

        # dummy (but must-be-set) values
        vol = (
            44  # for sign flip (ice water has negative volumes), value does not matter
        )
        d_v = 666  # products use conc., dividing there, multiplying here, value does not matter

        def hgh(t):
            return np.exp(-0.75 * rate * (t - total_time / 4))

        def low(t):
            return np.exp(-1.25 * rate * (t + total_time / 4))

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
                seed=seed,
            )
            builder = Builder(n_sd=n_sd, backend=backend_class(formulae=formulae))
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
            env["RH"] = 1.0001
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
            pyplot.step(
                out["dt"] * np.arange(len(out["unfrozen_fraction"])),
                out["unfrozen_fraction"],
                label=f"dt={out['dt']:.2g} / N={out['N']}",
                marker=".",
                linewidth=1 + out["N"] // 8,
            )

        _plot_fit(fit_x, fit_y, low, hgh, total_time)
        if plot:
            pyplot.show()

        # Assert
        for out in output.values():
            data = np.asarray(out["unfrozen_fraction"])
            arg = out["dt"] * np.arange(len(data))
            np.testing.assert_array_less(data, hgh(arg))
            np.testing.assert_array_less(low(arg), data)


def _plot_fit(fit_x, fit_y, low, hgh, total_time):
    pyplot.plot(
        fit_x, fit_y, color="black", linestyle="--", label="theory", linewidth=5
    )
    pyplot.plot(
        fit_x, hgh(fit_x), color="black", linestyle=":", label="assert upper bound"
    )
    pyplot.plot(
        fit_x, low(fit_x), color="black", linestyle=":", label="assert lower bound"
    )
    pyplot.legend()
    pyplot.yscale("log")
    pyplot.ylim(fit_y[-1], fit_y[0])
    pyplot.xlim(None, total_time)
    pyplot.xlabel("time")
    pyplot.ylabel("unfrozen fraction")
    pyplot.grid()
