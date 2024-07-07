# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from matplotlib import pyplot

from PySDM import Builder, Formulae
from PySDM.dynamics import Freezing
from PySDM.environments import Box
from PySDM.physics import si
from PySDM.products import IceWaterContent


class TestFreezingMethods:
    # TODO #599
    def test_record_freezing_temperature_on_time_dependent_freeze(self):
        pass

    # TODO #599
    def test_no_subsaturated_freezing(self):
        pass

    @staticmethod
    @pytest.mark.parametrize("singular", (True, False))
    @pytest.mark.parametrize("thaw", (True, False))
    @pytest.mark.parametrize("epsilon", (0, 1e-5))
    def test_thaw(backend_class, singular, thaw, epsilon):
        # arrange
        formulae = Formulae(particle_shape_and_density="MixedPhaseSpheres")
        env = Box(dt=1 * si.s, dv=1 * si.m**3)
        builder = Builder(
            n_sd=1, backend=backend_class(formulae=formulae), environment=env
        )
        builder.add_dynamic(Freezing(singular=singular, thaw=thaw))
        particulator = builder.build(
            products=(IceWaterContent(),),
            attributes={
                "multiplicity": np.ones(builder.particulator.n_sd),
                "volume": -1 * np.ones(builder.particulator.n_sd) * si.um**3,
                **(
                    {"freezing temperature": np.full(builder.particulator.n_sd, -1)}
                    if singular
                    else {
                        "immersed surface area": np.full(builder.particulator.n_sd, -1)
                    }
                ),
            },
        )
        env["T"] = formulae.constants.T0 + epsilon
        env["RH"] = np.nan
        if not singular:
            env["a_w_ice"] = np.nan
        assert particulator.products["ice water content"].get() > 0

        # act
        particulator.run(steps=1)

        # assert
        if thaw and epsilon > 0:
            assert particulator.products["ice water content"].get() == 0
        else:
            assert particulator.products["ice water content"].get() > 0

    @staticmethod
    def test_freeze_singular(backend_class):
        # arrange
        n_sd = 44
        dt = 1 * si.s
        dv = 1 * si.m**3
        T_fz = 250 * si.K
        water_mass = 1 * si.mg
        multiplicity = 1e10
        steps = 1

        formulae = Formulae(particle_shape_and_density="MixedPhaseSpheres")
        env = Box(dt=dt, dv=dv)
        builder = Builder(
            n_sd=n_sd, backend=backend_class(formulae=formulae), environment=env
        )
        builder.add_dynamic(Freezing(singular=True))
        attributes = {
            "multiplicity": np.full(n_sd, multiplicity),
            "freezing temperature": np.full(n_sd, T_fz),
            "water mass": np.full(n_sd, water_mass),
        }
        products = (IceWaterContent(name="qi"),)
        particulator = builder.build(attributes=attributes, products=products)
        env["T"] = T_fz
        env["RH"] = 1.000001

        # act
        particulator.run(steps=steps)

        # assert
        np.testing.assert_approx_equal(
            actual=np.asarray(particulator.products["qi"].get()),
            desired=n_sd * multiplicity * water_mass / dv,
            significant=7,
        )

    @staticmethod
    @pytest.mark.parametrize("double_precision", (True, False))
    # pylint: disable=too-many-locals
    def test_freeze_time_dependent(backend_class, double_precision, plot=False):
        if backend_class.__name__ == "Numba" and not double_precision:
            pytest.skip()

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
        initial_water_mass = (
            44  # for sign flip (ice water has negative volumes), value does not matter
        )
        d_v = 666  # products use conc., dividing there, multiplying here, value does not matter

        def hgh(t):
            return np.exp(-0.75 * rate * (t - total_time / 4))

        def low(t):
            return np.exp(-1.25 * rate * (t + total_time / 4))

        # Act
        output = {}

        formulae = Formulae(
            particle_shape_and_density="MixedPhaseSpheres",
            heterogeneous_ice_nucleation_rate="Constant",
            constants={"J_HET": rate / immersed_surface_area},
            seed=seed,
        )
        products = (IceWaterContent(name="qi"),)

        for case in cases:
            n_sd = int(number_of_real_droplets // case["N"])
            assert n_sd == number_of_real_droplets / case["N"]
            assert total_time // case["dt"] == total_time / case["dt"]

            key = f"{case['dt']}:{case['N']}"
            output[key] = {"unfrozen_fraction": [], "dt": case["dt"], "N": case["N"]}

            env = Box(dt=case["dt"], dv=d_v)
            builder = Builder(
                n_sd=n_sd,
                backend=backend_class(
                    formulae=formulae, double_precision=double_precision
                ),
                environment=env,
            )
            builder.add_dynamic(Freezing(singular=False))
            attributes = {
                "multiplicity": np.full(n_sd, int(case["N"])),
                "immersed surface area": np.full(n_sd, immersed_surface_area),
                "water mass": np.full(n_sd, initial_water_mass),
            }
            particulator = builder.build(attributes=attributes, products=products)
            env["RH"] = 1.0001
            env["a_w_ice"] = np.nan
            env["T"] = np.nan

            cell_id = 0
            for i in range(int(total_time / case["dt"]) + 1):
                particulator.run(0 if i == 0 else 1)

                ice_mass_per_volume = particulator.products["qi"].get()[cell_id]
                ice_mass = ice_mass_per_volume * d_v
                ice_number = ice_mass / initial_water_mass
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
    pyplot.xlabel("time [s]")
    pyplot.ylabel("unfrozen fraction")
    pyplot.grid()
