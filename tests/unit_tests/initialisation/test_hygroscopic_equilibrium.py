# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from matplotlib import pyplot

from PySDM import Formulae, Builder
from PySDM.backends import Numba
from PySDM.initialisation.hygroscopic_equilibrium import (
    equilibrate_wet_radii,
    equilibrate_dry_radii,
)
from PySDM.physics import constants_defaults as const
from PySDM.physics import si
from PySDM.environments import Box


class TestHygroscopicEquilibrium:
    @staticmethod
    @pytest.mark.parametrize("r_dry", (pytest.param(2.4e-09), pytest.param(2.5e-09)))
    @pytest.mark.parametrize(
        "surface_tension",
        ("Constant", "CompressedFilmOvadnevaite", "SzyszkowskiLangmuir"),
    )
    def test_equilibrate_wet_radii(r_dry, surface_tension, plot=False):
        # Arrange
        T = 280.0
        RH = 0.9
        f_org = 0.607
        kappa = 0.356

        class Particulator:  # pylint: disable=too-few-public-methods
            formulae = Formulae(
                surface_tension=surface_tension,
                constants={
                    "sgm_org": 40 * si.mN / si.m,
                    "delta_min": 0.1 * si.nm,
                    "RUEHL_nu_org": 7.47e-05,
                    "RUEHL_A0": 2.5e-19 * si.m**2,
                    "RUEHL_C0": 1e-5,
                    "RUEHL_sgm_min": 40 * si.mN / si.m,
                },
            )

        class Env:  # pylint: disable=too-few-public-methods
            particulator = Particulator()
            thermo = {
                "T": Numba.Storage.from_ndarray(np.full(1, T)),
                "RH": Numba.Storage.from_ndarray(np.full(1, RH)),
            }

            def __getitem__(self, item):
                return self.thermo[item]

        r_dry_arr = np.full(1, r_dry)

        # Plot
        r_wet = np.logspace(np.log(0.9 * r_dry), np.log(10 * si.nm), base=np.e, num=100)
        sigma = Env.particulator.formulae.surface_tension.sigma(
            T,
            Env.particulator.formulae.trivia.volume(r_wet),
            Env.particulator.formulae.trivia.volume(r_dry),
            f_org,
        )
        RH_eq = Env.particulator.formulae.hygroscopicity.RH_eq(
            r_wet, T, kappa, r_dry**3, sigma
        )

        pyplot.plot(r_wet / si.nm, (RH_eq - 1) * 100, label="RH_eq")
        pyplot.axhline((RH - 1) * 100, color="orange", label="RH")
        pyplot.axvline(r_dry / si.nm, label="a", color="red")
        pyplot.axvline(
            Env.particulator.formulae.hygroscopicity.r_cr(
                kappa, r_dry**3, T, const.sgm_w
            )
            / si.nm,
            color="green",
            label="b",
        )
        pyplot.grid()
        pyplot.xscale("log")
        pyplot.xlabel("Wet radius [nm]")
        pyplot.xlim(r_wet[0] / si.nm, r_wet[-1] / si.nm)
        pyplot.ylabel("Equilibrium supersaturation [%]")
        pyplot.legend()
        if plot:
            pyplot.show()
        else:
            pyplot.clf()

        # Act
        env = Env()
        r_wet = equilibrate_wet_radii(
            r_dry=r_dry_arr,
            environment=env,
            kappa_times_dry_volume=Env.particulator.formulae.trivia.volume(r_dry_arr)
            * kappa,
            f_org=np.full_like(r_dry_arr, f_org),
        )

        # Assert
        assert (r_wet >= r_dry_arr).all()

    @staticmethod
    @pytest.mark.parametrize(
        "kappas, temperature, relative_humidity",
        (
            pytest.param([0.0], 300 * si.K, 0.95, marks=pytest.mark.xfail(strict=True)),
            ([0.5, 1.0, 1.5], 300 * si.K, 0.95),
            ([0.5, 1.0, 1.5], 250 * si.K, 0.75),
        ),
    )
    def test_equilibrate_dry_radii(
        kappas, temperature, relative_humidity, backend_instance, plot=False
    ):
        # arrange
        r_wet = np.logspace(-8, -3, num=10)
        builder = Builder(
            environment=Box(dv=np.nan, dt=np.nan),
            backend=backend_instance,
            n_sd=len(r_wet),
        )
        builder.particulator.environment["T"] = temperature
        builder.particulator.environment["RH"] = relative_humidity

        # act
        r_dry = {}
        for kappa in kappas:
            r_dry[kappa] = equilibrate_dry_radii(
                r_wet=r_wet,
                environment=builder.particulator.environment,
                kappa=kappa if kappa is np.ndarray else np.full_like(r_wet, kappa),
            )

        # plot
        for kappa in kappas:
            pyplot.loglog(r_wet, r_dry[kappa], label=f"{kappa=}")
        pyplot.gca().set(
            ylim=(r_wet[0], r_wet[-1]),
            xlabel="wet radius [m]",
            ylabel="dry radius [m]",
            title=f"{temperature=} {relative_humidity=}",
        )
        pyplot.grid()
        pyplot.legend()
        if plot:
            pyplot.show()
        else:
            pyplot.clf()

        # assert
        assert (np.diff(kappas) > 0).all()
        kappa_prev = None
        for kappa in kappas:
            assert (r_dry[kappa] < r_wet).all()

            if kappa_prev is not None:
                assert (r_dry[kappa] < r_dry[kappa_prev]).all()
            kappa_prev = kappa

            np.testing.assert_allclose(
                rtol=1e-5,
                desired=r_wet,
                actual=equilibrate_wet_radii(
                    r_dry=r_dry[kappa],
                    kappa_times_dry_volume=kappa
                    * builder.particulator.formulae.trivia.volume(r_dry[kappa]),
                    environment=builder.particulator.environment,
                ),
            )
