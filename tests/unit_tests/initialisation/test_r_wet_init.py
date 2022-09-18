# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from matplotlib import pyplot

from PySDM import Formulae
from PySDM.backends import CPU
from PySDM.initialisation import equilibrate_wet_radii
from PySDM.physics import constants_defaults as const
from PySDM.physics import si


@pytest.mark.parametrize("r_dry", [pytest.param(2.4e-09), pytest.param(2.5e-09)])
# pylint: disable=unused-argument,redefined-outer-name
def test_r_wet_init(r_dry, plot=False):
    # Arrange
    T = 280
    RH = 0.9
    f_org = 0.607
    kappa = 0.356

    class Particulator:  # pylint: disable=too-few-public-methods
        formulae = Formulae(
            surface_tension="CompressedFilmOvadnevaite",
            constants={"sgm_org": 40 * si.mN / si.m, "delta_min": 0.1 * si.nm},
        )

    class Env:  # pylint: disable=too-few-public-methods
        particulator = Particulator()
        thermo = {
            "T": CPU.Storage.from_ndarray(np.full(1, T)),
            "RH": CPU.Storage.from_ndarray(np.full(1, RH)),
        }

        def __getitem__(self, item):
            return self.thermo[item]

    r_dry_arr = np.full(1, r_dry)

    # Plot
    r_wet = np.logspace(np.log(0.9 * r_dry), np.log(10 * si.nm), base=np.e, num=100)
    sigma = Env.particulator.formulae.surface_tension.sigma(
        np.nan,
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
        Env.particulator.formulae.hygroscopicity.r_cr(kappa, r_dry**3, T, const.sgm_w)
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

    # Act & Assert
    equilibrate_wet_radii(
        r_dry=r_dry_arr,
        environment=Env(),
        kappa_times_dry_volume=Env.particulator.formulae.trivia.volume(r_dry_arr)
        * kappa,
        f_org=np.full(1, f_org),
    )
