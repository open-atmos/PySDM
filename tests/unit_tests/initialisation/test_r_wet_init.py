from PySDM.initialisation import r_wet_init
from PySDM.physics import Formulae, si, constants as const
from PySDM.backends import CPU
import numpy as np
import pytest
from matplotlib import pylab


@pytest.mark.parametrize('r_dry', [
    pytest.param(2.4e-09),
    pytest.param(2.5e-09)
])
def test_r_wet_init(r_dry, plot=False):
    # Arrange
    T = 280
    RH = .9
    f_org = .607
    kappa = .356

    class Core:
        formulae = Formulae(surface_tension='CompressedFilm')

    class Env:
        core = Core()
        thermo = {
            'T': CPU.Storage.from_ndarray(np.full(1, T)),
            'RH': CPU.Storage.from_ndarray(np.full(1, RH))
        }

        def __getitem__(self, item):
            return self.thermo[item]

    r_dry_arr = np.full(1, r_dry)

    # Plot
    if plot:
        r_wet = np.logspace(np.log(.9*r_dry), np.log(10 * si.nm), base=np.e, num=100)
        sigma = Env.core.formulae.surface_tension.sigma(np.nan,
                                                        Env.core.formulae.trivia.volume(r_wet),
                                                        Env.core.formulae.trivia.volume(r_dry),
                                                        f_org)
        RH_eq = Env.core.formulae.hygroscopicity.RH_eq(r_wet, T, kappa, r_dry**3, sigma)
        pylab.plot(
            r_wet / si.nm,
            (RH_eq - 1) * 100,
            label='RH_eq'
        )
        pylab.axhline((RH-1)*100, color='orange', label='RH')
        pylab.axvline(r_dry / si.nm, label='a', color='red')
        pylab.axvline(
            Env.core.formulae.hygroscopicity.r_cr(kappa, r_dry**3, T, const.sgm_w) / si.nm,
            color='green', label='b'
        )
        pylab.grid()
        pylab.xscale('log')
        pylab.xlabel('Wet radius [nm]')
        pylab.xlim(r_wet[0] / si.nm, r_wet[-1] / si.nm)
        pylab.ylabel('Equilibrium supersaturation [%]')
        pylab.legend()
        pylab.show()

    # Act & Assert
    r_wet_init(
        r_dry=r_dry_arr,
        environment=Env(),
        kappa_times_dry_volume=Env.core.formulae.trivia.volume(r_dry_arr) * kappa,
        f_org=np.full(1, f_org)
    )
