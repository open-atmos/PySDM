""" comparing kappa-Koehler wet radius equilibration in PySDM and PartMC
  (based on PyPartMC-examples notebook by Zach D'Aquino) """

# pylint: disable=missing-function-docstring,no-member
import platform
from collections import namedtuple

import numpy as np
import pytest
from matplotlib import pyplot

from PySDM import Builder
from PySDM.backends import CPU
from PySDM.environments import Box
from PySDM.initialisation import equilibrate_wet_radii
from PySDM.initialisation.spectra import Lognormal
from PySDM.physics import si

if platform.architecture()[0] == "64bit":
    import PyPartMC

linestyles = {"PyPartMC": "dashed", "PySDM": "dotted"}
x_unit = si.um
y_unit = 1 / si.cm**3


def pysdm(dry_diam, temp, rel_humid, kpa):
    r_dry = dry_diam / 2
    environment = Box(dt=np.nan, dv=np.nan)
    _ = Builder(n_sd=0, backend=CPU(), environment=environment)
    environment["T"] = temp
    environment["RH"] = rel_humid
    kappa_times_dry_volume = kpa * (np.pi / 6) * dry_diam**3
    return 2 * equilibrate_wet_radii(
        r_dry=r_dry,
        environment=environment,
        kappa_times_dry_volume=kappa_times_dry_volume,
    )


def pypartmc(dry_diam, temp, rel_humid, kpa):
    env_state = PyPartMC.EnvState(
        {
            "rel_humidity": rel_humid,
            "latitude": 0.0,
            "longitude": 0.0,
            "altitude": 0.0,
            "start_time": 0.0,
            "start_day": 0,
        }
    )

    env_state.set_temperature(temp)

    aero_data = PyPartMC.AeroData(
        (
            {"H2O": [1000 * si.kg / si.m**3, 0, 18e-3 * si.kg / si.mol, 0]},
            {"XXX": [np.nan * si.kg / si.m**3, 0, np.nan * si.kg / si.mol, kpa]},
        )
    )

    dry_volumes = (np.pi / 6) * dry_diam**3
    aero_particles = [
        PyPartMC.AeroParticle(aero_data, np.array([0, 1]) * volume)
        for volume in dry_volumes
    ]

    for aero_particle in aero_particles:
        PyPartMC.condense_equilib_particle(env_state, aero_data, aero_particle)

    wet_volumes = [np.sum(particle.volumes) for particle in aero_particles]
    wet_diameters = ((6 / np.pi) * np.asarray(wet_volumes)) ** (1 / 3)

    return wet_diameters


@pytest.mark.skipif(
    platform.architecture()[0] != "64bit", reason="binary package availability"
)
@pytest.mark.parametrize("kappa", (0.1, 1))
@pytest.mark.parametrize("temperature", (300 * si.K,))
@pytest.mark.parametrize("relative_humidity", (0.5, 0.75, 0.99))
def test_dry_wet_equilibration(kappa, temperature, relative_humidity, plot=False):
    # arrange
    models = {"PySDM": pysdm, "PyPartMC": pypartmc}

    Mode = namedtuple("Mode", ("norm_factor", "m_mode", "s_geom"))
    modes = (
        Mode(norm_factor=50000 / si.cm**3, m_mode=0.9 * si.um, s_geom=1.3),
        Mode(norm_factor=80000 / si.cm**3, m_mode=5.8 * si.um, s_geom=2),
    )

    dry_diameters = np.logspace(-0.5, 1.5, 100) * si.um
    dn_dlndiam_sum = np.zeros_like(dry_diameters)
    for mode in modes:
        dn_dlndiam_sum += Lognormal(*mode).size_distribution(dry_diameters)
    wet_diameters = {}

    # act
    for model, func in models.items():
        wet_diameters[model] = func(
            dry_diameters, temp=temperature, rel_humid=relative_humidity, kpa=kappa
        )

    # plot
    pyplot.plot(
        dry_diameters / x_unit, dn_dlndiam_sum / y_unit, label="(dry)", linewidth=3
    )
    for model, func in models.items():
        pyplot.plot(
            wet_diameters[model] / x_unit,
            dn_dlndiam_sum / y_unit,
            label=f"(wet) Model={model}",
            linestyle=linestyles[model],
            marker=".",
        )

    pyplot.title(f"RH={relative_humidity} kappa={kappa}")
    pyplot.xlabel(r"Diameter, $D_p$ [$\mu m$]")
    pyplot.ylabel("$dN/dD$ [$cm^{-3}$]")
    pyplot.xscale("log")
    pyplot.grid()
    pyplot.legend()
    if plot:
        pyplot.show()

    # assert
    np.testing.assert_allclose(
        wet_diameters["PySDM"], wet_diameters["PyPartMC"], rtol=1e-1
    )
