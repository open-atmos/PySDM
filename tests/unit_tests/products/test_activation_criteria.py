"""tests four different ways of diagnosing activable fraction in a Parcel environment"""

import pytest
import numpy as np
from matplotlib import pyplot

from PySDM.products import ActivableFraction, AmbientRelativeHumidity, PeakSaturation
from PySDM.environments import Parcel
from PySDM.dynamics import Condensation, AmbientThermodynamics
from PySDM import Builder
from PySDM.physics import si
from PySDM.backends import CPU, GPU
from PySDM.initialisation.spectra import Lognormal
from PySDM.initialisation.sampling import spectral_sampling


@pytest.mark.parametrize(
    "backend",
    (
        pytest.param(GPU(), marks=pytest.mark.xfail(strict=True)),
        CPU(override_jit_flags={"parallel": False}),
    ),
)
def test_activation_criteria(backend, plot=False):
    # arrange
    n_sd = 1000
    n_steps = 50
    dz = 5 * si.m
    dt = 2 * si.s

    spectrum = Lognormal(norm_factor=1e4 / si.mg, m_mode=50 * si.nm, s_geom=1.5)
    r_dry, specific_concentration = spectral_sampling.ConstantMultiplicity(
        spectrum
    ).sample(n_sd)

    builder = Builder(
        n_sd=n_sd,
        backend=backend,
        environment=Parcel(
            dt=dt,
            mass_of_dry_air=100 * si.kg,
            p0=1000 * si.hPa,
            initial_water_vapour_mixing_ratio=22 * si.g / si.kg,
            T0=300 * si.K,
            w=dz / dt,
        ),
    )
    builder.add_dynamic(AmbientThermodynamics())
    builder.add_dynamic(Condensation())
    particulator = builder.build(
        attributes=builder.particulator.environment.init_attributes(
            n_in_dv=specific_concentration
            * builder.particulator.environment.mass_of_dry_air,
            kappa=0.666,
            r_dry=r_dry,
        ),
        products=(
            PeakSaturation(name="S_max"),
            AmbientRelativeHumidity(name="RH"),
            ActivableFraction(
                name="AF1 (r_wet > r_cri(T))",
                filter_attr="wet to critical volume ratio",
            ),
            ActivableFraction(
                name="AF2 (r_wet > r_cri(T0))",
                filter_attr="wet to critical volume ratio neglecting temperature variations",
            ),
            ActivableFraction(
                name="AF3 (S_crit(T) < S_max)", filter_attr="critical saturation"
            ),
            ActivableFraction(
                name="AF4 (S_crit(T0) < S_max)",
                filter_attr="critical saturation neglecting temperature variations",
            ),
        ),
    )

    # act
    data = {product: [] for product in particulator.products}
    S_max = np.nan
    for i in range(n_steps):
        particulator.run(steps=1)
        for key, product in particulator.products.items():
            if key == "S_max" and i > 0:
                S_max = np.nanmax([data["S_max"][-1], S_max])
            value = product.get(**({} if key.startswith("wet") else {"S_max": S_max}))
            if isinstance(value, np.ndarray):
                value = value[0]
            data[key].append(value)

    # plot
    pyplot.title(backend.__class__.__name__)
    t = np.arange(1, n_steps + 1) * dt
    for k, datum in data.items():
        if k.startswith("AF"):
            pyplot.plot(t, datum, label=k, marker=k[2], markersize=10)
    pyplot.xlabel("time [s] (values at the end of each timestep)")
    pyplot.ylabel("activated fraction [1]")
    pyplot.ylim(0.4, 0.65)
    pyplot.legend()
    pyplot.grid()

    twin = pyplot.gca().twinx()
    twin.plot(t, np.asarray(data["RH"]) * 100 - 100, color="red", marker="o")
    twin.set_ylabel("supersaturation [%]", color="red")
    twin.set_ylim(-1.5, 0.5)
    twin.set_yticks(np.linspace(-1.5, 0.5, 5, endpoint=True))

    if plot:
        pyplot.show()
    else:
        pyplot.clf()

    # assert
    for k, datum in data.items():
        if k.startswith("AF"):
            assert datum[0] == 0
            assert 0.4 < datum[-1] <= 0.65
            assert np.all(np.diff(datum[40:]) <= 0)
    assert (
        data["AF1 (r_wet > r_cri(T))"][-1]
        < data["AF3 (S_crit(T) < S_max)"][-1]
        < data["AF4 (S_crit(T0) < S_max)"][-1]
    )
