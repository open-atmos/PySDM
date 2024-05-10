""" tests the liquid-water-path computing product for Parcel env """

import numpy as np
from matplotlib import pyplot

from PySDM.products import (
    ParcelLiquidWaterPath,
    LiquidWaterContent,
    AmbientRelativeHumidity,
)
from PySDM.environments import Parcel
from PySDM.dynamics import Condensation, AmbientThermodynamics
from PySDM import Builder
from PySDM.physics import si


def test_parcel_liquid_water_path(
    backend_class, plot=False
):  # pylint: disable=too-many-locals
    # arrange
    n_sd = 1
    n_steps = 32
    dz = 5 * si.m
    dt = 1 * si.s

    env = Parcel(
        dt=dt,
        mass_of_dry_air=1 * si.mg,
        p0=1000 * si.hPa,
        initial_water_vapour_mixing_ratio=22.2 * si.g / si.kg,
        T0=300 * si.K,
        w=dz / dt,
    )

    builder = Builder(
        n_sd=n_sd, backend=backend_class(double_precision=True), environment=env
    )
    builder.add_dynamic(AmbientThermodynamics())
    builder.add_dynamic(Condensation())
    particulator = builder.build(
        attributes=env.init_attributes(
            n_in_dv=np.asarray([1000]), kappa=0.666, r_dry=np.asarray([0.01 * si.um])
        ),
        products=(
            ParcelLiquidWaterPath(
                name="LWP", count_unactivated=True, count_activated=True
            ),
            LiquidWaterContent(name="LWC"),
            AmbientRelativeHumidity(name="RH"),
        ),
    )

    # act
    data = {product: [] for product in particulator.products}
    for _ in range(n_steps):
        particulator.run(steps=1)
        for key, product in particulator.products.items():
            value = product.get()
            if isinstance(value, np.ndarray):
                value = value[0]
            data[key].append(value)
    for k, datum in data.items():
        data[k] = np.asarray(datum)
    cumsum = np.cumsum((data["LWC"] - np.diff(data["LWC"], prepend=0) / 2) * dz)
    t = np.arange(1, len(cumsum) + 1) * dt

    # plot
    pyplot.title(backend_class.__name__)
    pyplot.plot(
        t,
        cumsum,
        label="cumsum((LWC - diff(LWC)/2) * dz)",
        color="black",
        linestyle=":",
    )
    pyplot.plot(t, data["LWP"], label="LWP", color="black")
    pyplot.ylabel("LWP [kg/m^2]")
    pyplot.xlabel("time [s] (values at the end of each timestep)")
    pyplot.ylim(0, 0.016)
    pyplot.legend()
    pyplot.grid()

    twin = pyplot.gca().twinx()
    twin.plot(t, data["RH"], color="red", marker="o")
    twin.set_ylabel("RH [1]", color="red")
    twin.set_ylim(0.98, 1.02)

    if plot:
        pyplot.show()
    else:
        pyplot.clf()

    # assert
    np.testing.assert_allclose(cumsum, data["LWP"], atol=2e-10)
