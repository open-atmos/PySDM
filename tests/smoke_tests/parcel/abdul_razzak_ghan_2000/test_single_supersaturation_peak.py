# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from matplotlib import pyplot
from PySDM_examples.Abdul_Razzak_Ghan_2000.aerosol import CONSTANTS_ARG
from scipy import signal

from PySDM import Builder, Formulae
from PySDM import products as PySDM_products
from PySDM.backends import CPU
from PySDM.backends.impl_numba.test_helpers import scipy_ode_condensation_solver
from PySDM.dynamics import AmbientThermodynamics, Condensation
from PySDM.environments import Parcel
from PySDM.initialisation import equilibrate_wet_radii
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity
from PySDM.initialisation.spectra import Lognormal
from PySDM.physics import si


@pytest.mark.parametrize(
    "rtol_thd",
    (
        pytest.param(1e-6, marks=pytest.mark.xfail(strict=True)),
        1e-7,
        1e-8,
        1e-9,
    ),
)
@pytest.mark.parametrize("rtol_x", (1e-7,))
@pytest.mark.parametrize("adaptive", (True,))
@pytest.mark.parametrize("scheme", ("PySDM",))
def test_single_supersaturation_peak(
    adaptive, scheme, rtol_x, rtol_thd, plot=False
):  # pylint: disable=too-many-locals
    # arrange
    products = (
        PySDM_products.WaterMixingRatio(unit="g/kg", name="liquid water mixing ratio"),
        PySDM_products.PeakSupersaturation(name="S max"),
        PySDM_products.AmbientRelativeHumidity(name="RH"),
        PySDM_products.ParcelDisplacement(name="z"),
    )
    env = Parcel(
        dt=2 * si.s,
        mass_of_dry_air=1e3 * si.kg,
        p0=1000 * si.hPa,
        initial_water_vapour_mixing_ratio=22.76 * si.g / si.kg,
        w=0.5 * si.m / si.s,
        T0=300 * si.K,
    )
    n_steps = 70
    n_sd = 2
    kappa = 0.4
    spectrum = Lognormal(norm_factor=5000 / si.cm**3, m_mode=50.0 * si.nm, s_geom=2.0)
    formulae = Formulae(constants=CONSTANTS_ARG)
    builder = Builder(backend=CPU(formulae), n_sd=n_sd, environment=env)
    builder.add_dynamic(AmbientThermodynamics())
    builder.add_dynamic(
        Condensation(
            adaptive=adaptive,
            rtol_x=rtol_x,
            rtol_thd=rtol_thd,
        )
    )

    r_dry, concentration = ConstantMultiplicity(spectrum).sample(n_sd)
    v_dry = builder.formulae.trivia.volume(radius=r_dry)
    r_wet = equilibrate_wet_radii(
        r_dry=r_dry, environment=env, kappa_times_dry_volume=kappa * v_dry
    )
    specific_concentration = concentration / builder.formulae.constants.rho_STP
    attributes = {
        "multiplicity": specific_concentration * env.mass_of_dry_air,
        "dry volume": v_dry,
        "kappa times dry volume": kappa * v_dry,
        "volume": builder.formulae.trivia.volume(radius=r_wet),
    }

    particulator = builder.build(attributes, products=products)

    if scheme == "SciPy":
        scipy_ode_condensation_solver.patch_particulator(particulator)

    output = {product.name: [] for product in particulator.products.values()}
    output_attributes = {"volume": tuple([] for _ in range(particulator.n_sd))}

    # act
    for _ in range(n_steps):
        particulator.run(steps=1)
        for product in particulator.products.values():
            value = product.get()
            output[product.name].append(value[0])
        for key, attr in output_attributes.items():
            attr_data = particulator.attributes[key].to_ndarray()
            for drop_id in range(particulator.n_sd):
                attr[drop_id].append(attr_data[drop_id])

    # plot
    for drop_id, volume in enumerate(output_attributes["volume"]):
        pyplot.semilogx(
            particulator.formulae.trivia.radius(volume=np.asarray(volume)) / si.um,
            output["z"],
            color="black",
            label="drop size (bottom axis)",
        )
        pyplot.xlabel("radius [um]")
        pyplot.ylabel("z [m]")
    twin = pyplot.twiny()
    twin.plot(output["S max"], output["z"], label="S max (top axis)")
    twin.plot(np.asarray(output["RH"]) - 1, output["z"], label="ambient RH (top axis)")
    twin.legend(loc="upper center")
    twin.set_xlim(-0.001, 0.0015)
    pyplot.legend(loc="lower right")
    pyplot.grid()
    pyplot.title(f"rtol_thd={rtol_thd}; rtol_x={rtol_x}")
    if plot:
        pyplot.show()

    # assert
    assert signal.argrelextrema(np.asarray(output["RH"]), np.greater)[0].shape[0] == 1
