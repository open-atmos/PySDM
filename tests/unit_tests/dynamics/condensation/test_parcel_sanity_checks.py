""" tests ensuring proper condensation solver operation in some parcel-model based cases """

import numpy as np

import pytest
from matplotlib import pyplot
from scipy import signal

from PySDM import Builder, Formulae, products
from PySDM.backends import CPU, GPU
from PySDM.dynamics import AmbientThermodynamics, Condensation
from PySDM.environments import Parcel
from PySDM.initialisation import discretise_multiplicities, equilibrate_wet_radii
from PySDM.initialisation.sampling import spectral_sampling
from PySDM.initialisation.spectra import Lognormal
from PySDM.physics import si

FORMULAE = Formulae()
SPECTRUM = Lognormal(norm_factor=1e4 / si.mg, m_mode=50 * si.nm, s_geom=1.5)
N_SD = 64
R_DRY, specific_concentration = spectral_sampling.Logarithmic(SPECTRUM).sample(N_SD)
V_DRY = FORMULAE.trivia.volume(radius=R_DRY)
KAPPA = 0.5
CLOUD_RANGE = (0.5 * si.um, 25 * si.um)
PARCEL_CELL = 0


class TestParcelSanityChecks:
    @staticmethod
    @pytest.mark.parametrize(
        "backend_class",
        (
            CPU,
            pytest.param(
                GPU, marks=pytest.mark.xfail(strict=True)
            ),  # TODO #1117 (works with CUDA!)
        ),
    )
    def test_noisy_supersaturation_profiles(backend_class, plot=False):
        """cases found using the README parcel snippet"""
        # arrange
        env = Parcel(
            dt=1 * si.s,
            mass_of_dry_air=1e3 * si.kg,
            p0=1122 * si.hPa,
            initial_water_vapour_mixing_ratio=20 * si.g / si.kg,
            T0=300 * si.K,
            w=2.5 * si.m / si.s,
        )
        output_interval = 2
        output_points = 20

        builder = Builder(backend=backend_class(FORMULAE), n_sd=N_SD, environment=env)
        builder.add_dynamic(AmbientThermodynamics())
        builder.add_dynamic(Condensation())

        r_wet = equilibrate_wet_radii(
            r_dry=R_DRY,
            environment=env,
            kappa_times_dry_volume=KAPPA * V_DRY,
            rtol=1e-3,
        )

        particulator = builder.build(
            attributes={
                "multiplicity": discretise_multiplicities(
                    specific_concentration * env.mass_of_dry_air
                ),
                "dry volume": V_DRY,
                "kappa times dry volume": KAPPA * V_DRY,
                "volume": FORMULAE.trivia.volume(radius=r_wet),
            },
            products=(
                products.PeakSupersaturation(name="S_max", unit="%"),
                products.EffectiveRadius(
                    name="r_eff", unit="um", radius_range=CLOUD_RANGE
                ),
                products.ParticleConcentration(
                    name="n_c_cm3", unit="cm^-3", radius_range=CLOUD_RANGE
                ),
                products.WaterMixingRatio(
                    name="liquid water mixing ratio",
                    unit="g/kg",
                    radius_range=CLOUD_RANGE,
                ),
                products.ParcelDisplacement(name="z"),
            ),
        )

        output = {
            product.name: [product.get()[PARCEL_CELL]]
            for product in particulator.products.values()
        }

        for _ in range(output_points):
            particulator.run(steps=output_interval)
            for product in particulator.products.values():
                output[product.name].append(product.get()[PARCEL_CELL])

        # plot
        fig, axs = pyplot.subplots(1, len(particulator.products) - 1, sharey="all")
        for i, (key, product) in enumerate(particulator.products.items()):
            if key != "z":
                axs[i].plot(output[key], output["z"], marker=".")
                axs[i].set_title(product.name)
                axs[i].set_xlabel(product.unit)
                axs[i].grid()
        axs[0].set_ylabel(particulator.products["z"].unit)
        fig.suptitle(backend_class.__name__)
        if plot:
            pyplot.show()
        else:
            pyplot.clf()

        # assert
        supersaturation_peaks, _ = signal.find_peaks(output["S_max"])
        assert len(supersaturation_peaks) == 1

    @staticmethod
    @pytest.mark.parametrize("update_thd", (True, False))
    @pytest.mark.parametrize("substeps", (1, 2))
    def test_how_condensation_modifies_args(backend_class, update_thd, substeps):
        """asserting that condensation modifies env thd and water_vapour_mixing_ratio only,
        not rhod"""
        # arrange
        env = Parcel(
            dt=1 * si.s,
            mass_of_dry_air=1 * si.mg,
            p0=1000 * si.hPa,
            initial_water_vapour_mixing_ratio=22.2 * si.g / si.kg,
            T0=300 * si.K,
            w=1 * si.m / si.s,
        )
        builder = Builder(n_sd=10, backend=backend_class(), environment=env)
        builder.add_dynamic(AmbientThermodynamics())
        builder.add_dynamic(
            Condensation(
                update_thd=update_thd,
                substeps=substeps,
                adaptive=False,
            )
        )
        particulator = builder.build(
            products=(),
            attributes=builder.particulator.environment.init_attributes(
                kappa=1, r_dry=0.25 * si.um, n_in_dv=1000
            ),
        )

        particulator.dynamics["AmbientThermodynamics"]()

        cell_id = 0
        env = particulator.environment
        pred_rhod_old = env.get_predicted("rhod")[cell_id]
        pred_thd_old = env.get_predicted("thd")[cell_id]
        pred_water_vapour_mixing_ratio_old = env.get_predicted(
            "water_vapour_mixing_ratio"
        )[cell_id]

        env_rhod_old = env["rhod"][cell_id]
        env_thd_old = env["thd"][cell_id]
        env_water_vapour_mixing_ratio_old = env["water_vapour_mixing_ratio"][cell_id]

        # act
        particulator.dynamics["Condensation"]()

        # assert
        assert env_rhod_old == env["rhod"]
        assert env_water_vapour_mixing_ratio_old == env["water_vapour_mixing_ratio"]
        assert env_thd_old == env["thd"]

        assert pred_rhod_old == env.get_predicted("rhod")[cell_id]
        assert (
            pred_water_vapour_mixing_ratio_old
            != env.get_predicted("water_vapour_mixing_ratio")[cell_id]
        )

        if update_thd:
            assert pred_thd_old != env.get_predicted("thd")[cell_id]
        else:
            assert pred_thd_old == env.get_predicted("thd")[cell_id]

    @staticmethod
    def test_zero_initial_delta_liquid_water_mixing_ratio(backend_class):
        # arrange
        env = Parcel(
            dt=1 * si.s,
            mass_of_dry_air=np.nan * si.mg,
            p0=np.nan * si.hPa,
            initial_water_vapour_mixing_ratio=44 * si.g / si.kg,
            T0=np.nan * si.K,
            w=np.nan * si.m / si.s,
        )
        builder = Builder(n_sd=1, backend=backend_class(), environment=env)
        builder.add_dynamic(AmbientThermodynamics())
        builder.add_dynamic(Condensation())

        # act
        _ = builder.build(
            products=(),
            attributes=builder.particulator.environment.init_attributes(
                kappa=1, r_dry=0.25 * si.um, n_in_dv=1000
            ),
        )

        # assert
        assert env.delta_liquid_water_mixing_ratio == 0
